"""Main inference loop: cloud LingBot-VA policy → G1 arms + grippers.

Precondition: the robot must already be in 'ai' motion mode, set via the
Unitree app before starting this script — this code does not switch modes.

Pipeline:
  1. DDS setup, start arm + gripper controller threads, then camera client.
  2. Move arms to INIT_POSE_READY; close then open grippers.
  3. STANDBY: hold the ready pose until the operator presses Enter.
  4. Switch arm kp from the stiff init value to the softer inference value.
  5. Connect to the cloud WebSocket server, send reset({prompt}).
  6. Loop per chunk:
       a. infer({obs, prompt}) → action [16, 2, 16].
       b. Iterate frames × sub_steps, dispatch targets at 30 Hz.
       c. Every CAPTURE_EVERY sub-steps snap a keyframe — 8 per chunk,
          except the first chunk where frame 0 is skipped → 4 keyframes.
       d. compute_kv_cache: hand the keyframes + executed action back so
          the server's KV cache reflects reality.
       e. Request the next chunk and repeat.

Usage (run directly from inside the lingbot_g1_client/ directory):
    python main.py \\
        --iface enp0s31f6 \\
        --server-host 1.2.3.4 \\
        --server-port 29056 \\
        --prompt "pick up the pink object and place it on the blue cross mark"
"""

import argparse
import logging
import queue
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_client.arm_controller import ArmController, INIT_POSE_READY
from g1_client.gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX
from g1_client.camera_client import CameraClient
from g1_client.policy_client import PolicyClient


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("lingbot_g1.main")


# Cadence:
#   frame_chunk_size = latent frames per chunk (server-defined; 2 for the
#                      original g1 model, 4 for g1_500step — DERIVED from the
#                      returned action tensor's axis 1, not hard-coded)
#   action_per_frame = sub-steps per latent frame (derived from axis 2)
#   sub-step rate    = ~30 Hz (33.3 ms each)
#   capture cadence  = every CAPTURE_EVERY sub-steps → keyframes scale with
#                      the chunk size automatically
CAPTURE_EVERY = 4
# Nominal values for the original g1 model — kept only so a horizon change
# from the server is logged as a notice, not silently accepted.
FRAME_CHUNK = 2
SUBSTEPS_PER_FRAME = 16

# Action tensor layout — channels along axis 0 of action[16, FRAME_CHUNK, SUBSTEPS_PER_FRAME]:
#   [0:7]   = left arm  (order matches arm_controller.ARM_JOINTS[0:7]:
#                        L pitch/roll/yaw/elbow/wristR/wristP/wristY)
#   [7:14]  = right arm (mirrored, matches ARM_JOINTS[7:14])
#   [14]    = left gripper  (rad in [GRIPPER_MIN, GRIPPER_MAX])
#   [15]    = right gripper
ARM_CHANNELS = slice(0, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15


def log_chunk_summary(chunk_id, action):
    """One-line summary of the per-joint range for a chunk so we can see what
    the model is asking for before it streams to the robot."""
    arm = action[ARM_CHANNELS]
    grip_l = action[LEFT_GRIPPER_CHANNEL]
    grip_r = action[RIGHT_GRIPPER_CHANNEL]
    arm_min = arm.min(axis=(1, 2))
    arm_max = arm.max(axis=(1, 2))
    log.info(f"[chunk {chunk_id}] arm joint ranges (rad):")
    joint_names = [
        "L_pitch", "L_roll ", "L_yaw  ", "L_elbow", "L_wrR  ", "L_wrP  ", "L_wrY  ",
        "R_pitch", "R_roll ", "R_yaw  ", "R_elbow", "R_wrR  ", "R_wrP  ", "R_wrY  ",
    ]
    for i, name in enumerate(joint_names):
        log.info(f"   {name}  min={arm_min[i]:+.3f}  max={arm_max[i]:+.3f}")
    log.info(f"[chunk {chunk_id}] gripper L: [{grip_l.min():.2f}, {grip_l.max():.2f}] "
             f"R: [{grip_r.min():.2f}, {grip_r.max():.2f}]")


def execute_chunk(action, arm_ctrl, grip_ctrl, cam_client, is_first_chunk,
                  substep_dt=1.0 / 30.0, verbose_substeps=True):
    """Dispatch one [16, F, S] action chunk to arms+grippers at 30 Hz.

    F (frames/chunk) and S (sub-steps/frame) are taken from the action
    tensor, not assumed — the server's horizon (e.g. F=2 for g1, F=4 for
    g1_500step) can change without a client edit.

    Returns (keyframes, stats). A keyframe is captured every CAPTURE_EVERY
    sub-steps, so the count scales with the chunk: F*S/CAPTURE_EVERY for
    chunks 1+, (F-1)*S/CAPTURE_EVERY for chunk 0 (frame 0 skipped). Each is
    the camera-only image dict (no prompt key), shipped back in
    compute_kv_cache. stats — per-substep pacing profile.

    For the first chunk only, skip frame 0's sub-steps (the model treats
    them as 'observe the start position').
    """
    assert action.ndim == 3 and action.shape[0] == 16, \
        f"Unexpected action shape {action.shape} (want (16, F, S))"
    n_frames = action.shape[1]
    n_sub = action.shape[2]
    if (n_frames, n_sub) != (FRAME_CHUNK, SUBSTEPS_PER_FRAME):
        log.info(f"Server horizon is {n_frames}x{n_sub} sub-steps/chunk "
                 f"(nominal {FRAME_CHUNK}x{SUBSTEPS_PER_FRAME}) — adapting; "
                 f"keyframes/chunk now scale accordingly")

    keyframes = []
    start_frame = 1 if is_first_chunk else 0

    n_substeps = 0
    n_overrun = 0
    max_elapsed = 0.0
    cam_total = 0.0

    for t in range(start_frame, n_frames):
        for f in range(n_sub):
            # Abort fast if the arm's publish thread died — otherwise we keep
            # dispatching targets nobody is sending, with arm_sdk still latched.
            if arm_ctrl.faulted():
                raise RuntimeError("ArmController control thread faulted — aborting")
            tic = time.time()

            cmd = action[:, t, f]
            arm_q = cmd[ARM_CHANNELS].astype(np.float64)
            l_grip = float(np.clip(cmd[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX))
            r_grip = float(np.clip(cmd[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX))

            if verbose_substeps:
                # Pretty-print the target the controller will see this tick.
                # Format: L_arm(7 floats) | R_arm(7 floats) | L_grip R_grip
                l_arm_str = " ".join(f"{x:+.3f}" for x in arm_q[:7])
                r_arm_str = " ".join(f"{x:+.3f}" for x in arm_q[7:])
                log.info(f"  t={t} f={f:2d} | L=[{l_arm_str}] R=[{r_arm_str}] "
                         f"| g=({l_grip:.2f},{r_grip:.2f})")

            arm_ctrl.set_arm_target(arm_q)
            grip_ctrl.set_targets(l_grip, r_grip)

            # Snap a keyframe at the end of each CAPTURE_EVERY-step window
            # (f=3,7,11,15 — late in the window so motion has had time to settle).
            if (f + 1) % CAPTURE_EVERY == 0:
                try:
                    cap_tic = time.time()
                    keyframes.append(cam_client.get_obs_images())
                    cam_total += time.time() - cap_tic
                except Exception as e:
                    log.warning(f"Camera capture failed at frame={t} sub={f}: {e}")

            elapsed = time.time() - tic
            n_substeps += 1
            max_elapsed = max(max_elapsed, elapsed)
            if elapsed > substep_dt:
                n_overrun += 1
            sleep = substep_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    stats = {
        "n_substeps": n_substeps,
        "n_overrun": n_overrun,
        "max_substep_s": max_elapsed,
        "keyframe_cam_s": cam_total,
    }
    return keyframes, stats


# ---------- pipeline stages ----------

def _setup_arm(args) -> ArmController:
    log.info("Starting ArmController")
    arm = ArmController(publish_hz=50.0, velocity_limit=args.velocity_limit)
    arm.start()
    return arm


def _setup_gripper() -> GripperController:
    log.info("Starting GripperController")
    grip = GripperController(publish_hz=200.0)
    grip.start()
    return grip


def _setup_camera(args) -> CameraClient:
    log.info("Starting CameraClient")
    return CameraClient(host=args.image_server)


def _initialize_pose(arm, grip, args) -> None:
    """Ramp to INIT_POSE_READY, then close→open grippers, then settle."""
    log.info(f"Moving arms to ready pose over {args.init_duration:.1f}s "
             f"(velocity_limit={args.velocity_limit} rad/s)")
    arm.move_to_pose(INIT_POSE_READY,
                     duration=args.init_duration,
                     velocity_limit=args.velocity_limit)
    half = args.gripper_init_duration / 2
    log.info(f"Closing grippers to {GRIPPER_MIN} over {half:.1f}s")
    grip.move_to_targets(GRIPPER_MIN, GRIPPER_MIN, duration=half)
    log.info(f"Opening grippers to ({args.init_gripper_left}, {args.init_gripper_right}) "
             f"over {half:.1f}s")
    grip.move_to_targets(args.init_gripper_left,
                         args.init_gripper_right,
                         duration=half)
    log.info("Init complete.")
    time.sleep(args.settle_duration)
    log.info("Arms settled at ready pose.")


def _wait_for_operator(args) -> None:
    """Block on Enter so the operator can stage the scene. No-op with --auto-start.

    Arm + gripper publish threads keep streaming the current target during the
    wait, so the robot stays locked at INIT_POSE_READY.
    """
    if args.auto_start:
        return
    log.info("===============================================================")
    log.info("STANDBY: arms locked at ready pose.")
    log.info("Set up the scene (place pink object, blue cross mark, etc.)")
    log.info("Press [Enter] to connect to the policy server and start.")
    log.info("Press [Ctrl+C] at any time to abort safely.")
    log.info("===============================================================")
    try:
        input("")
    except EOFError:
        log.info("EOF on stdin — proceeding without prompt")


def _connect_policy(args) -> PolicyClient:
    log.info(f"Connecting to policy server ws://{args.server_host}:{args.server_port}")
    policy = PolicyClient(host=args.server_host, port=args.server_port)
    log.info(f"Server metadata: {policy.get_server_metadata()}")
    return policy


def _fmt_infer_timing(label, t):
    """One-line breakdown of a PolicyClient.last_timing dict."""
    if t is None:
        return f"{label}: <no timing>"
    return (f"{label}: total={t['total_s']*1e3:7.1f}ms "
            f"(pack={t['pack_s']*1e3:5.1f} send={t['send_s']*1e3:5.1f} "
            f"wait_recv={t['wait_recv_s']*1e3:7.1f} unpack={t['unpack_s']*1e3:5.1f}) "
            f"up={t['bytes_sent']/1024:.0f}KiB down={t['bytes_recv']/1024:.0f}KiB")


def _async_step_worker(policy, prev_kf, prev_act, cur, out_q):
    """Branch B: the single blocking async_step request, run on a daemon
    thread so it overlaps Branch A (execute_chunk on the main thread).

    Sends the PREVIOUS chunk's keyframes + the just-executed action to be
    grounded, plus the chunk executing right now (for the server's FDM
    "imagine the executing chunk" pass). state/executing_action are passed
    through verbatim — the server runs its own preprocess_action.
    """
    try:
        resp = policy.infer({
            "async_step": True,
            "obs": prev_kf,           # K_{n-1}
            "state": prev_act,        # a_{n-1}, verbatim
            "executing_action": cur,  # C_n,    verbatim
        })
        out_q.put(("ok", resp))
    except BaseException as e:  # surface daemon failure to the main thread
        out_q.put(("err", e))


def _run_inference_loop(arm, grip, cam, policy, args) -> None:
    """Asynchronous, FDM-grounded loop (Algorithm 2).

    reset → cold_start (TWO chunks) → the one non-overlapped chunk C0 →
    steady loop where each cycle overlaps the async_step request (daemon
    thread) with execute_chunk (main thread). compute_kv_cache is NOT used
    in this path — grounding folds into async_step.
    """
    log.info(f"Reset with prompt: {args.prompt!r}")
    policy.reset(args.prompt)

    # Cold start: ONE init frame in, TWO chunks back (C0 to execute now, C1
    # held to execute next so the first async_step can ground C0 in parallel
    # — a one-chunk cold start would force C0 to be grounded twice).
    log.info("Cold start: sending one init frame, expecting two chunks")
    init = cam.get_obs(args.prompt)
    resp = policy.infer({"cold_start": True, "obs": init})
    C0, C1 = resp["action"], resp["action1"]
    log.info(f"Cold start returned C0 {C0.shape}, C1 {C1.shape}")
    log_chunk_summary(0, C0)

    # The ONE non-overlapped chunk. C0 ran is_first_chunk=True → frame 0
    # skipped → 4 keyframes. Nothing to overlap it against yet.
    exec_tic = time.time()
    K0, st = execute_chunk(C0, arm, grip, cam, is_first_chunk=True,
                           substep_dt=1.0 / args.substep_hz,
                           verbose_substeps=not args.quiet_substeps)
    log.info(f"[cycle 0] C0 execute={time.time()-exec_tic:6.3f}s "
             f"keyframes={len(K0)} (expect 4)")

    prev_kf, prev_act = K0, C0
    cur = C1  # the chunk to execute next

    for n in range(1, args.max_chunks + 1):
        out_q = queue.Queue()
        th = threading.Thread(
            target=_async_step_worker,
            args=(policy, prev_kf, prev_act, cur, out_q),
            daemon=True, name=f"async_step-{n}")
        th.start()  # Branch B fires; do NOT block main on it

        # Branch A: execute the current chunk at 30 Hz on the main thread.
        # Steady chunks run is_first_chunk=False → 8 keyframes.
        exec_tic = time.time()
        Kn, st = execute_chunk(cur, arm, grip, cam, is_first_chunk=False,
                               substep_dt=1.0 / args.substep_hz,
                               verbose_substeps=not args.quiet_substeps)
        exec_s = time.time() - exec_tic

        status, payload = out_q.get()  # join: block until the request returns
        th.join()
        if status == "err":
            raise payload
        next_chunk = payload["action"]

        log.info(
            f"[cycle {n}] execute={exec_s:6.3f}s "
            f"(motion ~{st['n_substeps']/args.substep_hz:.2f}s, "
            f"overruns={st['n_overrun']}/{st['n_substeps']}, "
            f"keyframes={len(Kn)})  "
            + _fmt_infer_timing("async_step", policy.last_timing))
        log_chunk_summary(n, next_chunk)

        # Rotate: this cycle's keyframes/action become next cycle's "previous"
        # to ground; next_chunk becomes what we execute next.
        prev_kf, prev_act, cur = Kn, cur, next_chunk


def _cleanup(arm, grip, cam, policy) -> None:
    """Release every resource we may have acquired. Each step is isolated in
    its own try/except BaseException so a KeyboardInterrupt (e.g. a second
    Ctrl+C during shutdown) cannot skip later steps — most critically, it
    cannot skip disable_arm_sdk, which is the only thing returning arm
    authority to the locomotion service.

    BaseException (not Exception) is required: KeyboardInterrupt inherits
    from BaseException, so a plain `except Exception` would let a second
    Ctrl+C escape and abort the rest of cleanup. SystemExit is also caught
    by this same guard, which is the intended behaviour for shutdown.
    """
    log.info("Shutting down — releasing arm_sdk")
    # Stop publish loop FIRST so disable_arm_sdk has exclusive access to
    # self.cmd / self.pub. If the join is interrupted by Ctrl+C we still
    # fall through to disable_arm_sdk: the _cmd_lock makes a concurrent
    # write CRC-safe, and disable_arm_sdk's own finally guarantees q=0.
    try:
        arm.stop()
    except BaseException as e:
        log.warning(f"arm.stop() failed: {e}")
    try:
        arm.disable_arm_sdk()
    except BaseException as e:
        log.warning(f"disable_arm_sdk failed: {e}")
    if grip is not None:
        try:
            grip.stop()
        except BaseException as e:
            log.warning(f"grip.stop() failed: {e}")
    if cam is not None:
        try:
            cam.close()
        except BaseException as e:
            log.warning(f"cam.close() failed: {e}")
    if policy is not None:
        try:
            policy.close()
        except BaseException as e:
            log.warning(f"policy.close() failed: {e}")


def run(args):
    log.info(f"Initializing DDS on {args.iface}")
    ChannelFactoryInitialize(0, args.iface)

    # arm.start() is outside the try: if it fails, arm_sdk was never enabled
    # so no cleanup is needed. Everything after this point is wrapped.
    arm = _setup_arm(args)

    grip = None
    cam = None
    policy = None
    try:
        grip = _setup_gripper()
        cam = _setup_camera(args)
        _initialize_pose(arm, grip, args)
        _wait_for_operator(args)
        log.info(f"Switching arm kp to inference value: {args.inference_kp_arm}")
        arm.set_arm_kp(args.inference_kp_arm)
        policy = _connect_policy(args)
        _run_inference_loop(arm, grip, cam, policy, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--server-host", required=True, help="Cloud GPU server hostname or IP")
    p.add_argument("--server-port", type=int, default=29056, help="Cloud server WebSocket port")
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--prompt", default="pick up the pink object and place it on the blue cross mark")
    p.add_argument("--max-chunks", type=int, default=20,
                   help="How many action chunks to execute before stopping")
    p.add_argument("--video-guidance", type=float, default=5.0)
    p.add_argument("--action-guidance", type=float, default=1.0)

    # ---- Safety / motion limits ----
    p.add_argument("--velocity-limit", type=float, default=8.0,
                   help="rad/s velocity cap on the per-tick motion clamp, applied "
                        "for both the init move and model-driven motion (default 8.0)")
    p.add_argument("--inference-kp-arm", type=float, default=80.0,
                   help="kp applied to shoulder/elbow joints once inference starts "
                        "(default 80; init/standby uses the controller's default kp_arm=150)")
    p.add_argument("--init-duration", type=float, default=2.0,
                   help="Seconds for the arm to ramp from current pose to ready pose")
    p.add_argument("--gripper-init-duration", type=float, default=1.0,
                   help="Total seconds for the gripper close→open init gesture "
                        "(split evenly across the two phases, default 1.0)")
    p.add_argument("--settle-duration", type=float, default=1.0,
                   help="Seconds to wait after the init move so the arm settles "
                        "before entering standby (default 1.0)")
    p.add_argument("--init-gripper-left", type=float, default=5.0,
                   help="Open position after the close→open init sequence (0..5.4, default 5.0)")
    p.add_argument("--init-gripper-right", type=float, default=5.0,
                   help="Open position after the close→open init sequence (0..5.4, default 5.0)")
    p.add_argument("--substep-hz", type=float, default=30.0,
                   help="Action sub-step dispatch rate (default 30 Hz). Lower this "
                        "to slow down the whole chunk if motion still looks too fast.")
    p.add_argument("--quiet-substeps", action="store_true",
                   help="Suppress per-substep target printing (only chunk summaries).")
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start inference immediately.")
    args = p.parse_args()

    run(args)


if __name__ == "__main__":
    main()
