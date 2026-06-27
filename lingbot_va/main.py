"""Main inference loop: cloud LingBot-VA policy → G1 arms + grippers.

Precondition: the robot must already be in 'ai' motion mode, set via the
Unitree app before starting this script — this code does not switch modes.

Pipeline:
  1. DDS setup, start arm + gripper controller threads, then camera client.
  2. Move arms to INIT_POSE_READY; close then open grippers.
  3. STANDBY: hold the ready pose until the operator presses Enter.
  4. Switch arm kp from the stiff init value to the softer inference value.
  5. Connect to the cloud WebSocket server, send reset({prompt}).
  6. cold_start → ONE chunk C0 (Algorithm 2, single-chunk).
  7. Overlapped loop per chunk, from cycle 0 (Branch A ‖ Branch B):
       A. execute the current chunk at 30 Hz; every CAPTURE_EVERY
          sub-steps snap a keyframe — cycle 0 skips frame 0 → 4
          keyframes, cycle n≥1 → 8.
       B. on a daemon thread, async_step({executing_action:C_n
          [, obs:K_{n-1}, state:a_{n-1}]}) → next chunk. Cycle 0 sends
          no obs/state — the server skips grounding for C0, FDM-imagines
          C0 (z_0 pinned), predicts C1. Cycle n≥1 grounds C_{n-1}'s
          reality, FDM-imagines C_n, predicts C_{n+1}.
       Join B, rotate, repeat.

Usage (run from the repo root):
    python lingbot_va/main.py \\
        --iface enp0s31f6 \\
        --server-host 1.2.3.4 \\
        --server-port 29056 \\
        --prompt "pick up the pink object and place it on the blue cross mark"
"""

import argparse
import logging
import os
import queue
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import g1_client

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
                cap_tic = time.time()
                try:
                    kf = cam_client.get_obs_images()
                except Exception as e:
                    # Fail loud: a swallowed capture shortens the keyframe
                    # list, which silently desyncs the server's grounding/KV
                    # for the active horizon (no error, just drift). Abort so
                    # run()'s finally releases arm_sdk instead.
                    raise RuntimeError(
                        f"Camera capture failed at frame={t} sub={f}: {e!r} — "
                        f"aborting rather than shipping a short keyframe list "
                        f"(silent server desync)") from e
                keyframes.append(kf)
                cam_total += time.time() - cap_tic

            elapsed = time.time() - tic
            n_substeps += 1
            max_elapsed = max(max_elapsed, elapsed)
            if elapsed > substep_dt:
                n_overrun += 1
            sleep = substep_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    # Defense-in-depth: the keyframe count is a load-bearing wire contract
    # (server grounding expects exactly this many for the active horizon).
    # Catch any drift — misconfigured CAPTURE_EVERY, a non-divisible S, etc.
    expected_kf = (n_frames - start_frame) * (n_sub // CAPTURE_EVERY)
    if len(keyframes) != expected_kf:
        raise RuntimeError(
            f"keyframe count {len(keyframes)} != expected {expected_kf} "
            f"(F={n_frames} S={n_sub} start_frame={start_frame} "
            f"CAPTURE_EVERY={CAPTURE_EVERY}) — would desync server grounding")

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
    policy = PolicyClient(host=args.server_host, port=args.server_port,
                          recv_timeout=args.server_timeout)
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

    Sends the chunk executing right now (server's FDM "imagine the executing
    chunk" pass) and, from cycle 1 on, the PREVIOUS chunk's keyframes +
    just-executed action to ground. On cycle 0 there is no real feedback yet
    (prev_kf is None) — the server skips grounding for the cold-start chunk
    (author-confirmed); obs/state are omitted entirely. state/executing_action
    are passed through verbatim — the server runs its own preprocess_action.
    """
    try:
        payload = {
            "async_step": True,
            "executing_action": cur,  # C_n, verbatim
        }
        if prev_kf is not None:       # cycle >=1: ground C_{n-1}
            payload["obs"] = prev_kf      # K_{n-1}
            payload["state"] = prev_act   # a_{n-1}, verbatim
        resp = policy.infer(payload)
        out_q.put(("ok", resp))
    except BaseException as e:  # surface daemon failure to the main thread
        out_q.put(("err", e))


class ResetRequested(Exception):
    """Operator pressed 'r' during a running task (single keystroke, cbreak
    mode — no Enter). The current task aborts at the next chunk boundary;
    _run_inference_loop catches this, runs _initialize_pose, waits for the
    standby Enter, and starts a fresh task — which re-issues policy.reset
    (server clears its KV cache via _reset → create_empty_cache). Same
    effect as Ctrl+C → restart, but one key and the loop stays alive."""


def _reset_stdin_watcher(reset_event, stop_event, fd):
    """Daemon thread body: single-key reset. Terminal is in cbreak mode
    (set by _make_reset_watcher), so 'r'/'R' is delivered byte-by-byte
    without waiting for Enter. select() with a short timeout lets
    stop_event stop us promptly before any input() call elsewhere."""
    try:
        while not stop_event.is_set():
            rlist, _, _ = select.select([fd], [], [], 0.1)
            if not rlist:
                continue
            try:
                ch = os.read(fd, 1)
            except (BlockingIOError, OSError):
                continue
            if not ch:                # EOF — disable reset for this task
                return
            if ch in (b"r", b"R"):
                log.info("Reset key pressed — task will abort at next chunk "
                         "boundary (server KV cache will be cleared on re-run)")
                reset_event.set()
                return
            # any other keystroke is ignored
    except Exception:
        log.exception("Reset watcher crashed; reset key disabled this task")


def _make_reset_watcher(args):
    """Factory: returns (reset_event, stop_event, watcher_thread, restore_fn).

    Puts stdin in cbreak (single-keystroke) mode so pressing 'r' alone is
    detected immediately — no Enter required. cbreak keeps the ISIG flag
    so Ctrl+C still raises KeyboardInterrupt. restore_fn returns stdin to
    canonical mode and MUST be called before any input() in the loop
    (otherwise input() echoes nothing and line-editing is broken).

    Falls back to disabled (no terminal change, no watcher) if --repeat is
    off, stdin is not a TTY, or termios fails — never raises.
    """
    reset_event = threading.Event()
    stop_event = threading.Event()
    watcher = None
    restore = None
    # Reset key is always meaningful — pressing 'r' is itself an explicit
    # re-run request, so we don't require --repeat. (Without --repeat, the
    # reset triggers exactly one re-run; with --repeat, the loop continues
    # naturally after the re-run as it would post any task.)
    try:
        fd = sys.stdin.fileno()
    except (ValueError, OSError):
        log.info("stdin has no fileno — reset key disabled this task")
        return reset_event, stop_event, watcher, restore
    if not os.isatty(fd):
        log.info("stdin is not a TTY — reset key disabled this task")
        return reset_event, stop_event, watcher, restore
    try:
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except termios.error as e:
        log.info(f"Cannot set cbreak mode ({e}) — reset key disabled this task")
        return reset_event, stop_event, watcher, restore

    def restore_fn():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            # Drain anything the user typed in cbreak that the watcher
            # didn't consume (e.g. a habitual '\n' typed after 'r'). Without
            # this the next input() sees a pending '\n' and returns
            # immediately, auto-pressing the standby Enter.
            termios.tcflush(fd, termios.TCIFLUSH)
        except Exception:
            log.exception("Failed to restore terminal mode")
    restore = restore_fn

    watcher = threading.Thread(
        target=_reset_stdin_watcher,
        args=(reset_event, stop_event, fd),
        daemon=True, name="reset-watcher")
    watcher.start()
    return reset_event, stop_event, watcher, restore


def _run_task_once(arm, grip, cam, policy, args, reset_event=None) -> None:
    """One full task: reset → cold_start (ONE chunk C0) → steady async loop.

    Every cycle (incl. cycle 0) overlaps the async_step request (daemon
    thread) with execute_chunk (main thread). Cycle 0 sends no prev feedback
    — the server skips grounding for the cold-start chunk and
    FDM-imagines+PREDICTs C1 while the robot executes C0. compute_kv_cache is
    NOT used — grounding folds into async_step (cycle n≥1). Returns when
    max_chunks cycles have completed.
    """
    log.info(f"Reset with prompt: {args.prompt!r}")
    policy.reset(args.prompt)

    # Cold start: ONE init frame in, ONE chunk back. The server grounds
    # nothing for C0; it FDM-imagines C0 (z_0 pinned) and PREDICTs C1 on the
    # FIRST async_step, overlapped with executing C0 below — so no chunk is
    # predicted on an ungrounded hallucination and none is executed serially.
    log.info("Cold start: sending one init frame, expecting one chunk")
    init = cam.get_obs(args.prompt)
    resp = policy.infer({"cold_start": True, "obs": init})
    C0 = resp["action"]
    log.info(f"Cold start returned C0 {C0.shape}")
    log_chunk_summary(0, C0)

    # Uniform overlapped loop from cycle 0. Cycle 0 sends no prev feedback
    # (server skips grounding for C0) and runs C0 is_first_chunk=True (frame
    # 0 skipped → 4 keyframes); cycle n≥1 grounds C_{n-1} → 8 keyframes.
    prev_kf, prev_act = None, None  # no real feedback before cycle 0
    cur = C0
    sync_mode = getattr(args, "sync", False)
    for n in range(args.max_chunks):
        if not sync_mode:
            # Async: spawn worker BEFORE execute so the request overlaps it
            out_q = queue.Queue()
            th = threading.Thread(
                target=_async_step_worker,
                args=(policy, prev_kf, prev_act, cur, out_q),
                daemon=True, name=f"async_step-{n}")
            th.start()  # Branch B fires; do NOT block main on it

        # Branch A: execute the current chunk at 30 Hz on the main thread.
        # Cycle 0 is the first chunk → skip frame-0 ('observe start').
        exec_tic = time.time()
        Kn, st = execute_chunk(cur, arm, grip, cam, is_first_chunk=(n == 0),
                               substep_dt=1.0 / args.substep_hz,
                               verbose_substeps=not args.quiet_substeps)
        exec_s = time.time() - exec_tic

        if sync_mode:
            # Sync: do the request inline on the main thread, AFTER execute.
            # Wire payload is identical to async — only the threading and
            # the wall-clock relationship to execute_chunk differ. The
            # PolicyClient recv_timeout (built from args.server_timeout)
            # still bounds the wait, so a stalled server raises TimeoutError
            # the same way and routes to run()'s finally.
            payload = {"async_step": True, "executing_action": cur}
            if prev_kf is not None:
                payload["obs"] = prev_kf
                payload["state"] = prev_act
            resp = policy.infer(payload)
            next_chunk = resp["action"]
        else:
            # Async: collect from worker, bounded so we never freeze. The
            # PolicyClient recv timeout fires inside the worker; this
            # out_q.get timeout is the defense-in-depth backstop in case
            # the worker hangs somewhere other than recv.
            to = getattr(args, "server_timeout", 60.0)
            try:
                status, payload = out_q.get(timeout=to)
            except queue.Empty:
                raise TimeoutError(
                    f"async_step did not return within {to:.0f}s — "
                    f"server/tunnel stalled; aborting so arm_sdk is "
                    f"released (robot not latched)")
            th.join()
            if status == "err":
                raise payload
            next_chunk = payload["action"]

        log.info(
            f"[cycle {n}] {'SYNC ' if sync_mode else ''}execute={exec_s:6.3f}s "
            f"(motion ~{st['n_substeps']/args.substep_hz:.2f}s, "
            f"overruns={st['n_overrun']}/{st['n_substeps']}, "
            f"keyframes={len(Kn)})  "
            + _fmt_infer_timing("async_step", policy.last_timing))
        log_chunk_summary(n + 1, next_chunk)

        # Rotate: this cycle's keyframes/action become next cycle's "previous"
        # to ground; next_chunk becomes what we execute next.
        prev_kf, prev_act, cur = Kn, cur, next_chunk

        # Reset key polled at chunk boundary (~1-2 s worst-case latency,
        # well under one chunk). Don't break mid-chunk — the 30 Hz dispatch
        # has to finish cleanly before we hand control back to init.
        if reset_event is not None and reset_event.is_set():
            log.info(f"[cycle {n}] reset acknowledged — aborting task; "
                     f"will run INIT then wait for Enter to start fresh task")
            raise ResetRequested()


def _wait_for_rerun(args) -> bool:
    """Between-tasks gate. Returns True to proceed to INIT mode (arm to
    ready + gripper cycle), False to exit. After init, _wait_for_operator
    is the second Enter that starts MAIN mode — same gate as first-run
    startup. Auto-pressed by --auto-start (= unattended loop forever).
    """
    if getattr(args, "auto_start", False):
        log.info("--auto-start: skipping re-run prompt; proceeding to init")
        return True
    log.info("===============================================================")
    log.info("TASK COMPLETE.")
    log.info("Press [Enter] to enter INITIALIZATION mode (arm/gripper reset).")
    log.info("After init you'll be prompted again to start the next task.")
    log.info("Press [Ctrl+C] to quit safely.")
    log.info("===============================================================")
    try:
        input("")
    except EOFError:
        log.info("EOF on stdin — exiting repeat mode")
        return False
    return True


def _run_inference_loop(arm, grip, cam, policy, args) -> None:
    """Run the task once; with --repeat, prompt Enter between tasks to clear
    the server's KV cache (via reset → cold_start) and run it again.

    Between tasks: switch arm kp back to the stiff hold value captured at
    startup (less gravity sag while the operator stages), move the arm and
    grippers back to ready, then block on Enter (same gate as the initial
    standby). On Enter: switch kp back to the inference value and run again.
    """
    while True:
        # Start the single-key 'r' watcher only for this task's duration.
        # Terminal is put in cbreak mode while the task runs and restored
        # before any Enter prompt below — no stdin race, line-editing
        # works again for input().
        reset_event, stop_event, watcher, restore = _make_reset_watcher(args)
        if watcher is not None:
            log.info("Press 'r' at any time to reset to INIT mode "
                     "(no Enter needed; server KV cache cleared on next task).")
        task_was_reset = False
        try:
            try:
                _run_task_once(arm, grip, cam, policy, args,
                               reset_event=reset_event)
            except ResetRequested:
                task_was_reset = True
        finally:
            stop_event.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            # MUST restore canonical mode before any input() — otherwise
            # the Enter prompts below won't echo / line-edit properly.
            if restore is not None:
                restore()

        # Exit only when the task ended naturally AND --repeat is off.
        # A reset ALWAYS triggers one full INIT → standby → re-run cycle,
        # even without --repeat (pressing 'r' is itself the re-run signal).
        if not task_was_reset and not getattr(args, "repeat", False):
            return

        # ----- Enter #1: operator confirms ready for INIT -----
        # First-run startup runs init automatically. Between tasks the arm
        # is in an arbitrary post-task pose, so we ask before moving it.
        # The reset path SKIPS this prompt — pressing 'r' already implied
        # "I want to reset", no need to ask again.
        if not task_was_reset:
            if not _wait_for_rerun(args):
                return
        else:
            log.info("Reset acknowledged — proceeding to INIT (skipping "
                     "the 'Press Enter to init' gate)")

        # ----- INIT mode (mirrors _initialize_pose at startup) -----
        # Stiffer hold for the standby wait (less gravity sag). _init_kp_arm
        # is captured in run() before the first set_arm_kp(inference_kp_arm)
        # so we restore exactly what the arm was using at startup — no
        # hardcoded duplicate.
        init_kp = getattr(args, "_init_kp_arm", None)
        if init_kp is not None:
            log.info(f"Switching arm kp back to standby value: {init_kp}")
            arm.set_arm_kp(init_kp)
        _initialize_pose(arm, grip, args)

        # ----- Enter #2: standby (same call as first-run startup) -----
        _wait_for_operator(args)

        # ----- MAIN mode -----
        log.info(f"Switching arm kp back to inference value: {args.inference_kp_arm}")
        arm.set_arm_kp(args.inference_kp_arm)
        # loop: _run_task_once again — sends a fresh reset (clears server
        # KV cache) and cold_start, exactly like a clean start.


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
        # Capture the stiff-hold kp before we drop to inference, so --repeat
        # can restore it between tasks without duplicating the constant.
        args._init_kp_arm = arm.kp_arm
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
    p.add_argument("--server-timeout", type=float, default=60.0,
                   help="Max seconds to wait for one server reply (cold_start/"
                        "async_step). On a stall the client aborts and releases "
                        "arm_sdk instead of hanging forever with the robot "
                        "latched. Must exceed worst-case server compute "
                        "(~2-3s); default 60.")
    p.add_argument("--sync", action="store_true",
                   help="Run async_step requests synchronously on the main "
                        "thread, AFTER execute_chunk completes — no daemon "
                        "thread, no overlap. Wire contract unchanged. "
                        "Per-chunk wall becomes (execute + request) instead "
                        "of max(execute, request). Useful for debugging / "
                        "latency attribution; significantly slower steady-"
                        "state. Default is async (overlapped).")

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
    p.add_argument("--repeat", action="store_true",
                   help="After each task completes, return arm to ready pose "
                        "and wait for [Enter] to clear the server KV cache "
                        "and run the task again. Without this flag, exit "
                        "after one task. --auto-start with --repeat = loop "
                        "forever unattended.")
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start inference immediately.")
    args = p.parse_args()

    run(args)


if __name__ == "__main__":
    main()
