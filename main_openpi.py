"""Main inference loop for the G1, talking to a standard openpi serve_policy.py
server instead of the LingBot-VA cloud server.

WHAT CHANGED vs main.py
-----------------------
Nothing in the robot control path. arm_controller.py, gripper_controller.py and
camera_client.py are used verbatim. Only the data send/receive changes:

  * PolicyClient (LingBot, stateful reset/cold_start/async_step, [16,F,S])
        --> OpenPIPolicy (stateless infer(obs) -> {"actions": [H,16]})
  * the two-chunk FDM-grounded overlap loop (Algorithm 2)
        --> a plain receding-horizon loop: infer a chunk, dispatch it at a
            fixed rate, prefetch the next chunk on a daemon thread so the
            chunk boundary doesn't stall (same idea as the UR3 PrefetchBroker).

The robot init / standby / kp-switch / cleanup sequence is preserved exactly,
because those are safety-critical and have nothing to do with the server.

OBSERVATION / ACTION CONTRACT (must match your G1 openpi DataConfig)
--------------------------------------------------------------------
This is the one thing you must keep in lockstep with the server checkpoint —
the obs keys here must match the keys your RepackTransform consumes, and the
state/action dimension order must match how the LeRobot dataset was recorded.

  obs (sent every tick):
    observation.images.cam_left_high   uint8 HxWx3 RGB
    observation.images.cam_left_wrist  uint8 HxWx3 RGB
    observation.images.cam_right_wrist uint8 HxWx3 RGB
    observation.state                  float32 (16,) = [14 arm q | L grip | R grip]
    prompt                             str

  action returned: ndarray [H, 16]
    [:, 0:14] absolute arm joint targets (rad), order == ARM_JOINTS
    [:, 14]   left gripper  (rad, [GRIPPER_MIN, GRIPPER_MAX])
    [:, 15]   right gripper (rad, [GRIPPER_MIN, GRIPPER_MAX])

If your dataset stored the gripper normalized (e.g. 0..1) instead of rad, scale
[:, 14:16] here before set_targets — that is the only spot likely to need a
tweak, and it is data handling, not control code.

Precondition: robot already in 'ai' motion mode (set via the Unitree app).

Usage (run from the lingbot_g1_client/ directory):
  python main_openpi.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 8000 \\
      --prompt "pick up the pink object and place it on the blue cross mark"
"""

import argparse
import logging
import threading
import time

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_client.arm_controller import ArmController, INIT_POSE_READY
from g1_client.gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX
from g1_client.camera_client import CameraClient
from g1_client.policy_client import PolicyClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_openpi.main")

# Action tensor layout for the G1 openpi checkpoint: [H, 16].
ARM_CHANNELS = slice(0, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15

ARM_JOINT_NAMES = [
    "L_pitch", "L_roll ", "L_yaw ", "L_elbow", "L_wrR ", "L_wrP ", "L_wrY ",
    "R_pitch", "R_roll ", "R_yaw ", "R_elbow", "R_wrR ", "R_wrP ", "R_wrY ",
]


# ---------- observation assembly (the "receive" side) ----------

def _jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode the camera client's JPEG bytes back to an RGB uint8 array.

    camera_client.get_obs_images() returns BGR-encoded JPEG (it was built for a
    server that does its own decode). openpi expects decoded RGB arrays, so we
    decode here in the glue rather than editing camera_client.py. The q90 round
    trip is well within the model's training distribution (LeRobot mp4/H.264).
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on a camera frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_obs(cam: CameraClient, arm: ArmController, grip: GripperController,
              prompt: str) -> dict:
    """Assemble one openpi observation from the (unchanged) controllers."""
    imgs = cam.get_obs_images()  # dict of JPEG bytes, keys already LeRobot-style
    left_q, right_q = grip.get_state()
    arm_q = arm.get_arm_q()  # (14,)
    state = np.concatenate([arm_q, [left_q, right_q]]).astype(np.float32)  # (16,)
    return {
        "observation.images.cam_left_high": _jpeg_to_rgb(imgs["observation.images.cam_left_high"]),
        "observation.images.cam_left_wrist": _jpeg_to_rgb(imgs["observation.images.cam_left_wrist"]),
        "observation.images.cam_right_wrist": _jpeg_to_rgb(imgs["observation.images.cam_right_wrist"]),
        "observation.state": state,
        "prompt": prompt,
    }


def log_chunk_ranges(chunk_id: int, actions: np.ndarray) -> None:
    """One-line per-joint range sanity print before a chunk streams to the arm."""
    arm = actions[:, ARM_CHANNELS]
    gl = actions[:, LEFT_GRIPPER_CHANNEL]
    gr = actions[:, RIGHT_GRIPPER_CHANNEL]
    log.info(f"[chunk {chunk_id}] H={actions.shape[0]} arm joint ranges (rad):")
    for i, name in enumerate(ARM_JOINT_NAMES):
        log.info(f"    {name} min={arm[:, i].min():+.3f} max={arm[:, i].max():+.3f}")
    log.info(f"[chunk {chunk_id}] gripper L:[{gl.min():.2f},{gl.max():.2f}] "
             f"R:[{gr.min():.2f},{gr.max():.2f}]")


# ---------- inference loop (the "send" side) ----------

def _infer_worker(policy: PolicyClient, obs: dict, box: dict) -> None:
    """Run one blocking infer on a daemon thread; stash result/exception.

    Extracts and validates the action array (same as the first-chunk path) so
    the loop receives a [H, 16] ndarray, not the raw {"actions": ...} dict.
    """
    try:
        actions = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
        box["actions"] = actions
    except BaseException as e:  # surface to the main thread
        box["err"] = e


def _run_inference_loop(arm, grip, cam, policy, args) -> None:
    """Receding-horizon loop with one-chunk prefetch.

    Execute the current chunk at args.control_hz on the main thread; when
    prefetch_lead steps remain, snapshot a fresh obs and fire the next infer on
    a daemon thread so it overlaps the tail of this chunk. Join, swap, repeat.
    This mirrors the UR3 PrefetchBroker's "fetch while N steps remain" idea so
    chunk boundaries don't stall the dispatch (the arm/gripper publish threads
    keep holding the last target meanwhile, so even a stall is safe, just jerky).
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt

    # First chunk is a blocking infer (nothing to overlap it against yet).
    log.info(f"First inference (prompt={prompt!r})")
    result = policy.infer(build_obs(cam, arm, grip, prompt))
    actions = np.asarray(result["actions"],dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 16:
        raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
    log_chunk_ranges(0, actions)

    for c in range(1, args.max_chunks + 1):
        n = actions.shape[0] if args.exec_steps <= 0 else min(args.exec_steps, actions.shape[0])
        prefetch_lead = min(args.prefetch_lead, n)
        box: dict = {}
        th = None

        exec_tic = time.time()
        for t in range(n):
            if arm.faulted():
                raise RuntimeError("ArmController control thread faulted — aborting")
            tic = time.time()

            a = actions[t]
            arm.set_arm_target(a[ARM_CHANNELS].astype(np.float64))
            grip.set_targets(
                float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
            )

            # Fire the next-chunk request once prefetch_lead steps remain.
            if th is None and (n - t) <= prefetch_lead:
                obs_next = build_obs(cam, arm, grip, prompt)
                th = threading.Thread(target=_infer_worker,
                                      args=(policy, obs_next, box),
                                      daemon=True, name=f"prefetch-{c}")
                th.start()

            sleep = dt - (time.time() - tic)
            if sleep > 0:
                time.sleep(sleep)

        # Collect the prefetched next chunk (it should be done or nearly).
        if th is not None:
            th.join()
        if "err" in box:
            raise box["err"]
        next_actions = box["actions"]
        log_chunk_ranges(c, next_actions)
        actions = next_actions


# ---------- pipeline stages (kept identical in spirit to main.py) ----------

def _initialize_pose(arm, grip, args) -> None:
    log.info(f"Moving arms to ready pose over {args.init_duration:.1f}s "
             f"(velocity_limit={args.velocity_limit} rad/s)")
    arm.move_to_pose(INIT_POSE_READY, duration=args.init_duration,
                     velocity_limit=args.velocity_limit)
    half = args.gripper_init_duration / 2
    log.info(f"Closing grippers to {GRIPPER_MIN} over {half:.1f}s")
    grip.move_to_targets(GRIPPER_MIN, GRIPPER_MIN, duration=half)
    log.info(f"Opening grippers to ({args.init_gripper_left}, {args.init_gripper_right}) "
             f"over {half:.1f}s")
    grip.move_to_targets(args.init_gripper_left, args.init_gripper_right, duration=half)
    log.info("Init complete.")
    time.sleep(args.settle_duration)
    log.info("Arms settled at ready pose.")


def _wait_for_operator(args) -> None:
    if args.auto_start:
        return
    log.info("===============================================================")
    log.info("STANDBY: arms locked at ready pose.")
    log.info("Set up the scene, then press [Enter] to connect and start.")
    log.info("Press [Ctrl+C] at any time to abort safely.")
    log.info("===============================================================")
    try:
        input("")
    except EOFError:
        log.info("EOF on stdin — proceeding without prompt")


def _cleanup(arm, grip, cam, policy) -> None:
    """Release every resource. disable_arm_sdk MUST run — it returns arm
    authority to the locomotion service. Each step isolated so a second Ctrl+C
    cannot skip later steps."""
    log.info("Shutting down — releasing arm_sdk")
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


def run(args) -> None:
    log.info(f"Initializing DDS on {args.iface}")
    ChannelFactoryInitialize(0, args.iface)

    arm = ArmController(publish_hz=50.0, velocity_limit=args.velocity_limit)
    arm.start()
    grip = None
    cam = None
    policy = None
    try:
        grip = GripperController(publish_hz=200.0)
        grip.start()
        cam = CameraClient(host=args.image_server)
        _initialize_pose(arm, grip, args)
        _wait_for_operator(args)
        log.info(f"Switching arm kp to inference value: {args.inference_kp_arm}")
        arm.set_arm_kp(args.inference_kp_arm)
        policy = PolicyClient(host=args.server_host, port=args.server_port)
        _run_inference_loop(arm, grip, cam, policy, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--server-host", required=True, help="openpi serve_policy.py host or IP")
    p.add_argument("--server-port", type=int, default=8000, help="openpi server port (default 8000)")
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--prompt", default="pick the red bottle")
    p.add_argument("--max-chunks", type=int, default=20,
                   help="How many action chunks to run before stopping")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Per-step dispatch rate; match your LeRobot recording fps (default 30)")
    p.add_argument("--exec-steps", type=int, default=0,
                   help="Steps to execute per chunk before re-querying; 0 = full horizon. "
                        "Set smaller (e.g. 25) for tighter closed-loop reactivity.")
    p.add_argument("--prefetch-lead", type=int, default=8,
                   help="Start the next inference when this many steps remain in the "
                        "current chunk, so it overlaps and the boundary doesn't stall (default 8)")
    # ---- Safety / motion limits (same as main.py) ----
    p.add_argument("--velocity-limit", type=float, default=8.0,
                   help="rad/s velocity cap on the per-tick motion clamp (default 8.0)")
    p.add_argument("--inference-kp-arm", type=float, default=80.0,
                   help="kp for shoulder/elbow once inference starts (default 80)")
    p.add_argument("--init-duration", type=float, default=2.0)
    p.add_argument("--gripper-init-duration", type=float, default=1.0)
    p.add_argument("--settle-duration", type=float, default=1.0)
    p.add_argument("--init-gripper-left", type=float, default=5.0)
    p.add_argument("--init-gripper-right", type=float, default=5.0)
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start immediately.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
