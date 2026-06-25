"""Main inference loop for the G1, talking to a DiT4DiT
deployment/model_server/server_policy.py (WebsocketPolicyServer) instead of an
openpi serve_policy.py or the LingBot-VA cloud server.

WHAT CHANGED vs main_openpi.py
------------------------------
Nothing in the robot control path. arm_controller.py, gripper_controller.py and
camera_client.py are used verbatim, and the receding-horizon + one-chunk-prefetch
loop is identical. Only the data send/receive changes:

  * OpenPIPolicy / raw PolicyClient (server un-normalizes, returns raw actions)
        --> DitPolicy (server returns NORMALIZED actions and expects NORMALIZED
            state, so this client carries dataset_statistics.json and does the
            normalization on both ends, mirroring eval_policy.py).
  * obs is three cameras (ego + 2 wrists), sent in OBS_CAM_KEYS order; the
        server resizes each and concatenates them side-by-side into one
        conditioning frame, matching UnitreeG1Dex1ThreeCamConfig training.

The robot init / standby / kp-switch / cleanup sequence is preserved exactly,
because those are safety-critical and have nothing to do with the server.

OBSERVATION / ACTION CONTRACT (must match the served checkpoint)
---------------------------------------------------------------
  obs (sent every tick), assembled by build_obs and finished inside DitPolicy:
    image    list of 3 RGB uint8 cameras [ego, left_wrist, right_wrist];
             resized + side-by-side concatenated to (224, 672) server-side
    state    float32 (16,) = [14 arm q | L grip | R grip], min-max normalized
             to [-1,1] and padded to 64 server-side-expected dim
    prompt   str

  action returned by the server: normalized [H, 32]; DitPolicy un-normalizes to
  raw [H, 16]:
    [:, 0:14] absolute arm joint targets (rad), order == ARM_JOINTS
    [:, 14]   left gripper  (rad, [GRIPPER_MIN, GRIPPER_MAX])
    [:, 15]   right gripper (rad, [GRIPPER_MIN, GRIPPER_MAX])

Precondition: robot already in 'ai' motion mode (set via the Unitree app).

Usage (run from the g1-client/ directory):
  python main-dit.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 10093 \\
      --norm-stats /path/to/<run_dir>/dataset_statistics.json \\
      --prompt "pick the red bottle"
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
from g1_client.dit_policy import DitPolicy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_dit.main")

# Action tensor layout for the G1 DiT checkpoint after un-normalization: [H, 16].
ARM_CHANNELS = slice(0, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15

ARM_JOINT_NAMES = [
    "L_pitch", "L_roll ", "L_yaw ", "L_elbow", "L_wrR ", "L_wrP ", "L_wrY ",
    "R_pitch", "R_roll ", "R_yaw ", "R_elbow", "R_wrR ", "R_wrP ", "R_wrY ",
]

# The 3-camera G1 DiT checkpoint conditions on [ego, left_wrist, right_wrist],
# concatenated side-by-side server-side. ORDER MUST MATCH the training video_keys
# (UnitreeG1Dex1ThreeCamConfig: ego_view, left_wrist, right_wrist).
OBS_CAM_KEYS = [
    "observation.images.cam_left_high",    # ego / head (primary)
    "observation.images.cam_left_wrist",   # left wrist
    "observation.images.cam_right_wrist",  # right wrist
]


# ---------- observation assembly (the "receive" side) ----------

def _jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode the camera client's JPEG bytes back to an RGB uint8 array.

    camera_client.get_obs_images() returns BGR-encoded JPEG. eval_policy.py feeds
    the model LeRobot RGB frames, so we decode + BGR->RGB here to match, rather
    than editing camera_client.py.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on a camera frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_obs(cam: CameraClient, arm: ArmController, grip: GripperController,
              prompt: str) -> dict:
    """Assemble one DiT observation from the (unchanged) controllers.

    All three cameras are sent in OBS_CAM_KEYS order; the server resizes each to
    the training resolution and concatenates them side-by-side into one
    conditioning frame (matching the dataloader). The raw 16-dim state is
    normalized inside DitPolicy (server expects normalized state).
    """
    imgs = cam.get_obs_images()  # dict of JPEG bytes (BGR, q90), LeRobot keys
    views = [_jpeg_to_rgb(imgs[k]) for k in OBS_CAM_KEYS]
    left_q, right_q = grip.get_state()
    arm_q = arm.get_arm_q()  # (14,)
    state = np.concatenate([arm_q, [left_q, right_q]]).astype(np.float32)  # (16,)
    return {"image": views, "state": state, "prompt": prompt}


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

def _pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def _stat(xs):
    """(min, p50, p95, max, mean) of a list of numbers."""
    return (min(xs), _pct(xs, 50), _pct(xs, 95), max(xs), float(np.mean(xs)))


def _timing_rec(tm):
    """Turn PolicyClient.last_timing into a uniform per-infer record (ms)."""
    return {
        "wall_ms": tm.get("total_s", 0.0) * 1e3,
        "pack_ms": tm.get("pack_s", 0.0) * 1e3,
        "send_ms": tm.get("send_s", 0.0) * 1e3,
        "wait_recv_ms": tm.get("wait_recv_s", 0.0) * 1e3,
        "unpack_ms": tm.get("unpack_s", 0.0) * 1e3,
        "bytes_sent": tm.get("bytes_sent", -1),
        "bytes_recv": tm.get("bytes_recv", -1),
    }


def _summarize_timing(infer_recs, chunk_recs, args):
    """Compact end-of-run latency summary (per-infer breakdown + execute/stall)."""
    if not infer_recs:
        return
    budget_ms = (args.prefetch_lead / args.control_hz) * 1e3
    log.info("=" * 60)
    log.info(f"per-infer latency over {len(infer_recs)} calls (ms):")
    comps = [("wall(total)", "wall_ms"), ("pack", "pack_ms"), ("send", "send_ms"),
             ("wait_recv", "wait_recv_ms"), ("unpack", "unpack_ms")]
    means = {}
    for label, key in comps:
        mn, p50, p95, mx, me = _stat([r[key] for r in infer_recs])
        means[label.split("(")[0]] = me
        log.info(f"  {label:<14} {mn:7.1f} {p50:7.1f} {p95:7.1f} {mx:7.1f} {me:7.1f}")
    sub = {k: means[k] for k in ("pack", "send", "wait_recv", "unpack")}
    bn = max(sub, key=sub.get)
    log.info(f"  bottleneck: {bn} ~= {sub[bn]:.0f}ms "
             f"({sub[bn]/means['wall']*100:.0f}% of infer total)")
    up = [r["bytes_sent"] for r in infer_recs if r["bytes_sent"] > 0]
    if up:
        log.info(f"  upload payload ~= {np.mean(up)/1024:.0f} KiB/infer")
    if chunk_recs:
        ex = [r["exec_s"] for r in chunk_recs]
        jw = [r["join_wait_s"] * 1e3 for r in chunk_recs]
        stalled = sum(1 for x in jw if x > 1.0)
        log.info(f"execute/chunk: mean={np.mean(ex):.2f}s | join_wait: mean={np.mean(jw):.0f}ms "
                 f"p95={_pct(jw,95):.0f}ms | stalled {stalled}/{len(jw)} chunks "
                 f"(infer didn't finish within the {budget_ms:.0f}ms prefetch budget)")
    log.info("=" * 60)


def _infer_worker(policy: DitPolicy, obs: dict, box: dict) -> None:
    """Run one blocking infer on a daemon thread; stash result/exception/timing."""
    try:
        result = policy.infer(obs)
        box["timing"] = dict(policy.last_timing or {})
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
        box["actions"] = actions
    except BaseException as e:  # surface to the main thread
        box["err"] = e


def _run_inference_loop(arm, grip, cam, policy, args) -> None:
    """Receding-horizon loop with one-chunk prefetch + boundary smoothing.

    Identical in structure to main_openpi.py: execute the current chunk at
    args.control_hz on the main thread; when prefetch_lead steps remain, snapshot
    a fresh obs and fire the next infer on a daemon thread so it overlaps the tail
    of this chunk. Join, swap, repeat. The two anti-jitter measures (time
    alignment + cross-fade) are preserved unchanged.
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt

    # First chunk is a blocking infer (nothing to overlap it against yet).
    log.info(f"First inference (prompt={prompt!r})")
    result = policy.infer(build_obs(cam, arm, grip, prompt))
    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 16:
        raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
    log_chunk_ranges(0, actions)

    infer_recs = [_timing_rec(dict(policy.last_timing or {}))]
    chunk_recs = []

    # Boundary-smoothing state, persisted across chunks.
    start_idx = 0
    last_cmd = None
    for c in range(1, args.max_chunks + 1):
        H = actions.shape[0]
        end_idx = H if args.exec_steps <= 0 else min(start_idx + args.exec_steps, H)
        n = end_idx - start_idx
        if n <= 0:
            raise RuntimeError(
                f"chunk {c}: nothing left to execute (start_idx={start_idx}, "
                f"horizon={H}) — --prefetch-lead too large for this horizon")
        lead = min(args.prefetch_lead, n)
        box: dict = {}
        th = None
        pending_skip = 0

        exec_t0 = time.time()
        for i in range(n):
            if arm.faulted():
                raise RuntimeError("ArmController control thread faulted — aborting")
            tic = time.time()

            a = actions[start_idx + i].astype(np.float64)
            # Cross-fade the first --blend-steps from the last commanded pose.
            if last_cmd is not None and i < args.blend_steps:
                alpha = (i + 1) / (args.blend_steps + 1)
                a = (1.0 - alpha) * last_cmd + alpha * a
            arm.set_arm_target(a[ARM_CHANNELS])
            grip.set_targets(
                float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
            )
            last_cmd = a

            # Fire the next-chunk request once `lead` steps remain so it overlaps.
            if th is None and (n - i) <= lead:
                obs_next = build_obs(cam, arm, grip, prompt)
                pending_skip = (n - 1 - i) if args.chunk_align else 0
                th = threading.Thread(target=_infer_worker,
                                      args=(policy, obs_next, box),
                                      daemon=True, name=f"prefetch-{c}")
                th.start()

            sleep = dt - (time.time() - tic)
            if sleep > 0:
                time.sleep(sleep)

        # Collect the prefetched next chunk (it should be done or nearly).
        exec_s = time.time() - exec_t0
        join_t0 = time.time()
        if th is not None:
            th.join()
        join_wait_s = time.time() - join_t0
        if "err" in box:
            raise box["err"]
        next_actions = box["actions"]

        rec = _timing_rec(box.get("timing", {}))
        infer_recs.append(rec)
        chunk_recs.append({"exec_s": exec_s, "join_wait_s": join_wait_s})
        log.info(f"[chunk {c}] execute={exec_s:.2f}s join_wait={join_wait_s*1e3:.0f}ms | "
                 f"infer wall={rec['wall_ms']:.0f}ms pack={rec['pack_ms']:.0f} "
                 f"send={rec['send_ms']:.0f} wait_recv={rec['wait_recv_ms']:.0f} "
                 f"unpack={rec['unpack_ms']:.0f}")
        log_chunk_ranges(c, next_actions)
        actions = next_actions
        start_idx = min(pending_skip, next_actions.shape[0] - 1)

    _summarize_timing(infer_recs, chunk_recs, args)


# ---------- pipeline stages (kept identical to main_openpi.py) ----------

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
        policy = DitPolicy(host=args.server_host, port=args.server_port,
                           norm_stats_path=args.norm_stats, unnorm_key=args.unnorm_key,
                           zero_gripper_state=args.zero_gripper_state)
        _run_inference_loop(arm, grip, cam, policy, args)
        _initialize_pose(arm, grip, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--server-host", required=True, help="DiT4DiT server_policy.py host or IP")
    p.add_argument("--server-port", type=int, default=10093, help="DiT4DiT server port (default 10093)")
    p.add_argument("--norm-stats", required=True,
                   help="Path to the served checkpoint's dataset_statistics.json "
                        "(run_dir/dataset_statistics.json). Used to normalize state "
                        "and un-normalize actions client-side.")
    p.add_argument("--unnorm-key", default=None,
                   help="Dataset key inside dataset_statistics.json to use; omit if "
                        "the checkpoint was trained on a single dataset.")
    p.add_argument("--zero-gripper-state", action="store_true", dest="zero_gripper_state",
                   help="Zero the two gripper state dims instead of sending the live "
                        "gripper q. Only for checkpoints trained with constant-0 gripper "
                        "state; this dataset's grippers vary, so the default sends live q.")
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--prompt", default="pick the red bottle")
    p.add_argument("--max-chunks", type=int, default=30,
                   help="How many action chunks to run before stopping")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Per-step dispatch rate; match your LeRobot recording fps (30)")
    p.add_argument("--exec-steps", type=int, default=0,
                   help="Steps to execute per chunk before re-querying; 0 = full horizon.")
    p.add_argument("--prefetch-lead", type=int, default=5,
                   help="Start the next inference when this many steps remain in the "
                        "current chunk, so it overlaps and the boundary doesn't stall")
    p.add_argument("--blend-steps", type=int, default=5,
                   help="Cross-fade the first N steps of each new chunk from the last "
                        "commanded pose so a chunk swap ramps in smoothly (0 disables)")
    p.add_argument("--no-chunk-align", action="store_false", dest="chunk_align",
                   help="Disable chunk time-alignment (execute every chunk from index 0).")
    # ---- Safety / motion limits (same as main_openpi.py) ----
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
