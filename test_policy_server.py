"""Test harness for the G1 openpi policy server — NO ROBOT CONTROL.

Connects to a serve_policy.py websocket server, sends observations, and
inspects the returned action chunks. Never imports DDS / ArmController /
GripperController, never moves anything. Use it to validate the obs/action
contract and measure latency before you ever run main_openpi.py on hardware.

What it checks
--------------
  * connection + server metadata handshake
  * action chunk shape / dtype / finiteness
  * action dimension == 16, and per-channel ranges:
      - arm channels [0:14] vs the controller's joint limits
      - gripper channels [14:16] vs [GRIPPER_MIN, GRIPPER_MAX]
  * heuristics: warns if actions look like deltas/normalized rather than the
    absolute joint-radian / gripper-radian units the controllers expect
  * infer latency stats over N iterations

Two obs sources
---------------
  --synthetic (default): zero/random images + zero state. Runs ANYWHERE, no
      robot, no unitree_sdk2py, no teleimager. Pure server smoke test.
  --use-camera: pull real frames from the G1 image server (ZMQ only, still no
      DDS, still no robot motion). State stays synthetic (real joint state would
      need a DDS subscription; this harness deliberately avoids that).

Two client backends (test wire compatibility both ways)
-------------------------------------------------------
  --client openpi  (default): openpi_client.WebsocketClientPolicy
  --client lingbot          : your existing g1_client.policy_client.PolicyClient
      — use this to confirm the vendored msgpack_numpy is byte-compatible with
        the openpi server (the open question from the client comparison).

Usage:
  python test_policy_server.py --server-host 1.2.3.4 --server-port 8000 -n 5
  python test_policy_server.py --server-host 1.2.3.4 --use-camera --image-server 192.168.123.164
  python test_policy_server.py --server-host 1.2.3.4 --client lingbot
"""

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_openpi.test")

# ---- Image geometry: must match camera_client (CAM_HEIGHT/WIDTH) ----
CAM_HEIGHT = 256
CAM_WIDTH = 320

# ---- Action layout for the G1 checkpoint: [H, 16] ----
ARM_CHANNELS = slice(0, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15
EXPECTED_ACTION_DIM = 16

# ---- Limits inlined (NOT imported from arm_controller, so this file has zero
#      robot dependencies). Keep in sync with arm_controller.ARM_JOINT_MIN/MAX. ----
ARM_JOINT_MIN = np.array([
    -2.8, -0.4, -2.4, -0.5, -1.9, -1.5, -1.5,
    -2.8, -2.2, -2.4, -0.5, -1.9, -1.5, -1.5,
], dtype=np.float64)
ARM_JOINT_MAX = np.array([
    1.4, 2.2, 2.4, 2.9, 1.9, 1.5, 1.5,
    1.4, 0.4, 2.4, 2.9, 1.9, 1.5, 1.5,
], dtype=np.float64)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 5.4

ARM_JOINT_NAMES = [
    "L_pitch", "L_roll ", "L_yaw ", "L_elbow", "L_wrR ", "L_wrP ", "L_wrY ",
    "R_pitch", "R_roll ", "R_yaw ", "R_elbow", "R_wrR ", "R_wrP ", "R_wrY ",
]

# Obs keys — must match the server checkpoint's RepackTransform / DataConfig.
IMG_KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
STATE_KEY = "observation.state"
PROMPT_KEY = "prompt"


# ---------- client backends ----------

def make_client(args):
    """Return (close_fn, infer_fn, metadata) for the chosen backend."""
    if args.client == "openpi":
        from openpi_client import websocket_client_policy
        c = websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host, port=args.server_port)
        meta = c.get_server_metadata()

        def _close():
            ws = getattr(c, "_ws", None)
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
        return _close, c.infer, meta

    # lingbot backend: reuse the existing vendored-msgpack client unchanged
    from g1_client.policy_client import PolicyClient
    c = PolicyClient(host=args.server_host, port=args.server_port)
    return c.close, c.infer, c.get_server_metadata()


# ---------- obs sources ----------

def _synthetic_images(rng, mode):
    """Three uint8 HWC RGB frames (zeros or random) matching camera geometry."""
    shape = (CAM_HEIGHT, CAM_WIDTH, 3)
    if mode == "random":
        return [rng.integers(0, 256, shape, dtype=np.uint8) for _ in IMG_KEYS]
    return [np.zeros(shape, dtype=np.uint8) for _ in IMG_KEYS]


def _camera_images(cam):
    """Real frames from the image server, decoded JPEG -> RGB (no DDS, no motion)."""
    import cv2
    imgs = cam.get_obs_images()
    out = []
    for k in IMG_KEYS:
        arr = np.frombuffer(imgs[k], dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"imdecode failed for {k}")
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return out


def make_state(mode, rng, prev_action):
    """16-dim synthetic state. 'rollout' feeds back the last step of the
    previous action chunk so you can watch chunk-to-chunk behaviour without a
    robot (rough open-loop sim, no dynamics)."""
    if mode == "rollout" and prev_action is not None:
        return prev_action[-1, :EXPECTED_ACTION_DIM].astype(np.float32)
    if mode == "random":
        arm = rng.uniform(ARM_JOINT_MIN, ARM_JOINT_MAX)
        grip = rng.uniform(GRIPPER_MIN, GRIPPER_MAX, size=2)
        return np.concatenate([arm, grip]).astype(np.float32)
    return np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)


def build_obs(images, state, prompt):
    obs = {k: img for k, img in zip(IMG_KEYS, images)}
    obs[STATE_KEY] = state
    obs[PROMPT_KEY] = prompt
    return obs


# ---------- action analysis ----------

def analyze_actions(actions, idx):
    """Print shape + per-channel ranges, return a list of warning strings."""
    warnings = []
    a = np.asarray(actions)
    log.info(f"[infer {idx}] actions shape={a.shape} dtype={a.dtype}")

    if a.ndim != 2:
        warnings.append(f"expected 2D [H, A], got ndim={a.ndim}")
        return warnings
    H, A = a.shape
    if A != EXPECTED_ACTION_DIM:
        warnings.append(f"action dim is {A}, expected {EXPECTED_ACTION_DIM} "
                        f"(14 arm + 2 gripper) — obs/checkpoint mismatch?")
    if not np.all(np.isfinite(a)):
        warnings.append("actions contain NaN/inf")

    arm = a[:, ARM_CHANNELS] if A >= 14 else a
    arm_min = arm.min(axis=0)
    arm_max = arm.max(axis=0)
    log.info(f"[infer {idx}] arm joint ranges over H={H} (rad):")
    for i, name in enumerate(ARM_JOINT_NAMES[:arm.shape[1]]):
        lo, hi = arm_min[i], arm_max[i]
        flag = ""
        if i < len(ARM_JOINT_MIN) and (lo < ARM_JOINT_MIN[i] - 1e-3 or hi > ARM_JOINT_MAX[i] + 1e-3):
            flag = "  <-- OUTSIDE controller clip range"
            warnings.append(f"{name.strip()} range [{lo:+.3f},{hi:+.3f}] "
                            f"exceeds limits [{ARM_JOINT_MIN[i]:+.2f},{ARM_JOINT_MAX[i]:+.2f}]")
        log.info(f"    {name} min={lo:+.3f} max={hi:+.3f}{flag}")

    if A > RIGHT_GRIPPER_CHANNEL:
        gl = a[:, LEFT_GRIPPER_CHANNEL]
        gr = a[:, RIGHT_GRIPPER_CHANNEL]
        log.info(f"[infer {idx}] gripper L:[{gl.min():.2f},{gl.max():.2f}] "
                 f"R:[{gr.min():.2f},{gr.max():.2f}]  (rad, valid [{GRIPPER_MIN},{GRIPPER_MAX}])")
        for tag, g in (("L", gl), ("R", gr)):
            if g.min() < GRIPPER_MIN - 1e-3 or g.max() > GRIPPER_MAX + 1e-3:
                warnings.append(f"{tag} gripper outside [{GRIPPER_MIN},{GRIPPER_MAX}]")

    # Heuristics on units
    if A >= 14:
        if np.abs(arm).max() < 0.15:
            warnings.append("arm actions are all tiny (|max|<0.15) — model may be "
                            "outputting DELTAS or NORMALIZED values, not absolute rad")
        if A > RIGHT_GRIPPER_CHANNEL:
            gmax = max(a[:, LEFT_GRIPPER_CHANNEL].max(), a[:, RIGHT_GRIPPER_CHANNEL].max())
            if 0.0 <= gmax <= 1.05:
                warnings.append("gripper actions stay in [0,1] — likely NORMALIZED, "
                                "not rad; scale by GRIPPER_MAX before set_targets")

    # First/last row for eyeballing
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    log.info(f"[infer {idx}] step[0]  = {a[0]}")
    log.info(f"[infer {idx}] step[-1] = {a[-1]}")
    return warnings


def _pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def run(args):
    log.info(f"Connecting ({args.client}) to ws://{args.server_host}:{args.server_port}")
    close_fn, infer_fn, meta = make_client(args)
    log.info(f"Connected. Server metadata: {meta}")

    rng = np.random.default_rng(args.seed)
    cam = None
    if args.use_camera:
        from g1_client.camera_client import CameraClient
        log.info(f"Opening image server at {args.image_server} (cameras only, no DDS)")
        cam = CameraClient(host=args.image_server)

    latencies = []
    all_warnings = []
    prev_action = None
    try:
        for i in range(args.iterations):
            images = _camera_images(cam) if cam else _synthetic_images(rng, args.image_mode)
            state = make_state(args.state_mode, rng, prev_action)
            obs = build_obs(images, state, args.prompt)

            t0 = time.perf_counter()
            result = infer_fn(obs)
            dt = time.perf_counter() - t0
            latencies.append(dt * 1e3)

            if "actions" not in result:
                log.error(f"[infer {i}] no 'actions' key in result; keys={list(result.keys())}")
                all_warnings.append(f"infer {i}: missing 'actions' key")
                continue
            actions = np.asarray(result["actions"], dtype=np.float64)
            log.info(f"[infer {i}] round-trip {dt*1e3:.1f} ms")
            all_warnings += [f"infer {i}: {w}" for w in analyze_actions(actions, i)]
            prev_action = actions

            if args.delay > 0 and i < args.iterations - 1:
                time.sleep(args.delay)
    finally:
        try:
            close_fn()
        except Exception:
            pass

    # ---- summary ----
    log.info("=" * 60)
    if latencies:
        log.info(f"infer latency over {len(latencies)} calls (ms): "
                 f"min={min(latencies):.1f} p50={_pct(latencies,50):.1f} "
                 f"p95={_pct(latencies,95):.1f} max={max(latencies):.1f} "
                 f"mean={np.mean(latencies):.1f}")
    if all_warnings:
        log.warning(f"{len(all_warnings)} warning(s):")
        for w in all_warnings:
            log.warning(f"  - {w}")
    else:
        log.info("No warnings — obs/action contract looks consistent.")
    log.info("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-host", required=True)
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--client", choices=["openpi", "lingbot"], default="openpi",
                   help="openpi = WebsocketClientPolicy; lingbot = your PolicyClient "
                        "(tests vendored msgpack against the openpi server)")
    p.add_argument("-n", "--iterations", type=int, default=3,
                   help="How many infer calls to make (default 3)")
    p.add_argument("--prompt", default="pick up the pink object and place it on the blue cross mark")
    p.add_argument("--use-camera", action="store_true",
                   help="Pull real frames from the image server (ZMQ only, no DDS, no motion)")
    p.add_argument("--image-server", default="192.168.123.164")
    p.add_argument("--image-mode", choices=["zero", "random"], default="zero",
                   help="Synthetic image content (ignored with --use-camera)")
    p.add_argument("--state-mode", choices=["zero", "random", "rollout"], default="zero",
                   help="Synthetic state: zero, random-within-limits, or rollout "
                        "(feed last action step back as next state)")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to sleep between infer calls")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
