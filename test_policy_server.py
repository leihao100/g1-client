"""Latency test for the G1 openpi policy server — NO ROBOT MOTION.

Drives main_openpi's REAL inference loop (`_run_inference_loop`: prefetch +
time-alignment + cross-fade) but with self-generated images and state instead
of a camera / DDS joint feedback, and with the arm/gripper writes stubbed out so
nothing moves. The point is to measure inference latency *under the exact call
pattern main_openpi uses* — including whether each infer fits inside the prefetch
overlap window (if it doesn't, the real loop stalls at the chunk boundary = the
"顿").

Because it imports main_openpi to reuse the loop verbatim, this file now needs
`unitree_sdk2py` and `teleimager` installed (they are imported transitively, but
never used — no DDS channel is opened, no robot is touched). If you want a
zero-dependency server smoke test, run an older revision of this file.

What you get
------------
  * connection + server metadata handshake
  * the full main_openpi loop run against synthetic obs (max-chunks chunks)
  * the loop's achieved in-loop FPS (sub-steps actually dispatched per second),
    vs the target control_hz — chunk-boundary stalls show up as FPS < control_hz
  * per-infer latency + per-chunk action analysis (shape, NaN/inf, ranges, unit
    heuristics) — the NaN check is handy when a fresh checkpoint returns all-NaN
  * the overlap budget (prefetch_lead / control_hz): if p95 latency exceeds it,
    the loop stalls at chunk boundaries (and the measured FPS will confirm it)

Client: the vendored g1_client.policy_client.PolicyClient (msgpack wire). No
openpi_client dependency. Synthetic obs is self-generated (--image-mode /
--state-mode); nothing is read from a camera or the robot.

Usage:
  python test_policy_server.py --server-host 1.2.3.4 --server-port 8000 --max-chunks 10
  python test_policy_server.py --server-host 1.2.3.4 --image-mode random --state-mode random
"""

import argparse
import logging
import threading
import time

import cv2
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

# ---- Limits inlined (NOT imported from arm_controller, so the synthetic-state
#      generator has no robot dependency). Keep in sync with
#      arm_controller.ARM_JOINT_MIN/MAX. ----
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


# ---------- client backend ----------

def make_client(args):
    """Connect via the vendored PolicyClient (no openpi_client dependency).

    Returns the client itself (not just its .infer) so the timing wrapper can
    read PolicyClient.last_timing — the pack/send/wait_recv/unpack breakdown of
    each call, which is what lets us localize the latency bottleneck."""
    from g1_client.policy_client import PolicyClient
    return PolicyClient(host=args.server_host, port=args.server_port)


# ---------- self-generated obs content ----------

def _synth_arm_q(rng, mode):
    """Synthetic 14-dim arm joint vector (zeros or random within limits)."""
    if mode == "random":
        return rng.uniform(ARM_JOINT_MIN, ARM_JOINT_MAX).astype(np.float64)
    return np.zeros(14, dtype=np.float64)


def _synth_grip(rng, mode):
    """Synthetic (left, right) gripper q."""
    if mode == "random":
        g = rng.uniform(GRIPPER_MIN, GRIPPER_MAX, size=2)
        return float(g[0]), float(g[1])
    mid = (GRIPPER_MIN + GRIPPER_MAX) / 2.0
    return mid, mid


def _synth_jpeg(rng, mode):
    """One synthetic camera frame, JPEG-encoded (matches camera_client output)."""
    shape = (CAM_HEIGHT, CAM_WIDTH, 3)
    arr = (rng.integers(0, 256, shape, dtype=np.uint8) if mode == "random"
           else np.zeros(shape, dtype=np.uint8))
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed building synthetic frame")
    return buf.tobytes()


# ---------- fakes that stand in for the real controllers ----------
# Each implements only the methods main_openpi's loop / build_obs call. The
# set_* writes are no-ops (nothing moves); the get_* reads return self-generated
# content. So the loop, prefetch, alignment and blend all run for real — only
# the obs source and the robot output are faked.

class _FakeArm:
    def __init__(self, rng, state_mode):
        self._rng = rng
        self._mode = state_mode
        # In-loop FPS instrumentation: every sub-step dispatch lands here.
        self.steps = 0
        self.t_first = None
        self.t_last = None

    def get_arm_q(self):
        return _synth_arm_q(self._rng, self._mode)

    def set_arm_target(self, q):  # would drive DDS; here it just times the tick
        now = time.perf_counter()
        if self.t_first is None:
            self.t_first = now
        self.t_last = now
        self.steps += 1

    def faulted(self):
        return False


class _FakeGrip:
    def __init__(self, rng, state_mode):
        self._rng = rng
        self._mode = state_mode

    def get_state(self):
        return _synth_grip(self._rng, self._mode)

    def set_targets(self, left, right):  # no-op
        pass


class _FakeCam:
    def __init__(self, rng, image_mode):
        self._rng = rng
        self._mode = image_mode

    def get_obs_images(self):
        return {k: _synth_jpeg(self._rng, self._mode) for k in IMG_KEYS}


# ---------- action analysis ----------

def analyze_actions(actions, idx, quiet=False):
    """Return contract/NaN/unit warnings for one action chunk. quiet=False also
    logs shape + per-channel ranges; quiet=True only collects warnings (the
    reused main_openpi loop already prints ranges via log_chunk_ranges)."""
    warnings = []
    a = np.asarray(actions)
    if not quiet:
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
    if not quiet:
        log.info(f"[infer {idx}] arm joint ranges over H={H} (rad):")
    for i, name in enumerate(ARM_JOINT_NAMES[:arm.shape[1]]):
        lo, hi = arm_min[i], arm_max[i]
        flag = ""
        if i < len(ARM_JOINT_MIN) and (lo < ARM_JOINT_MIN[i] - 1e-3 or hi > ARM_JOINT_MAX[i] + 1e-3):
            flag = "  <-- OUTSIDE controller clip range"
            warnings.append(f"{name.strip()} range [{lo:+.3f},{hi:+.3f}] "
                            f"exceeds limits [{ARM_JOINT_MIN[i]:+.2f},{ARM_JOINT_MAX[i]:+.2f}]")
        if not quiet:
            log.info(f"    {name} min={lo:+.3f} max={hi:+.3f}{flag}")

    if A > RIGHT_GRIPPER_CHANNEL:
        gl = a[:, LEFT_GRIPPER_CHANNEL]
        gr = a[:, RIGHT_GRIPPER_CHANNEL]
        if not quiet:
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

    return warnings


# ---------- policy shim ----------

class _TimingPolicy:
    """Thin shim around PolicyClient for the reused main_openpi loop. It just
    delegates infer (and exposes .last_timing so the loop can read its own
    latency breakdown — main_openpi prints the per-step table now) and, on top,
    runs the action contract/NaN/unit analysis the loop doesn't do. The loop
    calls .infer on the main thread (first chunk) and prefetch daemon threads
    (steady chunks), so the counters are lock-guarded."""

    def __init__(self, client):
        self._client = client
        self._lock = threading.Lock()
        self.warnings = []
        self.nan_calls = 0
        self._n = 0

    @property
    def last_timing(self):
        return self._client.last_timing

    def infer(self, obs):
        with self._lock:
            idx = self._n
            self._n += 1
        result = self._client.infer(obs)
        ws = []
        if "actions" not in result:
            ws.append("missing 'actions' key")
        else:
            actions = np.asarray(result["actions"], dtype=np.float64)
            ws = analyze_actions(actions, idx, quiet=True)
        with self._lock:
            self.warnings += [f"infer {idx}: {w}" for w in ws]
            if any("NaN" in w for w in ws):
                self.nan_calls += 1
        return result


# ---------- driver ----------

def run(args):
    # Importing main_openpi pulls in unitree_sdk2py + teleimager (transitively),
    # but opens no DDS channel and touches no robot — we only call its loop.
    from main_openpi import _run_inference_loop

    log.info(f"Connecting (PolicyClient) to ws://{args.server_host}:{args.server_port}")
    client = make_client(args)
    log.info(f"Connected. Server metadata: {client.get_server_metadata()}")

    rng = np.random.default_rng(args.seed)
    arm = _FakeArm(rng, args.state_mode)
    grip = _FakeGrip(rng, args.state_mode)
    cam = _FakeCam(rng, args.image_mode)
    policy = _TimingPolicy(client)

    log.info(f"Driving main_openpi._run_inference_loop with synthetic obs: "
             f"max_chunks={args.max_chunks} control_hz={args.control_hz} "
             f"prefetch_lead={args.prefetch_lead} blend_steps={args.blend_steps} "
             f"chunk_align={args.chunk_align} send_jpeg={args.send_jpeg}")
    try:
        # The loop itself prints the per-step latency breakdown + bottleneck +
        # stall verdict (main_openpi._summarize_timing).
        _run_inference_loop(arm, grip, cam, policy, args)
    finally:
        try:
            client.close()
        except Exception:
            pass

    # ---- this harness adds: in-loop FPS + contract/NaN checks ----
    log.info("=" * 64)
    # FPS: sub-steps dispatched per second (first→last dispatch). A chunk-
    # boundary stall leaves a gap with no dispatch, dragging it below control_hz.
    if arm.steps > 1 and arm.t_last and arm.t_last > arm.t_first:
        span = arm.t_last - arm.t_first
        fps = (arm.steps - 1) / span
        ratio = fps / args.control_hz
        log.info(f"in-loop FPS: {arm.steps} sub-steps over {span:.2f}s (first->last) "
                 f"=> {fps:.2f} fps   (target control_hz={args.control_hz}, "
                 f"{ratio*100:.0f}% of target)")
        if ratio < 0.95:
            log.warning(f"achieved {fps:.2f} fps is only {ratio*100:.0f}% of target "
                        f"{args.control_hz} — chunk-boundary stalls are dropping the rate.")

    n_infers = policy._n
    if policy.nan_calls:
        log.warning(f"{policy.nan_calls}/{n_infers} infer calls returned NaN/inf actions "
                    f"— checkpoint norm_stats or weights problem (synthetic obs is finite, "
                    f"so the server produced them).")
    if policy.warnings:
        log.warning(f"{len(policy.warnings)} contract warning(s):")
        for w in policy.warnings:
            log.warning(f"  - {w}")
    elif n_infers:
        log.info("No contract warnings — obs/action looks consistent.")
    log.info("=" * 64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-host", required=True)
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--prompt", default="pick up the pink object and place it on the blue cross mark")
    p.add_argument("--send-jpeg", action="store_true",
                   help="Send compressed JPEG bytes instead of decoded RGB arrays "
                        "(~12x smaller upload). Use to measure the network win; the "
                        "server must imdecode + BGR->RGB these image keys.")
    # ---- self-generated obs content ----
    p.add_argument("--image-mode", choices=["zero", "random"], default="zero",
                   help="Synthetic image content")
    p.add_argument("--state-mode", choices=["zero", "random"], default="zero",
                   help="Synthetic state: zeros, or random-within-limits")
    p.add_argument("--seed", type=int, default=0)
    # ---- main_openpi loop knobs (names/defaults mirror main_openpi) ----
    p.add_argument("--max-chunks", type=int, default=10,
                   help="How many action chunks the reused loop runs (default 10)")
    p.add_argument("--control-hz", type=float, default=15.0,
                   help="Per-step dispatch rate of the reused loop (default 15)")
    p.add_argument("--exec-steps", type=int, default=0,
                   help="Steps to execute per chunk before re-querying; 0 = full horizon")
    p.add_argument("--prefetch-lead", type=int, default=5,
                   help="Start the next inference when this many steps remain (default 5)")
    p.add_argument("--blend-steps", type=int, default=5,
                   help="Cross-fade length at chunk boundary (default 5; 0 disables)")
    p.add_argument("--no-chunk-align", action="store_false", dest="chunk_align",
                   help="Disable chunk time-alignment in the reused loop")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
