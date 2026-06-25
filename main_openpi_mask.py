"""Masked / weighted-blend inference loop for the G1 against a standard openpi
serve_policy.py server (32-action-horizon checkpoint).

WHAT CHANGED vs main_openpi.py
------------------------------
Nothing in the robot control path. arm_controller.py, gripper_controller.py and
camera_client.py are used verbatim, and build_obs / the init / cleanup / standby
sequence are imported unchanged from main_openpi.py. The ONLY new thing is the
chunk-scheduling + blending policy in `_run_masked_loop`.

THE SCHEDULE (for a horizon-32 model)
-------------------------------------
A single merge cycle is `lead + wait` control steps (7 + 7 = 14 by default):

    command 7 steps from the current plan
      -> snapshot an obs and fire the next inference on a daemon thread
    command 7 more steps  (inference runs hidden behind this window)
      -> join the inference: a fresh 32-step chunk arrives
      -> MERGE it into the plan, repeat

TIME ALIGNMENT
--------------
The prefetch obs is taken `wait` steps before we adopt the new chunk, so the
chunk's first `wait` steps are already "in the past" by the time it lands. We
drop exactly those leading steps (`new_chunk[wait:]`) so the new prediction is
wall-clock aligned with the plan it is replacing — otherwise the arm snaps back
to a stale pose then forward again.

THE WEIGHTED MERGE  (the "mask")
--------------------------------
`old_future[k]` and `new_aligned[k]` are two predictions for the SAME future
timestep. We cross-fade them with a weight that ramps the old:new ratio from
1:1 at the first overlapping step to 0:1 at the last:

    a_new(k) = 0.5 + 0.5 * k/(L-1)      # 0.5 -> 1.0 over the overlap
    plan[k]  = (1 - a_new)*old_future[k] + a_new*new_aligned[k]

so the freshly predicted chunk eases in instead of snapping. Whatever of the new
chunk extends past the old plan (the non-overlapping tail) is appended verbatim.

Usage (run from the g1-client/ directory):
  python main_openpi_mask.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 8000 \\
      --prompt "pick up the pink object and place it on the blue cross mark"
"""

import argparse
import logging
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_client.arm_controller import ArmController
from g1_client.gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX
from g1_client.camera_client import CameraClient
from g1_client.policy_client import PolicyClient

# Reuse everything that is identical to the plain openpi client — controllers are
# untouched, only the scheduling/blending policy below is new.
from main_openpi import (
    ARM_CHANNELS, LEFT_GRIPPER_CHANNEL, RIGHT_GRIPPER_CHANNEL,
    build_obs, log_chunk_ranges, _infer_worker,
    _initialize_pose, _wait_for_operator, _cleanup,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_openpi.mask")


# ---------- the weighted overlap merge (the "mask") ----------

def _merge_chunks(old_future: np.ndarray, new_aligned: np.ndarray) -> np.ndarray:
    """Weighted temporal blend of the old plan tail and the time-aligned new chunk.

    old_future[k] and new_aligned[k] are two predictions for the same future
    timestep. Over the overlap the old:new weight ramps 1:1 -> 0:1 (a_new
    0.5 -> 1.0), so the new chunk fades in. The longer chunk's non-overlapping
    tail is appended unblended.

    In this loop's cadence new_aligned is always >= old_future, so the appended
    tail is the new chunk's extra steps — but the old-extends-further branch is
    kept so a slow inference (short new chunk) can't drop planned steps.
    """
    Lo, Ln = len(old_future), len(new_aligned)
    L = min(Lo, Ln)
    out = np.empty((max(Lo, Ln), old_future.shape[1]), dtype=np.float64)
    for k in range(L):
        a_new = 1.0 if L == 1 else 0.5 + 0.5 * (k / (L - 1))
        out[k] = (1.0 - a_new) * old_future[k] + a_new * new_aligned[k]
    if Ln > L:
        out[L:] = new_aligned[L:]
    elif Lo > L:
        out[L:] = old_future[L:]
    return out


# ---------- inference loop ----------

def _run_masked_loop(arm, grip, cam, policy, args) -> None:
    """Receding-horizon loop: command `lead` steps, prefetch, command `wait`
    steps, then merge the prefetched chunk with a weighted overlap blend.

    The arm/gripper publish threads keep holding the last commanded target while
    `th.join()` blocks, so even if inference overruns the `wait` window the worst
    case is a brief hold, not an unsafe state.
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt
    lead = args.exec_before_prefetch          # steps before firing the prefetch
    wait = args.infer_wait                    # steps the inference overlaps with
    cycle = lead + wait                       # steps commanded per merge
    skip = wait if args.time_align else 0     # stale leading steps to drop
    if args.send_jpeg:
        log.warning("--send-jpeg ON: sending compressed JPEG bytes. The SERVER must "
                    "imdecode + cv2.COLOR_BGR2RGB these keys, or it sees garbage.")

    # First chunk is a blocking infer (nothing to overlap it against yet).
    log.info(f"First inference (prompt={prompt!r})")
    result = policy.infer(build_obs(cam, arm, grip, prompt, args.send_jpeg))
    plan = np.asarray(result["actions"], dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] < 16:
        raise RuntimeError(f"Unexpected action shape {plan.shape} (want [H, 16])")
    if plan.shape[0] <= skip + cycle:
        raise RuntimeError(f"horizon {plan.shape[0]} too short for lead={lead} "
                           f"wait={wait} (need > {skip + cycle})")
    log_chunk_ranges(0, plan)

    ptr = 0              # index of the next step to command within `plan`
    since_merge = 0      # steps commanded since the last merge
    prefetch_fired = False
    box: dict = {}
    th = None

    for merges in range(1, args.max_chunks + 1):
        # Command `cycle` steps from the current plan, firing the prefetch
        # `wait` steps before the end of the window.
        for _ in range(cycle):
            if arm.faulted():
                raise RuntimeError("ArmController control thread faulted — aborting")
            if ptr >= len(plan):
                raise RuntimeError(f"plan exhausted (ptr={ptr}, len={len(plan)}) — "
                                   f"inference slower than {wait} steps")
            tic = time.time()

            a = plan[ptr]
            arm.set_arm_target(a[ARM_CHANNELS])
            grip.set_targets(
                float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
            )
            ptr += 1
            since_merge += 1

            # After `lead` steps, snapshot an obs and fire the next inference so
            # it overlaps the remaining `wait` steps of this window.
            if since_merge == lead and not prefetch_fired:
                obs_next = build_obs(cam, arm, grip, prompt, args.send_jpeg)
                box = {}
                th = threading.Thread(target=_infer_worker, args=(policy, obs_next, box),
                                      daemon=True, name=f"prefetch-{merges}")
                th.start()
                prefetch_fired = True

            sleep = dt - (time.time() - tic)
            if sleep > 0:
                time.sleep(sleep)

        # Adopt the prefetched chunk and merge it into the plan.
        join_t0 = time.time()
        th.join()
        join_wait_ms = (time.time() - join_t0) * 1e3
        if "err" in box:
            raise box["err"]
        new_chunk = box["actions"]            # [H, 16], new_chunk[0] ~ obs@(merge-wait)
        new_aligned = new_chunk[skip:]        # drop the `wait` already-elapsed steps
        old_future = plan[ptr:]               # what the old plan still has queued
        plan = _merge_chunks(old_future, new_aligned)
        ptr = 0
        since_merge = 0
        prefetch_fired = False
        th = None

        overlap = min(len(old_future), len(new_aligned))
        appended = max(0, len(new_aligned) - len(old_future))
        log.info(f"[merge {merges}] overlap={overlap} append={appended} "
                 f"plan_len={len(plan)} join_wait={join_wait_ms:.0f}ms"
                 + (f" STALLED" if join_wait_ms > 1.0 else ""))
        log_chunk_ranges(merges, plan)


# ---------- entry point ----------

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
        _run_masked_loop(arm, grip, cam, policy, args)
        _initialize_pose(arm, grip, args)
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
    p.add_argument("--send-jpeg", action="store_true",
                   help="Send compressed JPEG bytes instead of decoded RGB arrays "
                        "(~12x smaller upload). REQUIRES the server to imdecode + "
                        "BGR->RGB these image keys.")
    p.add_argument("--max-chunks", type=int, default=30,
                   help="How many merge cycles to run before stopping")
    p.add_argument("--control-hz", type=float, default=15.0,
                   help="Per-step dispatch rate; match your LeRobot recording fps")
    # ---- Masked-blend schedule (horizon-32 defaults) ----
    p.add_argument("--exec-before-prefetch", type=int, default=7,
                   help="Steps to command before firing the next inference (default 7)")
    p.add_argument("--infer-wait", type=int, default=7,
                   help="Steps to command while the inference runs; also the number "
                        "of stale leading steps dropped for time-alignment (default 7)")
    p.add_argument("--no-time-align", action="store_false", dest="time_align",
                   help="Do not drop the new chunk's leading `infer_wait` steps. By "
                        "default they are dropped so the new chunk is wall-clock "
                        "aligned with the plan it replaces; pass this for A/B testing.")
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
