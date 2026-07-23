"""Replay a recorded LeRobot episode on the G1 — open-loop playback of the
dataset's `action` column, no policy server and no cameras.

WHAT THIS IS
------------
A ground-truth motion check for the openpi datasets: stream a recorded episode's
actions straight to the arm + gripper controllers at the dataset fps, reproducing
the demonstration. It reuses openpi/main.py's safety scaffold verbatim (DDS init,
ready-pose ramp, standby Enter, kp switch, and the disable_arm_sdk cleanup) —
only the action source changes: a parquet file instead of a policy.

WHY A SEPARATE openpi/replay.py (vs lingbot_va/replay.py)
--------------------------------------------------------
The openpi datasets come in two action representations, and this script handles
both by auto-detecting from meta/info.json's `action` feature names:

  * JOINT datasets (names kLeftShoulderPitch ... kRightGripper): action[:, 0:14]
        are already absolute arm joint targets (rad) in ARM_JOINTS order — a
        direct pass-through, identical to lingbot_va/replay.py.
  * EEF datasets (names kLeftEEF_x ... kRightGripper): action[:, 0:14] are two
        7-dim EEF poses (xyz+quat, pelvis frame). These are converted back to 14
        joint targets with the SAME warm-started damped-least-squares IK that
        openpi/main_eef.py runs at dispatch (openpi/eef_kinematics.py). The whole
        episode is solved up front (before any motion) so playback stays a plain
        joint-space stream and any unreachable pose is logged before the arm moves.

Either way, after conversion the streamed contract is the [N,16] joint layout the
arm/gripper controllers expect:
    [:, 0:14] absolute arm joint targets (rad), order == ARM_JOINTS
    [:, 14]   left gripper  (rad)        [:, 15] right gripper (rad)

GRAVITY FEEDFORWARD (why replay used to sit low)
------------------------------------------------
xr_teleoperate drives the arm at collection time with a gravity-compensation
feedforward torque (arm_ik's sol_tauff = rnea(model, q, 0, 0), sent as motor
tau), so the demonstrated arm holds its commanded pose. Open-loop replay with
tau=0 has only the finite-kp PD term to fight gravity, so the arm settles below
the commanded pose (q_meas = q_cmd - tau_gravity/kp) and the replayed EEF Z reads
low. This script recomputes that same g(q) torque per frame and feeds it via
ArmController.set_arm_tauff, matching the collection dynamics. Scale/disable it
with --tauff-scale.

SAFETY NOTE: much of this data is from sim. On real hardware, replay a slow pass
first (--speed 0.3) and keep a hand on the e-stop. The arm's per-tick velocity
clamp + per-joint position limits still apply, but a sim / IK'd trajectory can
still be outside what's comfortable on your specific setup.

Precondition: robot already in 'ai' motion mode (set via the Unitree app).

Env setup (in the `unitree` env, one-time):
  pip install pyarrow        # parquet reader; numpy already present
  # pinocchio is only needed for EEF datasets (openpi/eef_kinematics.py)

Usage (run from the repo root):
  python openpi/replay.py \\
      --iface enp0s31f6 \\
      --data-root /home/ur3-exp/unitree/data/stack-cube-eef \\
      --episode 0 \\
      --speed 0.5
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import g1_client

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_client.arm_controller import ArmController, INIT_POSE_READY
from g1_client.gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_openpi.replay")

ARM_CHANNELS = slice(0, 14)
LEFT_EEF_CHANNELS = slice(0, 7)
RIGHT_EEF_CHANNELS = slice(7, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15

# IK residual (m) above which a converted EEF frame is loudly flagged.
IK_WARN_M = 0.02

ARM_JOINT_NAMES = [
    "L_pitch", "L_roll ", "L_yaw ", "L_elbow", "L_wrR ", "L_wrP ", "L_wrY ",
    "R_pitch", "R_roll ", "R_yaw ", "R_elbow", "R_wrR ", "R_wrP ", "R_wrY ",
]


# ---------- dataset loading ----------

def load_episode(data_root: Path, episode: int) -> tuple:
    """Read one LeRobot episode's action array + fps + task + action format.

    Returns (actions[N,16] float64, fps, task_str, is_eef). Resolves the parquet
    path from meta/info.json's data_path template so it works for any chunk
    layout, and detects EEF vs joint action space from the action feature names.
    """
    info = json.loads((data_root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    chunks_size = int(info.get("chunks_size", 1000))
    chunk = episode // chunks_size
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    parquet_path = data_root / rel
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode {episode} not found at {parquet_path}")

    names = info.get("features", {}).get("action", {}).get("names")
    # LeRobot nests names one level ([[...]]); flatten before matching.
    flat = names[0] if isinstance(names, list) and names and isinstance(names[0], list) else names
    is_eef = bool(flat) and any("EEF" in str(n) for n in flat)

    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 16:
        raise RuntimeError(f"Unexpected action shape {actions.shape} (want [N, >=16])")

    task = "?"
    tasks_file = data_root / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        tasks = [json.loads(l) for l in tasks_file.read_text().splitlines() if l.strip()]
        if tasks:
            task = tasks[0].get("task", "?")
    return actions, fps, task, is_eef


def eef_to_joint(actions: np.ndarray, kin) -> np.ndarray:
    """Convert an EEF-space episode [N,16] to joint-space [N,16] via warm-started
    IK (openpi/eef_kinematics.py), solving the whole episode before any motion.

    Mirrors main_eef.py's dispatch-time solve exactly: each frame's two EEF poses
    (channels 0:7, 7:14) go through solve_ik warm-started from the previous
    solution — keeping the redundant elbow DOF on one branch — and the grippers
    (14, 15) pass through untouched. The first solve is seeded from INIT_POSE_READY.
    """
    out = np.empty_like(actions)
    ik_q = np.asarray(INIT_POSE_READY, dtype=np.float64)
    ik_max_m = 0.0
    n_warn = 0
    t0 = time.time()
    for i in range(actions.shape[0]):
        a_eef = actions[i]
        ik_q, pos_err = kin.solve_ik(a_eef[LEFT_EEF_CHANNELS], a_eef[RIGHT_EEF_CHANNELS], ik_q)
        out[i, ARM_CHANNELS] = ik_q
        out[i, LEFT_GRIPPER_CHANNEL] = a_eef[LEFT_GRIPPER_CHANNEL]
        out[i, RIGHT_GRIPPER_CHANNEL] = a_eef[RIGHT_GRIPPER_CHANNEL]
        ik_max_m = max(ik_max_m, pos_err)
        if pos_err > IK_WARN_M:
            n_warn += 1
    log.info(f"IK done: {actions.shape[0]} frames in {time.time()-t0:.1f}s, "
             f"worst residual {ik_max_m*1e3:.1f} mm")
    if n_warn:
        log.warning(f"{n_warn}/{actions.shape[0]} frames had IK residual > "
                    f"{IK_WARN_M*1e3:.0f} mm (barely-reachable EEF pose) — inspect "
                    f"before running on hardware")
    return out


def compute_gravity_tauff(actions_joint: np.ndarray, kin, scale: float) -> np.ndarray:
    """Precompute the per-frame gravity-compensation feedforward torque [N,14].

    For each frame's 14 joint targets, rnea(model, q, v=0, a=0) on the same
    reduced arm model used for IK gives the static joint torque needed to hold
    that pose against gravity — the g(q) term. This is exactly what
    xr_teleoperate feeds as motor tau at collection time (arm_arm_ik's sol_tauff),
    which is why the demonstration didn't sag; feeding it here makes replay match.

    Solved up front (like the IK) so nothing runs pinocchio in the streaming hot
    loop, and so the torque range is visible before the arm moves. The per-frame
    g(q)+clip is kin.gravity_torque (shared with the inference paths).
    """
    from eef_kinematics import TAUFF_CLIP_NM
    out = np.stack([kin.gravity_torque(actions_joint[i, ARM_CHANNELS], scale)
                    for i in range(actions_joint.shape[0])])
    worst = float(np.max(np.abs(out)))
    log.info(f"gravity feedforward: scale={scale}, worst |tau| sent = {worst:.1f} N·m "
             f"(clipped at ±{TAUFF_CLIP_NM:.0f})")
    if worst >= TAUFF_CLIP_NM:
        log.warning(f"gravity tau reached the ±{TAUFF_CLIP_NM:.0f} N·m clip — inspect the "
                    f"episode/model before running on hardware")
    return out


def log_episode_ranges(actions: np.ndarray) -> None:
    """Per-joint range sanity print for the whole (joint-space) episode."""
    arm = actions[:, ARM_CHANNELS]
    gl = actions[:, LEFT_GRIPPER_CHANNEL]
    gr = actions[:, RIGHT_GRIPPER_CHANNEL]
    log.info(f"episode N={actions.shape[0]} frames, arm joint ranges (rad):")
    for i, name in enumerate(ARM_JOINT_NAMES):
        log.info(f"    {name} min={arm[:, i].min():+.3f} max={arm[:, i].max():+.3f}")
    log.info(f"gripper L:[{gl.min():.2f},{gl.max():.2f}] R:[{gr.min():.2f},{gr.max():.2f}]")


# ---------- replay loop ----------

def _run_replay(arm, grip, actions, tauff, fps, args) -> None:
    """Ramp to the episode's first frame, then stream the rest at fps/speed.

    The recorded start pose differs from INIT_POSE_READY, so frame 0 is reached
    with a smooth move_to_pose ramp (arms) + move_to_targets (grippers) before
    open-loop streaming begins — otherwise the first set_arm_target would be a
    jump that the per-tick velocity clamp would only partially absorb.

    `tauff` is either None (no gravity feedforward) or an [N,14] torque array; the
    frame-0 torque is applied before the approach ramp so the arm is already
    gravity-compensated when streaming starts, and it is zeroed again on return.
    """
    n = actions.shape[0] if args.max_frames <= 0 else min(args.max_frames, actions.shape[0])
    dt = 1.0 / (fps * args.speed)

    first = actions[0]
    if tauff is not None:
        arm.set_arm_tauff(tauff[0])
    log.info(f"Ramping to episode start over {args.approach_duration:.1f}s")
    arm.move_to_pose(first[ARM_CHANNELS], duration=args.approach_duration,
                     velocity_limit=args.velocity_limit)
    grip.move_to_targets(
        float(np.clip(first[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
        float(np.clip(first[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
        duration=args.approach_duration,
    )
    time.sleep(args.settle_duration)

    log.info(f"Streaming {n} frames at {fps:.0f} Hz x speed {args.speed} "
             f"(effective {fps*args.speed:.1f} Hz, ~{n/(fps*args.speed):.1f}s)")
    play_t0 = time.time()
    for i in range(1, n):  # frame 0 already reached by the ramp
        if arm.faulted():
            raise RuntimeError("ArmController control thread faulted — aborting")
        tic = time.time()

        a = actions[i]
        arm.set_arm_target(a[ARM_CHANNELS])
        if tauff is not None:
            arm.set_arm_tauff(tauff[i])
        grip.set_targets(
            float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
            float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
        )

        sleep = dt - (time.time() - tic)
        if sleep > 0:
            time.sleep(sleep)

    # Drop the feedforward torque before the caller ramps back to the ready pose,
    # so the return move and cleanup run with the arm's default (tau=0) dynamics.
    if tauff is not None:
        arm.set_arm_tauff(np.zeros(14))
    log.info(f"Replay done: {n} frames in {time.time()-play_t0:.1f}s")


# ---------- pipeline stages (mirrors openpi/main.py) ----------

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
    log.info("Set up the scene, then press [Enter] to start the replay.")
    log.info("Press [Ctrl+C] at any time to abort safely.")
    log.info("===============================================================")
    try:
        input("")
    except EOFError:
        log.info("EOF on stdin — proceeding without prompt")


def _cleanup(arm, grip) -> None:
    """Release every resource. disable_arm_sdk MUST run — it returns arm authority
    to the locomotion service. Each step isolated so a second Ctrl+C cannot skip
    later steps."""
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


def run(args) -> None:
    actions, recorded_fps, task, is_eef = load_episode(
        Path(args.data_root).expanduser().resolve(), args.episode)
    fps = args.fps if args.fps > 0 else recorded_fps
    log.info(f"Loaded episode {args.episode} (task={task!r}, "
             f"{'EEF' if is_eef else 'joint'} space) from {args.data_root} "
             f"| fps={fps:.0f}{' (override)' if args.fps > 0 else ''}")

    # The reduced arm model is needed for EEF->joint IK and/or the gravity
    # feedforward. Build it once, up front, and reuse for both — anything that
    # needs pinocchio happens here, before DDS/motion, so a bad model/solve is
    # caught while the robot is untouched.
    kin = None
    if is_eef or args.tauff_scale > 0:
        from eef_kinematics import G1DualArmKinematics, DEFAULT_URDF, DEFAULT_ASSETS
        urdf = args.urdf or DEFAULT_URDF
        log.info(f"Loading G1 dual-arm model from {urdf} (IK / gravity feedforward)")
        kin = G1DualArmKinematics(urdf, args.assets or DEFAULT_ASSETS)

    if is_eef:
        actions = eef_to_joint(actions, kin)
    log_episode_ranges(actions)

    # Gravity-compensation feedforward: matches how the demonstration was
    # collected (xr_teleoperate feeds sol_tauff), so the arm holds the commanded
    # pose instead of sagging below it — otherwise replayed EEF Z sits low.
    tauff = None
    if args.tauff_scale > 0:
        tauff = compute_gravity_tauff(actions, kin, args.tauff_scale)
    else:
        log.warning("gravity feedforward OFF (--tauff-scale 0): the arm will sag "
                    "below the commanded pose and replayed Z will read low")

    log.info(f"Initializing DDS on {args.iface}")
    ChannelFactoryInitialize(0, args.iface)

    arm = ArmController(publish_hz=50.0, velocity_limit=args.velocity_limit)
    arm.start()
    grip = None
    try:
        grip = GripperController(publish_hz=200.0)
        grip.start()
        _initialize_pose(arm, grip, args)
        _wait_for_operator(args)
        log.info(f"Switching arm kp to replay value: {args.replay_kp_arm}")
        arm.set_arm_kp(args.replay_kp_arm)
        _run_replay(arm, grip, actions, tauff, fps, args)
        log.info("Returning arms to ready pose")
        _initialize_pose(arm, grip, args)
    finally:
        _cleanup(arm, grip)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--data-root", default="/home/ur3-exp/unitree/data/stack-cube-eef",
                   help="LeRobot dataset root (contains meta/ and data/).")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay (default 0)")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Override playback fps; 0 = use the dataset's recorded fps (info.json).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0). Use <1 (e.g. 0.3) for a "
                        "slow, safe first pass on real hardware.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many frames; 0 = whole episode.")
    p.add_argument("--approach-duration", type=float, default=3.0,
                   help="Seconds to ramp from the ready pose to the episode's first frame.")
    # ---- Kinematics: EEF->joint IK + gravity feedforward (eef_kinematics.py) ----
    p.add_argument("--urdf", default=None,
                   help="G1 URDF for IK / gravity feedforward (defaults to the packaged model)")
    p.add_argument("--assets", default=None,
                   help="Mesh assets dir for IK / gravity feedforward (defaults to the packaged dir)")
    p.add_argument("--tauff-scale", type=float, default=1.0,
                   help="Scale on the gravity-compensation feedforward torque fed to the "
                        "arm (default 1.0 = full comp, matching collection). Use <1 (e.g. "
                        "0.5) for a cautious first hardware pass, or 0 to disable (the arm "
                        "will then sag and replayed Z reads low). Requires pinocchio.")
    # ---- Safety / motion limits (same defaults as openpi/main.py) ----
    p.add_argument("--velocity-limit", type=float, default=8.0,
                   help="rad/s velocity cap on the per-tick motion clamp (default 8.0)")
    p.add_argument("--replay-kp-arm", type=float, default=80.0,
                   help="kp for shoulder/elbow during replay (default 80, matching inference)")
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
