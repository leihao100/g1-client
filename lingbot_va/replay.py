"""Replay a recorded LeRobot episode on the G1 — open-loop playback of the
dataset's `action` column, no policy server and no cameras.

WHAT THIS IS
------------
A ground-truth motion check: stream a recorded episode's actions straight to the
arm + gripper controllers at the dataset fps, reproducing the demonstration. It
reuses main_fastwam.py's safety scaffold verbatim (DDS init, ready-pose ramp,
standby Enter, kp switch, and the disable_arm_sdk cleanup) — only the action
source changes: a parquet file instead of a policy.

The pick_red dataset stores `action` in exactly the controller layout, in raw
units already compatible with the hardware (verified: arm joints in rad, grippers
in [GRIPPER_MIN, GRIPPER_MAX]=[0,5.4]) — so playback is a direct pass-through, the
same [H,16] contract the FastWAM/openpi action chunks use:
    [:, 0:14] absolute arm joint targets (rad), order == ARM_JOINTS
    [:, 14]   left gripper  (rad)        [:, 15] right gripper (rad)

SAFETY NOTE: this data is from `Unitree_G1_Dex1_Sim`. On real hardware, replay a
slow pass first (--speed 0.3) and keep a hand on the e-stop. The arm's per-tick
velocity clamp + per-joint position limits still apply, but a sim trajectory can
still be outside what's comfortable on your specific setup.

Precondition: robot already in 'ai' motion mode (set via the Unitree app).

Env setup (in the `unitree` env, one-time):
  pip install pyarrow        # parquet reader; numpy already present

Usage (run from the repo root):
  python lingbot_va/replay.py \\
      --iface enp0s31f6 \\
      --data-root /home/ur3-exp/unitree/data/pick_red \\
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
log = logging.getLogger("g1_replay.main")

ARM_CHANNELS = slice(0, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15

ARM_JOINT_NAMES = [
    "L_pitch", "L_roll ", "L_yaw ", "L_elbow", "L_wrR ", "L_wrP ", "L_wrY ",
    "R_pitch", "R_roll ", "R_yaw ", "R_elbow", "R_wrR ", "R_wrP ", "R_wrY ",
]


# ---------- dataset loading ----------

def load_episode(data_root: Path, episode: int) -> tuple:
    """Read one LeRobot episode's action array + the dataset fps + task string.

    Returns (actions[N,16] float64, fps, task_str). Resolves the parquet path from
    meta/info.json's data_path template so it works for any chunk layout.
    """
    info = json.loads((data_root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    chunks_size = int(info.get("chunks_size", 1000))
    chunk = episode // chunks_size
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    parquet_path = data_root / rel
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode {episode} not found at {parquet_path}")

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
    return actions, fps, task


def log_episode_ranges(actions: np.ndarray) -> None:
    """Per-joint range sanity print for the whole episode before any motion."""
    arm = actions[:, ARM_CHANNELS]
    gl = actions[:, LEFT_GRIPPER_CHANNEL]
    gr = actions[:, RIGHT_GRIPPER_CHANNEL]
    log.info(f"episode N={actions.shape[0]} frames, arm joint ranges (rad):")
    for i, name in enumerate(ARM_JOINT_NAMES):
        log.info(f"    {name} min={arm[:, i].min():+.3f} max={arm[:, i].max():+.3f}")
    log.info(f"gripper L:[{gl.min():.2f},{gl.max():.2f}] R:[{gr.min():.2f},{gr.max():.2f}]")


# ---------- replay loop ----------

def _run_replay(arm, grip, actions, fps, args) -> None:
    """Ramp to the episode's first frame, then stream the rest at fps/speed.

    The recorded start pose differs from INIT_POSE_READY, so frame 0 is reached
    with a smooth move_to_pose ramp (arms) + move_to_targets (grippers) before
    open-loop streaming begins — otherwise the first set_arm_target would be a
    jump that the per-tick velocity clamp would only partially absorb.
    """
    n = actions.shape[0] if args.max_frames <= 0 else min(args.max_frames, actions.shape[0])
    dt = 1.0 / (fps * args.speed)

    first = actions[0]
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
        grip.set_targets(
            float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
            float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
        )

        sleep = dt - (time.time() - tic)
        if sleep > 0:
            time.sleep(sleep)

    log.info(f"Replay done: {n} frames in {time.time()-play_t0:.1f}s")


# ---------- pipeline stages (mirrors main_fastwam.py) ----------

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
    actions, recorded_fps, task = load_episode(
        Path(args.data_root).expanduser().resolve(), args.episode)
    fps = args.fps if args.fps > 0 else recorded_fps
    log.info(f"Loaded episode {args.episode} (task={task!r}) from {args.data_root} "
             f"| fps={fps:.0f}{' (override)' if args.fps > 0 else ''}")
    log_episode_ranges(actions)

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
        _run_replay(arm, grip, actions, fps, args)
        log.info("Returning arms to ready pose")
        _initialize_pose(arm, grip, args)
    finally:
        _cleanup(arm, grip)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--data-root", default="/home/ur3-exp/unitree/data/pick_red",
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
    # ---- Safety / motion limits (same defaults as main_fastwam.py) ----
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
