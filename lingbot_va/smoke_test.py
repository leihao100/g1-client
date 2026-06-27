"""Async-contract smoke test — no robot, no cameras, server only.

Drives the g1_async wire contract against a running server with synthetic
JPEG observations and asserts the shapes/return keys the real client relies
on, BEFORE involving the physical robot:

  reset(prompt)
  cold_start(obs=<single dict>)            -> {action: C0}, (16,2,16)
  async_step(executing_action=C0)          -> {action: C1}  # cycle 0: no ground
  async_step(obs=<4 kf>, state=C0, executing_action=C1) -> {action: C2}
  async_step(obs=<8 kf>, state=C1, executing_action=C2) -> {action: C3}

This mirrors the first three cycles of _run_inference_loop's schedule:
cycle 0 sends no obs/state (server skips grounding for the cold-start
chunk), then 4 then 8 keyframes. It needs a running CONFIG_NAME=g1_async
server reachable at --server-host/--server-port; it cannot run without one
(PolicyClient retries the connect forever).

Usage (run from the repo root):
    python lingbot_va/smoke_test.py --server-host localhost --server-port 29536
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import g1_client

from g1_client.policy_client import PolicyClient


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("lingbot_g1.smoke")

EXPECTED_SHAPE = (16, 2, 16)


def _jpeg() -> bytes:
    # 256x320 BGR JPEG q90 — same wire format CameraClient produces.
    arr = (np.random.rand(256, 320, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _frame_dict() -> dict:
    return {
        "observation.images.cam_left_high":   _jpeg(),
        "observation.images.cam_left_wrist":  _jpeg(),
        "observation.images.cam_right_wrist": _jpeg(),
    }


def single_obs(prompt: str) -> dict:
    """One cold-start observation dict (3 cams + task), as cam.get_obs."""
    obs = _frame_dict()
    obs["task"] = prompt
    return obs


def keyframes(n: int) -> list:
    """A list of n camera-only keyframe dicts, as execute_chunk returns."""
    return [_frame_dict() for _ in range(n)]


def _check_chunk(name: str, arr) -> None:
    if not isinstance(arr, np.ndarray):
        raise AssertionError(f"{name}: expected ndarray, got {type(arr)}")
    if arr.shape != EXPECTED_SHAPE:
        raise AssertionError(f"{name}: shape {arr.shape}, expected {EXPECTED_SHAPE}")
    log.info(f"  {name}: shape {arr.shape} dtype {arr.dtype} "
             f"range [{arr.min():.3f}, {arr.max():.3f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-host", required=True)
    p.add_argument("--server-port", type=int, default=29536)
    p.add_argument("--prompt",
                   default="pick up the pink object and place it on the blue cross mark")
    args = p.parse_args()

    policy = PolicyClient(host=args.server_host, port=args.server_port)
    log.info(f"Connected. Server metadata: {policy.get_server_metadata()}")

    log.info(f"reset({args.prompt!r})")
    policy.reset(args.prompt)

    log.info("cold_start: 1 init frame in, expecting C0 only")
    r = policy.infer({"cold_start": True, "obs": single_obs(args.prompt)})
    if "action" not in r or "action1" in r:
        log.error(f"single-chunk cold_start must return only 'action'; "
                  f"got keys={sorted(r)}")
        return 1
    C0 = r["action"]
    _check_chunk("cold_start C0", C0)

    log.info("async_step #0 (cycle 0): no obs/state, C0 executing — "
             "server skips grounding for C0 and predicts C1")
    r0 = policy.infer({"async_step": True, "executing_action": C0})
    if "action" not in r0:
        log.error(f"async_step must return 'action'; got keys={sorted(r0)}")
        return 1
    C1 = r0["action"]
    _check_chunk("async_step #0 -> C1", C1)

    log.info("async_step #1: ground C0 (4 keyframes), C1 executing")
    r1 = policy.infer({"async_step": True, "obs": keyframes(4),
                       "state": C0, "executing_action": C1})
    if "action" not in r1:
        log.error(f"async_step must return 'action'; got keys={sorted(r1)}")
        return 1
    C2 = r1["action"]
    _check_chunk("async_step #1 -> C2", C2)

    log.info("async_step #2: ground C1 (8 keyframes), C2 executing")
    r2 = policy.infer({"async_step": True, "obs": keyframes(8),
                       "state": C1, "executing_action": C2})
    if "action" not in r2:
        log.error(f"async_step must return 'action'; got keys={sorted(r2)}")
        return 1
    C3 = r2["action"]
    _check_chunk("async_step #2 -> C3", C3)

    policy.close()
    log.info("OK — single-chunk cold_start (C0); async_step #0 predicted C1 "
             "with no grounding; cycles #1/#2 grounded C0/C1 (4 then 8 kf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
