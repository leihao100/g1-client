"""Offline contract test for the async FDM inference loop (Algorithm 2).

No server, no robot, no DDS — injects fakes for arm/grip/cam/policy/args and
drives _run_inference_loop, asserting the EXACT wire schedule:

  reset(prompt)
  cold_start(obs=<single dict>)            -> {action: C0, action1: C1}
  per cycle n=1..max_chunks:
     async_step(obs=K_{n-1}, state=a_{n-1}, executing_action=C_n) -> {action: C_{n+1}}

Pins the silent-desync-class invariants:
  * message kinds are exactly {reset, cold_start, async_step} — never
    compute_kv_cache, never a plain {obs,prompt} infer
  * cold_start obs is a single dict (1 frame); async_step obs is a list,
    length 4 on the FIRST async_step (C0 ran is_first_chunk=True), 8 after
  * state / executing_action are the SAME ndarray objects the fake server
    returned (verbatim pass-through, no copy/reshape/renorm)
  * one-cycle lag/rotation: at cycle n, state == chunk executed at n-1,
    executing_action == chunk executed at n, obs == keyframes from n-1
  * the async_step request runs on a NON-main (daemon) thread

Run:  conda activate unitree_deploy && python test_async_loop.py
Exit 0 = pass, 1 = fail.
"""

import sys
import threading
import types

import numpy as np

from main import _run_inference_loop

MAIN_IDENT = threading.get_ident()


class FakeArm:
    def faulted(self):
        return False

    def set_arm_target(self, q):
        pass


class FakeGrip:
    def set_targets(self, l, r):
        pass


class FakeCam:
    """get_obs -> single obs dict (cold start); get_obs_images -> one keyframe
    dict per call (execute_chunk appends one per CAPTURE_EVERY window)."""

    def get_obs(self, prompt):
        return {"observation.images.cam_left_high": b"jpeg", "task": prompt}

    def get_obs_images(self):
        return {"observation.images.cam_left_high": b"jpeg"}


class FakePolicy:
    """Records every infer() payload, the thread it ran on, and replays the
    g1_async server side. Each returned chunk is a uniquely-tagged (16,2,16)
    array so rotation/identity can be asserted."""

    def __init__(self):
        self.calls = []          # list of (kind, payload, thread_ident)
        self.returned = []       # every chunk array the "server" emitted, in order
        self.last_timing = None
        self._n = 0

    def _chunk(self):
        self._n += 1
        a = np.full((16, 2, 16), float(self._n), dtype=np.float64)
        self.returned.append(a)
        return a

    def infer(self, payload):
        ident = threading.get_ident()
        if payload.get("reset"):
            self.calls.append(("reset", payload, ident))
            return {}
        if payload.get("cold_start"):
            self.calls.append(("cold_start", payload, ident))
            return {"action": self._chunk(), "action1": self._chunk()}
        if payload.get("async_step"):
            self.calls.append(("async_step", payload, ident))
            return {"action": self._chunk()}
        self.calls.append(("UNKNOWN", payload, ident))
        raise AssertionError(f"Unexpected infer payload keys: {sorted(payload)}")

    def reset(self, prompt):
        self.infer({"reset": True, "prompt": prompt})


def _args(max_chunks):
    return types.SimpleNamespace(
        prompt="pick up the pink object and place it on the blue cross mark",
        max_chunks=max_chunks,
        substep_hz=1_000_000.0,   # tiny substep_dt → no real sleeping
        quiet_substeps=True,
        video_guidance=5.0,
        action_guidance=1.0,
    )


def main():
    N = 5
    pol = FakePolicy()
    _run_inference_loop(FakeArm(), FakeGrip(), FakeCam(), pol, _args(N))

    kinds = [k for k, _, _ in pol.calls]

    # 1. message order: reset, cold_start, then exactly N async_step
    assert kinds == ["reset", "cold_start"] + ["async_step"] * N, \
        f"wrong message schedule: {kinds}"

    # 2. no forbidden messages anywhere
    for k, p, _ in pol.calls:
        assert "compute_kv_cache" not in p, "compute_kv_cache must not appear in async path"
        if k == "async_step":
            assert set(p) >= {"async_step", "obs", "state", "executing_action"}, \
                f"async_step missing keys: {sorted(p)}"
            assert "prompt" not in p, "async_step must not carry prompt"

    # 3. cold_start obs is a SINGLE dict (1 frame), not a list
    _, cs_payload, cs_ident = pol.calls[1]
    assert isinstance(cs_payload["obs"], dict), \
        f"cold_start obs must be a dict, got {type(cs_payload['obs'])}"
    assert cs_ident == MAIN_IDENT, "cold_start must run on the main thread"
    assert pol.calls[0][2] == MAIN_IDENT, "reset must run on the main thread"

    # server's two cold-start chunks
    C0, C1 = pol.returned[0], pol.returned[1]

    # 4. per-cycle async_step contract: keyframe count, rotation, identity, thread
    async_calls = pol.calls[2:]
    for i, (_, p, ident) in enumerate(async_calls):  # i=0 is the FIRST async_step
        # 4a. keyframe count: 4 on the first (C0 ran is_first_chunk=True), 8 after
        expected_kf = 4 if i == 0 else 8
        assert isinstance(p["obs"], list), "async_step obs must be a list"
        assert len(p["obs"]) == expected_kf, \
            f"async_step #{i}: obs has {len(p['obs'])} keyframes, expected {expected_kf}"

        # 4b. rotation + verbatim identity pass-through
        executed_this_cycle = pol.returned[i + 1]   # C1 at i=0, then C2, C3, ...
        executed_prev_cycle = pol.returned[i]       # C0 at i=0, then C1, C2, ...
        assert p["executing_action"] is executed_this_cycle, \
            f"async_step #{i}: executing_action is not the chunk running this cycle (identity)"
        assert p["state"] is executed_prev_cycle, \
            f"async_step #{i}: state is not the previous executed chunk (identity)"

        # 4c. request runs OFF the main thread (Branch B daemon)
        assert ident != MAIN_IDENT, \
            f"async_step #{i} ran on the main thread; must be a daemon thread"

    # explicit spelling-out of the first cycle (the one most prone to desync)
    assert async_calls[0][1]["state"] is C0, "first async_step must ground C0"
    assert async_calls[0][1]["executing_action"] is C1, \
        "first async_step must report C1 as executing"
    assert len(async_calls[0][1]["obs"]) == 4, \
        "first async_step must carry exactly 4 keyframes (C0 frame-0 skipped)"

    print(f"PASS — schedule {kinds}")
    print(f"PASS — cold_start obs is a single dict; reset/cold_start on main thread")
    print(f"PASS — async_step keyframe counts: "
          f"{[len(p['obs']) for _, p, _ in async_calls]} (expect [4,8,8,8,8])")
    print(f"PASS — state/executing_action identity rotation holds for {N} cycles")
    print(f"PASS — all {N} async_step requests ran off the main thread")
    return 0


if __name__ == "__main__":
    sys.exit(main())
