"""Offline safety tests for the two latched-robot failure modes.

No server/robot. Fakes injected into _run_inference_loop.

A. #17 — a non-responding server must NOT hang the client forever. With a
   bounded server_timeout, _run_inference_loop must raise TimeoutError
   promptly (so run()'s finally releases arm_sdk) instead of blocking in
   out_q.get() / ws.recv() indefinitely with the robot latched.

B. #2  — a camera-capture failure mid-chunk must NOT silently shorten the
   keyframe list (which would desync the server's grounding with no error).
   execute_chunk must fail loud: _run_inference_loop raises rather than
   completing and shipping a short keyframe list.

Run:  conda activate unitree_deploy && python test_async_safety.py
Exit 0 = pass, 1 = fail.
"""

import sys
import threading
import types

import numpy as np

from main import _run_inference_loop


def _chunk():
    return np.zeros((16, 2, 16), dtype=np.float64)


class FakeArm:
    def faulted(self): return False
    def set_arm_target(self, q): pass


class FakeGrip:
    def set_targets(self, l, r): pass


class FakeCam:
    def get_obs(self, prompt): return {"cam": b"j", "task": prompt}
    def get_obs_images(self): return {"cam": b"j"}


def _args(max_chunks, server_timeout):
    return types.SimpleNamespace(
        prompt="t", max_chunks=max_chunks, substep_hz=1_000_000.0,
        quiet_substeps=True, server_timeout=server_timeout)


# ---- A. non-responding server -> bounded TimeoutError, not an infinite hang ----

class HangPolicy:
    """reset/cold_start reply; async_step blocks forever (server never answers)."""
    def __init__(self):
        self.last_timing = None
        self._never = threading.Event()

    def infer(self, payload):
        if payload.get("reset"):
            return {}
        if payload.get("cold_start"):
            return {"action": _chunk()}
        if payload.get("async_step"):
            self._never.wait()          # simulate server that never responds
            return {"action": _chunk()}
        raise AssertionError(f"unexpected payload {sorted(payload)}")

    def reset(self, prompt): self.infer({"reset": True, "prompt": prompt})


def test_server_timeout():
    pol = HangPolicy()
    box = {}

    def run():
        try:
            _run_inference_loop(FakeArm(), FakeGrip(), FakeCam(), pol,
                                _args(max_chunks=5, server_timeout=0.5))
            box["exc"] = None
        except BaseException as e:  # noqa: BLE001 - want to capture TimeoutError
            box["exc"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=8.0)  # server_timeout=0.5 → must raise well within this

    assert not t.is_alive(), \
        "FAIL #17: _run_inference_loop is STILL HUNG after 8s with a " \
        "non-responding server (out_q.get/recv has no timeout)"
    assert isinstance(box.get("exc"), TimeoutError), \
        f"FAIL #17: expected TimeoutError, got {box.get('exc')!r}"
    print("PASS #17 — non-responding server → TimeoutError "
          f"(~0.5s), loop did not hang ({type(box['exc']).__name__})")


# ---- B. camera failure mid-chunk -> fail loud, no short keyframe list ----

class OkPolicy:
    def __init__(self): self.last_timing = None
    def infer(self, payload):
        if payload.get("reset"):
            return {}
        return {"action": _chunk()}        # cold_start + every async_step
    def reset(self, prompt): self.infer({"reset": True, "prompt": prompt})


class FlakyCam:
    """Cold-start obs OK; the 2nd in-chunk keyframe capture raises (a
    transient teleimager hiccup)."""
    def __init__(self): self._n = 0
    def get_obs(self, prompt): return {"cam": b"j", "task": prompt}
    def get_obs_images(self):
        self._n += 1
        if self._n == 2:
            raise RuntimeError("teleimager returned None (simulated glitch)")
        return {"cam": b"j"}


def test_keyframe_failclosed():
    raised = None
    try:
        _run_inference_loop(FakeArm(), FakeGrip(), FlakyCam(), OkPolicy(),
                            _args(max_chunks=3, server_timeout=30.0))
    except BaseException as e:  # noqa: BLE001
        raised = e

    assert raised is not None, (
        "FAIL #2: a mid-chunk camera failure was SWALLOWED — loop completed "
        "and would have shipped a short keyframe list (silent server desync)")
    assert isinstance(raised, RuntimeError), \
        f"FAIL #2: expected RuntimeError on capture failure, got {raised!r}"
    print(f"PASS #2 — camera capture failure → loud {type(raised).__name__} "
          "(no short keyframe list shipped)")


# ---- C. --repeat: Enter between tasks must reset+cold_start again ----

class CountingOkPolicy(OkPolicy):
    def __init__(self):
        super().__init__()
        self.resets = 0
        self.cold_starts = 0

    def infer(self, payload):
        if payload.get("reset"):
            self.resets += 1
            return {}
        if payload.get("cold_start"):
            self.cold_starts += 1
        return super().infer(payload)


class FakeArmRe(FakeArm):
    """Adds the move/kp shims the between-tasks block calls."""
    def __init__(self): self.kp_arm = 150.0
    def set_arm_kp(self, k): self.kp_arm = float(k)
    def move_to_pose(self, q, duration=None, velocity_limit=None): pass


class FakeGripRe(FakeGrip):
    def move_to_targets(self, l, r, duration=None): pass


def test_repeat_enter_init_then_main():
    """Between tasks: Enter #1 → INIT mode (arm to ready, gripper cycle).
    Enter #2 → MAIN mode (re-run with reset + cold_start). Mirrors the
    startup flow. EOF on the FIRST prompt exits cleanly."""
    import builtins
    call_log = []
    # one task → ENTER#1 (init) → ENTER#2 (main) → second task → EOF exits
    inputs = iter(["", "", EOFError("eof")])

    def fake_input(prompt=""):
        v = next(inputs)
        call_log.append("input")
        if isinstance(v, BaseException):
            raise v
        return v

    orig_input = builtins.input
    builtins.input = fake_input
    try:
        pol = CountingOkPolicy()
        args = _args(max_chunks=2, server_timeout=30.0)
        # Fields run() would have set / argparse provides for --repeat path
        args.repeat = True
        args.auto_start = False
        args.init_duration = 0.01
        args.gripper_init_duration = 0.01
        args.settle_duration = 0.0
        args.init_gripper_left = 5.0
        args.init_gripper_right = 5.0
        args.inference_kp_arm = 80.0
        args.velocity_limit = 20.0
        args._init_kp_arm = 150.0

        _run_inference_loop(FakeArmRe(), FakeGripRe(), FakeCam(), pol, args)

        assert pol.resets == 2, \
            f"FAIL repeat: expected 2 resets (initial + ONE re-run via " \
            f"Enter→init→Enter→main), got {pol.resets}"
        assert pol.cold_starts == 2, \
            f"FAIL repeat: expected 2 cold_starts, got {pol.cold_starts}"
        # 3 input() calls total: init prompt + main standby (between tasks)
        # + the init prompt of the next iteration that EOFs to exit.
        assert len(call_log) == 3, \
            f"FAIL repeat: expected 3 input() prompts (Enter#1+Enter#2 " \
            f"between tasks, then EOF on next iter's Enter#1), got " \
            f"{len(call_log)} — likely still using the single-Enter flow"
    finally:
        builtins.input = orig_input
    print(f"PASS repeat — Enter→INIT→Enter→MAIN flow "
          f"({pol.resets} resets, {len(call_log)} input prompts); "
          f"EOF on first prompt exits cleanly")


# ---- D. reset key mid-task -> abort -> init -> Enter -> re-run ----

_test_captured_reset_event = None


def _stub_make_reset_watcher(args):
    """Test stub for main._make_reset_watcher: captures the event so the
    test can set it via FakePolicy; does NOT start a real stdin thread or
    touch the terminal. Returns the 4-tuple the real factory does."""
    global _test_captured_reset_event
    _test_captured_reset_event = threading.Event()
    return _test_captured_reset_event, threading.Event(), None, None


class ResetTriggerPolicy(CountingOkPolicy):
    """Counts async_step calls; on the Nth async_step of the FIRST task,
    sets the captured reset event (simulates operator pressing 'r'+Enter
    during that cycle). Returns to normal for the second task."""

    def __init__(self, trigger_at=2):
        super().__init__()
        self._async = 0
        self._fired = False
        self._trigger_at = trigger_at

    def infer(self, payload):
        if payload.get("async_step"):
            self._async += 1
            if (not self._fired
                    and self._async == self._trigger_at
                    and _test_captured_reset_event is not None):
                _test_captured_reset_event.set()
                self._fired = True
        return super().infer(payload)


def test_reset_during_task_aborts_and_reruns():
    """Pressing 'r'+Enter mid-task must: (1) abort the in-flight task at the
    next chunk boundary, (2) skip the Enter#1 'go to init' prompt (the reset
    already implies it), (3) run _initialize_pose, (4) wait for Enter#2
    (standby), (5) re-run — which re-issues policy.reset, clearing the
    server's KV cache."""
    import main
    import builtins

    orig_factory = main._make_reset_watcher
    main._make_reset_watcher = _stub_make_reset_watcher

    # Only ONE input call expected on the reset path: Enter#2 after init.
    # Enter#1 is skipped because the reset was explicit. Then task 2 runs
    # to completion and the NEXT iteration's Enter#1 prompt gets EOF -> exit.
    inputs = iter(["", EOFError("eof")])

    def fake_input(prompt=""):
        v = next(inputs)
        if isinstance(v, BaseException):
            raise v
        return v

    orig_input = builtins.input
    builtins.input = fake_input

    try:
        pol = ResetTriggerPolicy(trigger_at=2)  # fire on cycle 2's async_step
        args = _args(max_chunks=10, server_timeout=30.0)
        args.repeat = True
        args.auto_start = False
        args.init_duration = 0.01
        args.gripper_init_duration = 0.01
        args.settle_duration = 0.0
        args.init_gripper_left = 5.0
        args.init_gripper_right = 5.0
        args.inference_kp_arm = 80.0
        args.velocity_limit = 20.0
        args._init_kp_arm = 150.0

        main._run_inference_loop(FakeArmRe(), FakeGripRe(), FakeCam(), pol, args)

        # 2 tasks total: task 1 aborted by reset, task 2 ran to completion.
        assert pol.resets == 2, (
            f"FAIL reset: expected 2 resets (initial aborted + re-run after "
            f"reset), got {pol.resets}")
        assert pol.cold_starts == 2, (
            f"FAIL reset: expected 2 cold_starts (server KV cache cleared "
            f"each task via reset), got {pol.cold_starts}")
    finally:
        main._make_reset_watcher = orig_factory
        builtins.input = orig_input

    print(f"PASS reset — 'r'+Enter mid-task aborts → INIT → Enter#2 → re-run "
          f"(resets={pol.resets}, kv cache cleared each task)")


def test_reset_works_without_repeat():
    """The reset key is always enabled — not gated on --repeat. Pressing 'r'
    is itself an explicit 're-run' request, so it must trigger the
    init+standby+re-run cycle once even when --repeat is OFF. After that
    single re-run completes naturally, the loop exits."""
    import main
    import builtins

    orig_factory = main._make_reset_watcher
    main._make_reset_watcher = _stub_make_reset_watcher

    inputs = iter([""])  # Enter#2 after init; should be the ONLY input() call
    n_input_calls = [0]

    def fake_input(prompt=""):
        n_input_calls[0] += 1
        return next(inputs)

    orig_input = builtins.input
    builtins.input = fake_input

    try:
        pol = ResetTriggerPolicy(trigger_at=2)
        args = _args(max_chunks=10, server_timeout=30.0)
        args.repeat = False              # KEY: --repeat OFF
        args.auto_start = False
        args.init_duration = 0.01
        args.gripper_init_duration = 0.01
        args.settle_duration = 0.0
        args.init_gripper_left = 5.0
        args.init_gripper_right = 5.0
        args.inference_kp_arm = 80.0
        args.velocity_limit = 20.0
        args._init_kp_arm = 150.0

        main._run_inference_loop(FakeArmRe(), FakeGripRe(), FakeCam(), pol, args)

        assert pol.resets == 2, (
            f"FAIL: reset without --repeat expected 2 resets (original task "
            f"aborted + one re-run), got {pol.resets} — reset path is being "
            f"swallowed by the early --repeat-off return")
        assert pol.cold_starts == 2, \
            f"FAIL: expected 2 cold_starts, got {pol.cold_starts}"
        assert n_input_calls[0] == 1, (
            f"FAIL: expected exactly 1 input prompt (Enter#2 standby after "
            f"reset's init), got {n_input_calls[0]}")
    finally:
        main._make_reset_watcher = orig_factory
        builtins.input = orig_input

    print(f"PASS reset-without-repeat — 'r' works without --repeat; one "
          f"re-run then exit (resets={pol.resets}, inputs={n_input_calls[0]})")


# ---- E. --sync mode: same wire contract, requests on main thread ----

MAIN_IDENT_E = threading.get_ident()


class ThreadRecordingPolicy(OkPolicy):
    """Records the thread each infer call ran on."""

    def __init__(self):
        super().__init__()
        self.idents = []  # list of (kind, thread_ident)

    def infer(self, payload):
        kind = ("reset" if payload.get("reset")
                else "cold_start" if payload.get("cold_start")
                else "async_step" if payload.get("async_step")
                else "UNKNOWN")
        self.idents.append((kind, threading.get_ident()))
        return super().infer(payload)


def test_sync_mode_runs_requests_on_main_thread():
    """With --sync, every async_step request runs on the MAIN thread (no
    daemon overlap). Wire schedule and keyframe counts are identical to
    async mode — only the threading model differs."""
    import main as main_mod

    pol = ThreadRecordingPolicy()
    N = 3
    args = _args(max_chunks=N, server_timeout=30.0)
    args.sync = True
    args.repeat = False
    args.auto_start = False

    main_mod._run_inference_loop(FakeArm(), FakeGrip(), FakeCam(), pol, args)

    async_idents = [ident for kind, ident in pol.idents if kind == "async_step"]
    assert len(async_idents) == N, \
        f"FAIL sync: expected {N} async_step calls, got {len(async_idents)}"
    for ident in async_idents:
        assert ident == MAIN_IDENT_E, (
            f"FAIL sync: async_step ran on non-main thread {ident} — sync "
            f"mode must NOT spawn a daemon worker")
    print(f"PASS sync — all {N} async_step requests on MAIN thread (no daemon)")


def main():
    test_server_timeout()
    test_keyframe_failclosed()
    test_repeat_enter_init_then_main()
    test_reset_during_task_aborts_and_reruns()
    test_reset_works_without_repeat()
    test_sync_mode_runs_requests_on_main_thread()
    print("ALL SAFETY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
