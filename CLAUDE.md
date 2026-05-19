# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A standalone runtime inference client that drives a Unitree G1 humanoid's arms and grippers from action chunks produced by a remote LingBot-VA policy server.

It depends on the Unitree Python SDK (`unitree_sdk2py`, CycloneDDS-backed) — installed separately as an external prerequisite (see Commands) — but is otherwise self-contained. `main.py`, `smoke_test.py`, and `test_async_loop.py` are entry-point scripts at the repo root; the controllers and clients live in the `g1_client/` package.

## Commands

Two prerequisites are installed separately (not on PyPI under these names):
- **`unitree_sdk2py`** — the Unitree Python SDK. Clone https://github.com/unitreerobotics/unitree_sdk2_python and `pip3 install -e .` there. If the build can't find cyclonedds, build it and `export CYCLONEDDS_HOME=~/cyclonedds/install` (see that repo's README).
- **`teleimager`** — the G1 image-server client, from the `xr_teleoperate` repo.

The remaining deps are pip-installable:

```bash
pip install -r requirements.txt
```

Run the full pipeline (from the repo root):

```bash
python main.py \
    --iface enp0s31f6 \
    --server-host <cloud-ip> \
    --server-port 29056 \
    --prompt "pick up the pink object and place it on the blue cross mark"
```

`main.py` / `smoke_test.py` / `test_async_loop.py` import the `g1_client` package and are run directly from the repo root — Python finds the package because the script's own directory (the repo root) is on `sys.path`. Running them as `python -m ...` will NOT work.

Two contract tests (no robot, no DDS) — exit 0 on pass:

```bash
# Wire-contract check against a running g1_async policy server (cloud round-trip).
python smoke_test.py --server-host <cloud-ip> --server-port 29056

# Offline check of the async loop's wire schedule using fakes (no server).
python test_async_loop.py
```

Both tests *pin the wire schedule* that `_run_inference_loop` is supposed to drive: `reset → cold_start → async_step` only, no `compute_kv_cache`, exact keyframe counts (0 / 4 / 8 / 8 / …), and verbatim identity on `state` / `executing_action`. They are deliberately strict because a desync between client and server (e.g. wrong keyframe count, wrong color order) corrupts the autoregressive context with no error — see "Wire-contract divergence" below for an active example.

Caveat for `test_async_loop.py`: although it fakes the arm/gripper/camera/policy and never touches DDS, it `from main import _run_inference_loop`, which transitively imports `unitree_sdk2py`. So even the "offline" test requires `unitree_sdk2py` (and therefore CycloneDDS) to be installed — it just doesn't *use* them. A missing SDK install will fail this test at import time, not on a real contract issue.

There are no linters or build configs in this repo.

## Architecture

The inference pipeline overlaps the policy-server request with on-robot chunk execution (Algorithm 2 in `main.py`, "async FDM-grounded loop"):

```
Cloud LingBot-VA server  ──(WebSocket + msgpack_numpy)──►  PolicyClient
        ▲                                                     │
        │ async_step (daemon thread)                          ▼ action [16,F,S]
        │  obs=K_{n-1}, state=a_{n-1},                    execute_chunk()
        │  executing_action=C_n                           ├─► ArmController     ──(DDS rt/arm_sdk @ 50 Hz)──► G1 arms+body lock
        │                                                ├─► GripperController ──(DDS rt/dex1/{l,r}/cmd @ 200 Hz)──► Dex1 grippers
        └────────────────────── keyframes (JPEG bytes) ◄─┴─► CameraClient.get_obs_images() ◄──(ZMQ teleimager)── G1 PC2 image server
```

**Action tensor layout.** Each chunk is shape `(16, F, S)` — 16 channels, `F` latent frames, `S` sub-steps/frame. `F` and `S` are read from the returned tensor (`action.shape[1]`/`[2]`), **not assumed**: the original g1 model is `F=2, S=16`; `g1_500step` is `F=4, S=16`. `FRAME_CHUNK=2`/`SUBSTEPS_PER_FRAME=16` remain at the top of `main.py` only as *nominal* values — `execute_chunk` logs a notice when the server's horizon differs and adapts, rather than asserting a fixed shape. The named constants live at the top of `main.py`:
- `ARM_CHANNELS = slice(0, 14)` — 14 arm joints, ordered L{pitch,roll,yaw,elbow,wristR,wristP,wristY} then R{same}. This order is hard-coded in `g1_client.arm_controller.ARM_JOINTS` and the model is trained to match it — do not reorder.
- `LEFT_GRIPPER_CHANNEL = 14`, `RIGHT_GRIPPER_CHANNEL = 15` — gripper q in `[GRIPPER_MIN=0, GRIPPER_MAX=5.4]` rad.

**Chunk execution cadence.** 30 Hz sub-step dispatch → one chunk = `F × S` sub-steps (≈1.07 s at F=2, ≈2.13 s at F=4). Keyframes are captured every `CAPTURE_EVERY=4` sub-steps (within each executed frame the captures fire at `f=3,7,11,15`) and are sent back in the **next** `async_step` request as the `obs` field, so the server can ground its FDM rollout on actually-observed reality. Counts:
- Chunk 0 runs with `is_first_chunk=True` → frame 0 skipped → `(F-1)×S/CAPTURE_EVERY` keyframes (**4 at F=2**).
- Chunks 1+ run all frames → `F×S/CAPTURE_EVERY` keyframes (**8 at F=2**).
- Cycle 0's `async_step` carries no `obs`/`state` at all — the server grounds `z_0` from its own init pose.

**Wire schedule.** `reset(prompt)` → `cold_start(obs=<single dict>)` → repeated `async_step` until `--max-chunks` is hit. `compute_kv_cache` is **not** used on the async path; grounding folds into `async_step`. `test_async_loop.py` and `smoke_test.py` pin this schedule exactly.

**Contract: keyframe counts and identity must match the server's expectation.** The first grounding cycle (cycle 1) must carry exactly 4 keyframes (not 8) because chunk 0 skipped frame 0; steady cycles carry 8. `state` and `executing_action` are passed as the **same ndarray objects** the server returned — no copy/reshape/renormalize on the client side. Too few/many keyframes, or a quietly reshaped `state`, desync the server's autoregressive context with no error — the same silent-failure class as the camera color-order contract below.

**ArmController (`g1_client/arm_controller.py`).** Publishes a `LowCmd_` on `rt/arm_sdk` at 50 Hz. To take control of arm joints from the locomotion service, it sets `motor_cmd[29].q = 1.0` (the `kNotUsedJoint` slot doubles as the arm_sdk handover weight). The same `LowCmd_` also pins legs (`LEG_JOINTS` 0–11) and waist (`WAIST_JOINTS` 12–14) at their startup pose with `kp_body_lock=300`, so locomotion stays balanced while arms are driven. Per-tick velocity is clamped via `_clip_target` against `velocity_limit`; `move_to_pose` temporarily overrides it for the duration of the ramp and restores it on exit, but the inference loop never touches it — so the clamp is effectively a single fixed value during model-driven motion. `set_arm_target` also clips every command to per-joint position limits (`ARM_JOINT_MIN`/`ARM_JOINT_MAX`, defined at `g1_client/arm_controller.py:100-107`) before it ever reaches the wire, a second safety layer beyond the velocity clamp. These limits are deliberately *tighter than the hardware limits* to keep a margin — do not widen them to match the spec sheet without a specific reason. On exit, `disable_arm_sdk()` ramps the weight back to 0 to hand control to the loco service. **kp switching pattern**: `kp_arm` defaults to **150** (stiffer hold, less gravity sag during the standby wait); `arm.set_arm_kp(80)` is called once Enter is pressed (just before connecting to the policy server) to drop into the softer kp the model was trained against. `set_arm_kp` only touches the 8 shoulder/elbow joints — the 6 wrist joints stay at `kp_wrist=40` for the whole run, so any future kp tuning that should also affect the wrists has to be added explicitly. Required precondition: robot is in **ai** motion mode and standing — `main.py` does not switch modes, operator sets it via the Unitree app.

**Publish-thread fault detector.** `ArmController` sets a `_faulted` flag if its publish or subscribe loop crashes on an unhandled exception, and exposes `faulted()`. `execute_chunk` polls `arm.faulted()` every sub-step (`main.py:136`) and raises — otherwise a dead publish thread would silently stop `arm_sdk` frames while `motor_cmd[29].q` is still latched at 1, and the loco service would not regain authority. The raise routes through the outer `try`/`finally` in `run()`, which then calls `disable_arm_sdk()` to ramp the handover weight back to 0.

**GripperController (`g1_client/gripper_controller.py`).** Publishes `MotorCmds_` on `rt/dex1/left/cmd` and `rt/dex1/right/cmd` at 200 Hz with a `DELTA_GRIPPER_CMD=0.18 rad` per-tick rate cap (the publish thread, not the user, enforces this). State is read back from `rt/dex1/{left,right}/state`.

**CameraClient (`g1_client/camera_client.py`).** Uses `teleimager.ImageClient` (host defaults to `192.168.123.164`, the G1 PC2). The head camera may be binocular; if so, only its **left half** is taken to match the training format. All three streams are resized to **256×320** (H×W) and **JPEG-encoded at q90 in cv2's native BGR order** — the wire format is JPEG `bytes`, not a NumPy array. This cuts each frame from ~240 KiB raw to ~15–25 KiB, ~10–15× smaller upload per keyframe.

**Color-order contract.** Because frames are encoded straight from BGR, the **server is responsible for the BGR→RGB conversion** after `cv2.imdecode` (which returns BGR). The client no longer does this. Keep client encode and server decode in lockstep — getting the order wrong silently feeds the model channel-swapped images.

`get_obs(prompt)` returns the cold-start payload (3 cams + `"task"` key); `get_obs_images()` returns the camera-only dict used for keyframes inside `execute_chunk`.

**PolicyClient (`g1_client/policy_client.py`).** Synchronous `websockets.sync.client` connection with `ping_interval=None` so long inference calls (multi-second GPU work) don't trip the keepalive. The wire format is msgpack with NumPy support (`g1_client/msgpack_numpy.py`, mirrors the server-side module byte-for-byte — keep them in sync). The client itself is **protocol-agnostic** — `infer(payload)` packs whatever dict you give it; `reset(prompt)` is the only helper. The async-loop protocol (`reset` → `cold_start` → `async_step` × N) is implemented in `main.py`'s `_run_inference_loop` and `_async_step_worker`. `last_timing` exposes a pack/send/wait_recv/unpack breakdown of the most recent call, used by `_fmt_infer_timing` for per-cycle logging. `_wait_for_server` retries forever on 5-second intervals — this is intentional, since the operator may start the policy server *after* the client (the arms hold at `INIT_POSE_READY` while it waits). If the server is down, `_connect_policy` blocks indefinitely; Ctrl+C is still safe there because the arm/gripper publish threads stay alive and the outer `finally` runs. Do not "fix" this to a bounded retry without also revisiting the operator workflow.

**Branch A / Branch B overlap.** Inside `_run_inference_loop`, each steady cycle dispatches Branch B (`_async_step_worker` on a daemon thread) **before** running Branch A (`execute_chunk` on the main thread). The daemon thread does the blocking `policy.infer(...)`; the main thread streams the current chunk to DDS at 30 Hz. Only after `execute_chunk` returns does the main thread `out_q.get()` on Branch B's result and join the thread. Errors raised inside the daemon are surfaced via the queue (`("err", e)`) so a network failure aborts the loop instead of getting silently swallowed. The two-chunk buffered cold start (see "Wire-contract divergence" below) exists specifically so cycle 0 already has a chunk to execute in Branch A while Branch B is grounding the previous one.

**Startup sequence.** `arm.start()` → `move_to_pose(INIT_POSE_READY)` (5 s ramp at the operating vlim, no separate "init vlim") → grippers do a `close→open` sequence (each half of `--init-duration`) so the operator can see the dex1 has joined the SDK → `--settle-duration` pause → `input("")` standby (skipped with `--auto-start`) → `arm.set_arm_kp(args.inference_kp_arm)` (default 80) → connect to policy server. The arm and gripper publish threads keep streaming the current target during standby so the robot stays locked at `INIT_POSE_READY`.

**Shutdown safety.** A single `try`/`finally` block in `run()` wraps everything from gripper init onwards. On any exit path (normal completion, exception, or Ctrl+C — including during the `input("")` standby), the finally calls `_cleanup`, which runs each release step in its own `try/except BaseException` so any single failure (including a second Ctrl+C landing mid-cleanup) cannot skip the others. The steps are:
1. `arm.stop()` — joins the publish thread so step 2 has exclusive access to `self.cmd`.
2. `arm.disable_arm_sdk()` — ramps `motor_cmd[29].q` 1→0 over 1 s, and has its own internal `try/finally` that **guarantees a terminal `q=0` write** even if the ramp is interrupted mid-sleep. The loco service always sees the handover release.
3. `grip.stop()` — parks the gripper target at the current measured position before signaling stop, then joins the publish thread, so the last frames on the wire hold-in-place rather than continuing whatever the model last commanded.
4. `cam.close()`, `policy.close()` — each may be `None` if the failure happened before that resource was constructed.

Two subtleties that matter for safety:
- **`except BaseException`, not `except Exception`.** `KeyboardInterrupt` is `BaseException`, so a plain `except Exception` would let a second Ctrl+C escape and skip the rest of cleanup. The pattern in `_cleanup` is deliberate; do not "tighten" it.
- **Order is preferential, not load-bearing.** `arm.stop()` running before `disable_arm_sdk` means the 1→0 ramp isn't fighting fresh 50 Hz arm-target writes, which is cleaner. But the `_cmd_lock` (held across mutation+CRC+Write in `_publish_loop`) makes a concurrent write CRC-safe regardless of order, and `disable_arm_sdk`'s own terminal-write `finally` survives even if `arm.stop()` itself is interrupted.

**SDK channel choice.** G1 uses `unitree_sdk2py.idl.unitree_hg` (NOT `unitree_go`, which is for Go2/B2/H1) — documented in the Unitree SDK's `example/g1/` notes, and easy to get wrong. The Dex1 grippers in `g1_client/gripper_controller.py` are an exception — they use the older `unitree_go` IDL because dex1 is a separate accessory; do not "fix" this to match the arms.

**Wire-contract divergence (active — verify before edits).** `main.py` and the contract tests currently disagree on the `cold_start` response shape, and a future edit will need to settle this:

- `main.py:303-305` expects **two** chunks back: `resp = policy.infer({"cold_start": True, "obs": init}); C0, C1 = resp["action"], resp["action1"]`. C0 runs as the non-overlapped first chunk; C1 is the pipeline buffer kept one chunk ahead so the steady loop stays overlapped. The first `async_step` then grounds C0's executed reality.
- `smoke_test.py` and `test_async_loop.py` both assert that `cold_start` returns **only `"action"`** (no `"action1"`) and that cycle 0's `async_step` carries no `obs`/`state` (server grounds `z_0` itself) and returns C1.

These cannot both be right against the same server. The tests embody the "single-chunk cold_start" protocol; `main.py` implements the "two-chunk buffered cold_start" protocol. Whichever you keep, both ends (this client *and* the server) must agree — this is the same silent-desync class as the keyframe-count and color-order contracts above.

**Resolution direction.** Verified against the deployed server at `/home/nuwm-1/Workspace/dev/lingbot-va/wan_va/wan_va_server.py` (cross-repo check, May 2026): the **server implements the two-chunk protocol that `main.py` matches**. Specifically:

- `_cold_start` (`wan_va_server.py:824-862`) returns `dict(action=action0, action1=action1)` — both chunks, with `C1` FDM-grounded on `z_0` chunk-aligned.
- `_fdm_step` (`wan_va_server.py:864-891`) calls `_compute_kv_cache(obs)` which unconditionally reads `obs['obs']` and `obs['state']`, then `preprocess_action(obs['executing_action'])`. **All three fields are required on every async_step** — there is no code path for a "cycle 0 with no obs/state."
- The server-side test `lingbot-va/tests/test_fdm_cache_invariant.py:88-92` explicitly asserts `'action' in cs and 'action1' in cs` and replays steady-state async_step starting from `n=1` with full obs/state/executing_action.

So `main.py` is **correct against the server**. The two client-side tests (`smoke_test.py`, `test_async_loop.py`) were changed in commit `f472ca5` to pin a single-chunk protocol that the server never implemented — running either of them against the real server fails (`smoke_test.py` aborts on `"action1" in r`; the cycle-0 `async_step` with no `obs`/`state` would crash the server at `KeyError: 'obs'` in `_compute_kv_cache`). The right next edit is to **revert the two client tests to the two-chunk schedule** `main.py` and the server already share. Do not change `main.py`'s `_run_inference_loop` to match the tests — that would break against the deployed server.

**Status: known-but-deferred.** The production path (`main.py` ↔ g1_async server) is verified aligned and works as-is. The two broken tests are not in CI and only run when invoked manually, so leaving them mismatched does **not** break anything that ships. Two related minor items in `main.py` are also known and intentionally left alone for the same reason:
- `_run_inference_loop`'s `for n in range(1, args.max_chunks + 1)` fires one extra `async_step` whose returned chunk is never executed — wastes one GPU inference per run, harmless.
- `--video-guidance` / `--action-guidance` CLI args are defined but never sent in any payload — guidance is fully controlled by the server config. Dead args, harmless.

Do not "clean up" any of the four items above as part of an unrelated edit. They are surface area for a focused future PR, not drive-by cleanups — touching them in passing risks reintroducing the silent-desync class.

Run `python test_async_loop.py` after any change to `_run_inference_loop` to catch schedule-level regressions, but only *after* fixing the test to match the real wire — until then, the test is enforcing a fiction. Run `smoke_test.py` against the deployed server only after the same fix.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
