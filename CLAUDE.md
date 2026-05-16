# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A standalone runtime inference client that drives a Unitree G1 humanoid's arms and grippers from action chunks produced by a remote LingBot-VA policy server.

It depends on the Unitree Python SDK (`unitree_sdk2py`, CycloneDDS-backed) — installed separately as an external prerequisite (see Commands) — but is otherwise self-contained. `main.py` and `smoke_test.py` are entry-point scripts at the repo root; the controllers and clients live in the `g1_client/` package.

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

`main.py` / `smoke_test.py` import the `g1_client` package and are run directly from the repo root — Python finds the package because the script's own directory (the repo root) is on `sys.path`. Running them as `python -m ...` will NOT work.

Validate cloud connectivity without touching the robot:

```bash
python smoke_test.py --server-host <cloud-ip> --server-port 29056
```

There are no tests, linters, or build configs in this repo.

## Architecture

The inference pipeline is three asynchronous loops glued together in `main.py`:

```
Cloud LingBot-VA server  ──(WebSocket + msgpack_numpy)──►  PolicyClient
                                                              │
                                                              ▼ action [16,2,16]
                                                          execute_chunk()
                                                          ├─► ArmController  ──(DDS rt/arm_sdk @ 50 Hz)──► G1 arms+body lock
                                                          ├─► GripperController ──(DDS rt/dex1/{l,r}/cmd @ 200 Hz)──► Dex1 grippers
                                                          └─► CameraClient.get_obs() ◄──(ZMQ teleimager)── G1 PC2 image server
```

**Action tensor layout.** Each chunk is shape `(16, FRAME_CHUNK=2, SUBSTEPS_PER_FRAME=16)`. The named constants live at the top of `main.py`:
- `ARM_CHANNELS = slice(0, 14)` — 14 arm joints, ordered L{pitch,roll,yaw,elbow,wristR,wristP,wristY} then R{same}. This order is hard-coded in `g1_client.arm_controller.ARM_JOINTS` and the model is trained to match it — do not reorder.
- `LEFT_GRIPPER_CHANNEL = 14`, `RIGHT_GRIPPER_CHANNEL = 15` — gripper q in `[GRIPPER_MIN=0, GRIPPER_MAX=5.4]` rad.

**Chunk execution cadence.** 30 Hz sub-step dispatch → one chunk = 2 × 16 sub-steps ≈ 1.07 s. Keyframes are captured every `CAPTURE_EVERY=4` sub-steps and shipped back to the server in a `compute_kv_cache` call so the server's autoregressive context tracks reality. **8 keyframes per chunk** for chunks 1+, but **only 4 for chunk 0** — the first chunk's frame 0 is skipped (the model treats it as "observe the start position"), so only frame 1's four `f=3,7,11,15` captures fire.

**ArmController (`g1_client/arm_controller.py`).** Publishes a `LowCmd_` on `rt/arm_sdk` at 50 Hz. To take control of arm joints from the locomotion service, it sets `motor_cmd[29].q = 1.0` (the `kNotUsedJoint` slot doubles as the arm_sdk handover weight). The same `LowCmd_` also pins legs (`LEG_JOINTS` 0–11) and waist (`WAIST_JOINTS` 12–14) at their startup pose with `kp_body_lock=300`, so locomotion stays balanced while arms are driven. Per-tick velocity is clamped via `_clip_target` against `velocity_limit`; `move_to_pose` temporarily overrides it for the duration of the ramp and restores it on exit, but the inference loop never touches it — so the clamp is effectively a single fixed value during model-driven motion. `set_arm_target` also clips every command to per-joint position limits (`ARM_JOINT_MIN`/`ARM_JOINT_MAX`) before it ever reaches the wire, a second safety layer beyond the velocity clamp. On exit, `disable_arm_sdk()` ramps the weight back to 0 to hand control to the loco service. **kp switching pattern**: `kp_arm` defaults to **150** (stiffer hold, less gravity sag during the standby wait); `arm.set_arm_kp(80)` is called once Enter is pressed (just before connecting to the policy server) to drop into the softer kp the model was trained against. `set_arm_kp` only touches the 8 shoulder/elbow joints — the 6 wrist joints stay at `kp_wrist=40` for the whole run, so any future kp tuning that should also affect the wrists has to be added explicitly. Required precondition: robot is in **ai** motion mode and standing — `main.py` does not switch modes, operator sets it via the Unitree app.

**Publish-thread fault detector.** `ArmController` sets a `_faulted` flag if its publish or subscribe loop crashes on an unhandled exception, and exposes `faulted()`. `execute_chunk` polls `arm.faulted()` every sub-step (`main.py:109`) and raises — otherwise a dead publish thread would silently stop `arm_sdk` frames while `motor_cmd[29].q` is still latched at 1, and the loco service would not regain authority. The raise routes through the outer `try`/`finally` in `run()`, which then calls `disable_arm_sdk()` to ramp the handover weight back to 0.

**GripperController (`g1_client/gripper_controller.py`).** Publishes `MotorCmds_` on `rt/dex1/left/cmd` and `rt/dex1/right/cmd` at 200 Hz with a `DELTA_GRIPPER_CMD=0.18 rad` per-tick rate cap (the publish thread, not the user, enforces this). State is read back from `rt/dex1/{left,right}/state`.

**CameraClient (`g1_client/camera_client.py`).** Uses `teleimager.ImageClient` (host defaults to `192.168.123.164`, the G1 PC2). The head camera may be binocular; if so, only its **left half** is taken to match the training format. All three streams are converted BGR→RGB and resized to **256×320** (height × width) — these dimensions must match the model's expected input.

**PolicyClient (`g1_client/policy_client.py`).** Synchronous `websockets.sync.client` connection with `ping_interval=None` so long inference calls (multi-second GPU work) don't trip the keepalive. The wire format is msgpack with NumPy support (`g1_client/msgpack_numpy.py`, mirrors the server-side module byte-for-byte — keep them in sync). `_wait_for_server` retries forever on 5-second intervals — if the server is down, `_connect_policy` blocks indefinitely with arms still locked at `INIT_POSE_READY`. Ctrl+C is still safe there because the arm/gripper publish threads stay alive and the outer `finally` runs.

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
