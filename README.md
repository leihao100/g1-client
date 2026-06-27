# LingBot-VA G1 Inference Client

A runtime inference client that drives a Unitree G1 humanoid's arms (14 joints) and
Dex1 grippers from action chunks streamed by a remote LingBot-VA policy server.

The client connects to the cloud policy over a WebSocket, dispatches arm/gripper
targets to the robot over CycloneDDS, captures camera keyframes from the G1's
on-board image server, and feeds them back so the server's autoregressive context
tracks reality.

## Prerequisites

- A **Unitree G1** in **`ai` motion mode**, standing on the ground. The operator
  sets this via the Unitree app — this client does not switch motion modes.
- The G1 **image server** running on the on-board PC2 (default `192.168.123.164`),
  with the head + both wrist cameras enabled.
- A reachable **LingBot-VA policy server** (WebSocket).
- **`unitree_sdk2py`** — the Unitree Python SDK. Clone
  https://github.com/unitreerobotics/unitree_sdk2_python and `pip3 install -e .`
  there. (Needs CycloneDDS; see that repo's README if the build can't find it.)
- **`teleimager`** — the G1 image-server client, from the `xr_teleoperate` repo.

## Install

```bash
pip install -r requirements.txt
```

plus the two prerequisites above (`unitree_sdk2py`, `teleimager`).

## Usage

Each model has its own folder; run that model's `main.py` directly from the repo
root. For the default LingBot-VA cloud pipeline:

```bash
python lingbot_va/main.py \
    --iface enp0s31f6 \
    --server-host <policy-server-ip> \
    --server-port 29056 \
    --prompt "pick up the pink object and place it on the blue cross mark"
```

`--iface` is the network interface connected to the robot. Run any script with
`--help` for all options (init durations, velocity limit, kp, guidance scales,
etc.). `--server-host`/`--server-port` point at that model's policy server.

The other models are launched the same way, each from the repo root:

```bash
# FastWAM (FastWAM/serve_fastwam_g1.py, default port 8000)
python fastwam/main.py \
    --iface enp0s31f6 \
    --server-host 1.2.3.4 \
    --server-port 8000 \
    --prompt "pick the red bottle"

# DiT4DiT (server_policy.py) — also needs the checkpoint's normalization stats
python dit/main.py \
    --iface enp0s31f6 \
    --server-host 1.2.3.4 \
    --server-port 10093 \
    --norm-stats /path/to/<run_dir>/dataset_statistics.json \
    --prompt "pick the red bottle"

# openpi (serve_policy.py, default port 8000)
python openpi/main.py \
    --iface enp0s31f6 \
    --server-host 1.2.3.4 \
    --server-port 8000 \
    --prompt "pick up the pink object and place it on the blue cross mark"

# openpi masked / weighted-blend variant (same server as openpi)
python openpi/main_mask.py \
    --iface enp0s31f6 \
    --server-host 1.2.3.4 \
    --server-port 8000 \
    --prompt "pick up the pink object and place it on the blue cross mark"
```

Validate cloud connectivity without touching the robot:

```bash
python lingbot_va/smoke_test.py --server-host <policy-server-ip> --server-port 29056
```

## Safety

- The robot **must** be in `ai` mode and standing before you start — the client
  takes arm authority from the locomotion service and assumes it stays balanced.
- After the init move the client holds at the ready pose and waits for **Enter**
  before contacting the policy server, so you can stage the scene. `--auto-start`
  skips this gate.
- **Ctrl+C is safe at any point** — init, standby, or inference. Cleanup always
  ramps arm authority (`arm_sdk`) back to the locomotion service.

## Repo layout

The repo is organized as a **shared package plus one folder per model**:

```
g1_client/                 Shared package — controllers + transport, imported by every model.
├── arm_controller.py      14 arm joints on rt/arm_sdk @ 50 Hz; body-lock, velocity clamp, kp switching.
├── gripper_controller.py  Dex1 grippers on rt/dex1/{l,r}/cmd @ 200 Hz with a per-tick rate cap.
├── camera_client.py       3-camera capture via teleimager, JPEG-encoded BGR at 256x320 (server does BGR->RGB after decode).
├── policy_client.py       WebSocket client (msgpack wire format) — used by lingbot_va + openpi.
└── msgpack_numpy.py       NumPy support for msgpack — must mirror the server's module.

lingbot_va/                LingBot-VA cloud model — the default / base pipeline.
├── main.py                Entry point — full async FDM-grounded inference loop.
├── smoke_test.py          Cloud round-trip test, no robot/cameras.
├── test_async_loop.py     Offline wire-schedule check using fakes.
├── test_async_safety.py   Offline safety/threading checks.
├── test_policy_server.py  Local mock policy server for manual testing.
└── replay.py              Open-loop replay of a recorded LeRobot episode (no server).

fastwam/                   FastWAM model.
├── main.py                Entry point.
└── fastwam_policy.py      FastWAM client adapter (msgpack/WebSocket).

dit/                       DiT4DiT model.
├── main.py                Entry point.
└── dit_policy.py          DiT client adapter (client-side normalization).

openpi/                    openpi serve_policy.py model.
├── main.py                Entry point — receding-horizon loop + one-chunk prefetch.
├── main_mask.py           Masked / weighted-blend variant (imports openpi/main.py).
└── openpi_policy.py       openpi client adapter (standalone helper).
```

Each entry script is run directly from the repo root (e.g. `python fastwam/main.py …`);
each inserts the repo root onto `sys.path` so the shared `g1_client` package
resolves no matter where you launch from. Running them as `python -m ...` will not work.

See [`CLAUDE.md`](CLAUDE.md) for the full architecture write-up — action tensor
layout, chunk cadence, the `arm_sdk` handover, shutdown ordering, and more.
