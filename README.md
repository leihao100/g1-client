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

Run the full pipeline from the repo root:

```bash
python main.py \
    --iface enp0s31f6 \
    --server-host <policy-server-ip> \
    --server-port 29056 \
    --prompt "pick up the pink object and place it on the blue cross mark"
```

`--iface` is the network interface connected to the robot. Run
`python main.py --help` for all options (init durations, velocity limit, kp,
guidance scales, etc.).

Validate cloud connectivity without touching the robot:

```bash
python smoke_test.py --server-host <policy-server-ip> --server-port 29056
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

```
main.py                    Entry point — runs the full inference pipeline.
smoke_test.py              Entry point — cloud round-trip test, no robot/cameras.
g1_client/                 The client package, imported by the entry scripts above.
├── arm_controller.py      14 arm joints on rt/arm_sdk @ 50 Hz; body-lock, velocity clamp, kp switching.
├── gripper_controller.py  Dex1 grippers on rt/dex1/{l,r}/cmd @ 200 Hz with a per-tick rate cap.
├── camera_client.py       3-camera capture via teleimager, JPEG-encoded BGR at 256x320 (server does BGR->RGB after decode).
├── policy_client.py       WebSocket client for the LingBot-VA server (msgpack wire format).
└── msgpack_numpy.py       NumPy support for msgpack — must mirror the server's module.
```

`main.py` / `smoke_test.py` are run directly from the repo root (`python main.py`);
they import the `g1_client` package. Running them as `python -m ...` will not work.

See [`CLAUDE.md`](CLAUDE.md) for the full architecture write-up — action tensor
layout, chunk cadence, the `arm_sdk` handover, shutdown ordering, and more.
