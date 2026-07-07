"""Main inference loop for the G1 driving a GR00T EEF-space policy
(Isaac-GR00T N1.7, embodiment NEW_EMBODIMENT + g1_eef_config.py, trained on
the *-eef LeRobot datasets).

WHAT CHANGED vs openpi/main_eef.py
----------------------------------
Nothing in the robot control path, the EEF<->joint kinematics, or the chunk
scheduling (prefetch overlap, time-alignment, cross-fade). Those are proven and
reused verbatim from openpi/main_eef.py. The ONLY change is the transport:
openpi's WebSocket PolicyClient is swapped for groot_client.GrootClient, which
speaks the GR00T ZMQ/msgpack_numpy `get_action` protocol and re-assembles the
per-modality action dict into the same [H, 16] EEF array the loop expects.

Because GrootClient exposes the identical .infer(obs)->{"actions": [H,16]},
.last_timing and .close() interface, the whole receding-horizon loop, the
observation/action contract, and every CLI flag carry over unchanged.

Precondition: robot already in 'ai' motion mode (set via the Unitree app).

Usage (run from the repo root):
  python groot/main_eef.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 5555 \\
      --prompt "stack the cube"

Start the server first on the GPU box:
  python gr00t/eval/run_gr00t_server.py \\
      --model-path <checkpoint> --embodiment-tag NEW_EMBODIMENT \\
      --modality-config-path ~/unitree/data/g1_eef_config.py \\
      --host 0.0.0.0 --port 5555
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root -> g1_client package
OPENPI = os.path.join(ROOT, "openpi")  # proven loop + eef_kinematics live here
for _p in (ROOT, OPENPI, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from groot_client import GrootClient  # noqa: E402

# Load openpi/main_eef.py as a module and reuse its loop wholesale. It runs its
# own sys.path bootstrap and imports eef_kinematics + g1_client on exec, which
# the path setup above satisfies.
_spec = importlib.util.spec_from_file_location(
    "openpi_main_eef", os.path.join(OPENPI, "main_eef.py")
)
base = importlib.util.module_from_spec(_spec)
sys.modules["openpi_main_eef"] = base
_spec.loader.exec_module(base)

# The single substitution: run() constructs `PolicyClient(host=..., port=...)`.
# GrootClient has the same constructor signature, so swapping the module-level
# name redirects the loop to the GR00T server with zero other edits.
base.PolicyClient = GrootClient


if __name__ == "__main__":
    # Reuses openpi/main_eef.py's argparse + run(). Note: --server-port defaults
    # to 8000 there; pass --server-port 5555 for the GR00T server.
    base.main()
