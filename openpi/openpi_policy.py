"""openpi data send/receive layer for the G1 client.

Drop-in replacement for g1_client/policy_client.py (the LingBot-VA client).
This is the ONLY thing that changes: the controllers (arm/gripper/camera) are
untouched. The robot-facing control code is identical; only how we talk to the
policy server is different.

LingBot protocol (old)            openpi protocol (new, this file)
--------------------------------  --------------------------------------------
stateful: reset(prompt) →         stateless: every infer() ships one full obs
cold_start → async_step           and gets back one action chunk. No reset,
returns {"action","action1"}      cold_start, or async_step.
action tensor [16, F, S]          returns {"actions": ndarray[H, A]}
server does FDM grounding /        server just runs policy.infer(obs) and
KV-cache imagination               un-normalizes via the checkpoint's norm_stats

Because openpi un-normalizes server-side (this is why you run
compute_norm_stats.py before serving), the returned actions are already in the
dataset's RAW units — i.e. absolute joint radians + gripper radians if that is
how the LeRobot dataset stored them. So dispatch to the controllers is a direct
pass-through, no client-side scaling.

Place this file at g1_client/openpi_policy.py.
"""

import logging

import numpy as np

# Same client lib the UR3 script uses (openpi_client). Install with openpi.
from openpi_client import websocket_client_policy

log = logging.getLogger("g1_openpi.policy")


class OpenPIPolicy:
    """Thin, stateless wrapper over openpi's WebsocketClientPolicy.

    The constructor itself performs the connect + metadata handshake (the
    library retries internally until the server is reachable), exactly the
    same wire protocol the UR3 tcp_infer script speaks to serve_policy.py.
    """

    def __init__(self, host: str, port: int, api_key: str | None = None):
        log.info(f"Connecting to openpi policy server ws://{host}:{port}")
        # WebsocketClientPolicy(host, port) connects and reads the packed
        # server metadata frame on construction.
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port, api_key=api_key
        )
        log.info(f"Server metadata: {self._client.get_server_metadata()}")

    def get_server_metadata(self) -> dict:
        return self._client.get_server_metadata()

    def infer(self, obs: dict) -> np.ndarray:
        """Send one observation, return the action chunk as ndarray[H, A].

        obs keys MUST match the server checkpoint's RepackTransform / DataConfig
        (see build_obs() in main_openpi.py for the exact contract).
        """
        result = self._client.infer(obs)
        return np.asarray(result["actions"], dtype=np.float64)

    def close(self):
        # Not all openpi versions expose .close(); close the underlying socket
        # directly if present so we don't leak the connection on shutdown.
        ws = getattr(self._client, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
