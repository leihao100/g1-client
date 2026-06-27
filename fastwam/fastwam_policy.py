"""FastWAM client adapter for the G1 client (runs in the `unitree` env, no torch).

Drop-in sibling of g1_client/openpi_policy.py and g1_client/dit_policy.py. Talks
to FastWAM/serve_fastwam_g1.py (the GPU-side process in the `fastwam` env) over the same
msgpack/WebSocket protocol PolicyClient already speaks. FastWAM ships no server,
so FastWAM/serve_fastwam_g1.py wraps its in-process inference reference; this side stays a
thin, torch-free network client so the robot loop never imports torch.

    openpi / DiT4DiT                         FastWAM (this file)
    --------------------------------------   --------------------------------------------
    OpenPIPolicy / DitPolicy over WebSocket  FastWAMPolicy over WebSocket (same PolicyClient)
    server un-/normalizes                    server un-normalizes; returns RAW actions
    obs = decoded RGB / resized RGB          obs = 3 decoded RGB views + raw 16-dim state

The server un-normalizes server-side, so the returned actions are already in raw
joint radians — dispatch to the controllers is a direct pass-through, no
client-side scaling (identical to the openpi path).
"""

import logging

import numpy as np

from g1_client.policy_client import PolicyClient

log = logging.getLogger("g1_fastwam.policy")


class FastWAMPolicy:
    """Stateless wrapper over PolicyClient speaking the FastWAM obs/result protocol.

    infer(obs) returns {"actions": ndarray[H, 16]} in raw joint radians, so the
    receding-horizon loop in main_fastwam.py is identical in shape to
    main_openpi.py's / main-dit.py's (which also read result["actions"]).
    """

    def __init__(self, host: str, port: int, api_key=None):
        log.info(f"Connecting to FastWAM policy server ws://{host}:{port}")
        self._client = PolicyClient(host=host, port=port, api_key=api_key)
        log.info(f"Server metadata: {self._client.get_server_metadata()}")

    @property
    def last_timing(self):
        # Delegate so main_fastwam.py's per-chunk latency logging works unchanged.
        return self._client.last_timing

    def get_server_metadata(self) -> dict:
        return self._client.get_server_metadata()

    def infer(self, obs: dict) -> dict:
        """obs = {"image": [head, left_wrist, right_wrist] RGB uint8 HWC,
                  "state": raw float32 (16,), "prompt": str}.

        Returns {"actions": ndarray[H, 16]} in raw joint radians.
        """
        resp = self._client.infer(obs)
        if isinstance(resp, dict) and "actions" in resp:
            return {"actions": np.asarray(resp["actions"], dtype=np.float64)}
        raise RuntimeError(f"FastWAM server returned an unexpected payload: {resp!r}")

    def close(self):
        self._client.close()
