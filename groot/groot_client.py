"""GR00T (Isaac-GR00T N1.7) data send/receive layer for the G1 client.

Analog of openpi/openpi_policy.py, but for a GR00T inference server
(gr00t/eval/run_gr00t_server.py). It exposes the SAME interface the proven
openpi/main_eef.py loop already drives:

    .infer(obs) -> {"actions": ndarray[H, 16]}
    .last_timing  (dict: pack/send/wait_recv/unpack + byte counts)
    .close()

so the robot-facing control code (arm/gripper/camera, IK, prefetch, cross-fade)
is reused byte-for-byte — only how we talk to the policy server changes.

openpi protocol (old)                 GR00T protocol (new, this file)
------------------------------------  --------------------------------------------
WebSocket + msgpack                    ZeroMQ REQ/REP + msgpack_numpy
infer(flat_obs) -> {"actions": [H,A]}  call endpoint "get_action" with a NESTED,
flat obs keys, one action array        BATCHED obs; server returns (action_dict, info)
                                       keyed by action modality (left_eef/right_eef/
                                       gripper). We concat them back to [H, 16].

WIRE CONTRACT (must match the g1_eef_config.py registered on the server)
------------------------------------------------------------------------
Observation sent to `get_action` (B=1, T=1):
  video    {front, left_wrist, right_wrist}   uint8 (1, 1, H, W, 3)  RGB
  state    {left_eef, right_eef, gripper}      float32 (1, 1, D)
  language {annotation.human.task_description} [[prompt]]  (list[list[str]])

Action returned:  dict {left_eef:(1,H,7), right_eef:(1,H,7), gripper:(1,H,2)}
  concatenated in the server's action modality order -> ndarray[H, 16]:
    [:, 0:3] L eef pos  [:, 3:7] L eef quat  [:, 7:10] R eef pos
    [:, 10:14] R eef quat  [:, 14] L grip     [:, 15] R grip
  i.e. identical to the pi05_g1_eef layout main_eef.py already dispatches.

The server un-normalizes into the dataset's RAW units (absolute EEF poses +
gripper radians), so the returned actions are a direct pass-through to the IK
step in main_eef.py — no client-side scaling.

Deps (pip): pyzmq, msgpack, msgpack-numpy  (must match the server's versions).
Place this file at g1-client/groot/groot_client.py.
"""

import logging
import time

import numpy as np

# The SAME libraries the GR00T server (gr00t/policy/server_client.py) speaks.
# NOT g1_client/msgpack_numpy.py (that mirrors the LingBot/openpi websocket
# format) — the GR00T server uses the PyPI msgpack + msgpack_numpy pair.
import msgpack
import msgpack_numpy as mnp
import zmq

log = logging.getLogger("g1_groot.client")

# Camera stream (LeRobot key) -> GR00T video modality key (from meta/modality.json).
VIDEO_KEY_MAP = {
    "observation.images.cam_left_high": "front",
    "observation.images.cam_left_wrist": "left_wrist",
    "observation.images.cam_right_wrist": "right_wrist",
}

# State slices of the 16-dim observation.state, matching g1_eef_config.py /
# meta/modality.json. If you point this client at the JOINT checkpoint instead,
# the slices are the same (7|7|2) but the state/action keys are left_arm /
# right_arm / gripper — set STATE_SLICES/FALLBACK_ACTION_KEYS accordingly.
STATE_SLICES = {
    "left_eef": slice(0, 7),
    "right_eef": slice(7, 14),
    "gripper": slice(14, 16),
}
# Used only if the server's get_modality_config handshake fails to report order.
FALLBACK_ACTION_KEYS = ["left_eef", "right_eef", "gripper"]
FALLBACK_LANGUAGE_KEY = "annotation.human.task_description"


def _to_rgb_uint8(img) -> np.ndarray:
    """Return an HxWx3 uint8 RGB array. Decodes JPEG bytes if --send-jpeg fed us
    the camera client's compressed frames (BGR q90) instead of decoded RGB."""
    if isinstance(img, (bytes, bytearray)) or (
        isinstance(img, np.ndarray) and img.ndim == 1
    ):
        import cv2  # local import; only needed on the --send-jpeg path

        arr = np.frombuffer(img, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2.imdecode failed on a camera frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise RuntimeError(f"expected HxWx3 image, got shape {a.shape}")
    return a.astype(np.uint8, copy=False)


class GrootClient:
    """Thin, stateless ZMQ client for a GR00T PolicyServer.

    The constructor connects, pings, and reads the server's modality config so
    the action-key concatenation order and the language key stay in lockstep
    with the checkpoint (no silent desync). Signature matches PolicyClient so it
    is a drop-in for openpi/main_eef.py's `PolicyClient(host=..., port=...)`.
    """

    def __init__(self, host: str, port: int = 5555, timeout_ms: int = 30000,
                 api_token: str | None = None):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._closed = False
        self.context = zmq.Context()
        self._init_socket()
        self.last_timing: dict = {}

        log.info(f"Connecting to GR00T policy server tcp://{host}:{port}")
        if not self.ping():
            log.warning("Server did not answer ping yet — will still try to infer")

        # Sync the action-key order + language key with the server's config.
        self.action_keys = list(FALLBACK_ACTION_KEYS)
        self.language_key = FALLBACK_LANGUAGE_KEY
        try:
            mc = self.call_endpoint("get_modality_config", requires_input=False)
            self.action_keys = self._extract_keys(mc.get("action")) or self.action_keys
            lang = self._extract_keys(mc.get("language"))
            if lang:
                self.language_key = lang[0]
            log.info(f"Server modality: action_keys={self.action_keys} "
                     f"language_key={self.language_key!r}")
        except Exception as e:
            log.warning(f"get_modality_config failed ({e}); using fallbacks "
                        f"action_keys={self.action_keys} language_key={self.language_key!r}")

    # ---------- ZMQ transport (mirrors gr00t PolicyClient) ----------

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None,
                      requires_input: bool = True):
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        payload = mnp.packb(request)
        t_pack = time.time()
        try:
            self.socket.send(payload)
            t_send = time.time()
            message = self.socket.recv()
            t_recv = time.time()
        except zmq.error.Again:
            # Timeout leaves a REQ socket unusable — recreate before re-raising.
            self._init_socket()
            raise
        response = mnp.unpackb(message, raw=False)
        t_unpack = time.time()

        self._last_call_timing = {
            "send_s": t_send - t_pack,
            "wait_recv_s": t_recv - t_send,
            "unpack_s": t_unpack - t_recv,
            "bytes_sent": len(payload),
            "bytes_recv": len(message),
        }
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    # ---------- the interface main_eef.py drives ----------

    def infer(self, obs: dict) -> dict:
        """Send one flat observation (the dict build_obs() assembles), return
        {"actions": ndarray[H, 16]}. Converts flat->nested/batched, calls
        get_action, and concatenates the per-modality action chunks."""
        t0 = time.time()
        observation = self._to_groot_obs(obs)
        t_pack = time.time()

        action_dict, _info = self.call_endpoint(
            "get_action", {"observation": observation, "options": None}
        )

        # (1, H, D) per key -> concat on last axis in the server's order -> (H, 16)
        parts = []
        for k in self.action_keys:
            a = np.asarray(action_dict[k])
            parts.append(a[0])  # drop batch dim -> (H, D)
        actions = np.concatenate(parts, axis=-1)

        ct = getattr(self, "_last_call_timing", {})
        self.last_timing = {
            "total_s": time.time() - t0,
            "pack_s": (t_pack - t0) + 0.0,  # our flat->nested build
            "send_s": ct.get("send_s", 0.0),
            "wait_recv_s": ct.get("wait_recv_s", 0.0),
            "unpack_s": ct.get("unpack_s", 0.0),
            "bytes_sent": ct.get("bytes_sent", -1),
            "bytes_recv": ct.get("bytes_recv", -1),
        }
        return {"actions": actions}

    def _to_groot_obs(self, obs: dict) -> dict:
        """Flat build_obs() dict -> nested, batched (B=1, T=1) GR00T observation."""
        video = {}
        for flat_key, groot_key in VIDEO_KEY_MAP.items():
            img = _to_rgb_uint8(obs[flat_key])  # (H, W, 3)
            video[groot_key] = img[None, None]  # (1, 1, H, W, 3)

        state_vec = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        state = {
            name: state_vec[sl][None, None]  # (1, 1, D)
            for name, sl in STATE_SLICES.items()
        }

        language = {self.language_key: [[obs["prompt"]]]}  # (B=1, T=1)
        return {"video": video, "state": state, "language": language}

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for res, kw in ((getattr(self, "socket", None), {"linger": 0}),
                        (getattr(self, "context", None), {})):
            if res is not None:
                try:
                    (res.close(**kw) if hasattr(res, "close") else res.term())
                except Exception:
                    pass

    @staticmethod
    def _extract_keys(modality_cfg) -> list[str] | None:
        """Pull modality_keys out of a get_modality_config entry, whether it came
        back as a raw dict or a ModalityConfig marker ({'as_json': {...}})."""
        if not isinstance(modality_cfg, dict):
            return None
        src = modality_cfg.get("as_json", modality_cfg.get(b"as_json", modality_cfg))
        if isinstance(src, dict):
            keys = src.get("modality_keys", src.get(b"modality_keys"))
            if keys:
                return [k.decode() if isinstance(k, bytes) else k for k in keys]
        return None
