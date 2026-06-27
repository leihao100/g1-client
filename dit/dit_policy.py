"""DiT4DiT data send/receive layer for the G1 client.

Drop-in alternative to g1_client/openpi_policy.py, talking to a DiT4DiT
`deployment/model_server/server_policy.py` (WebsocketPolicyServer) instead of an
openpi serve_policy.py. As with the openpi layer, the robot-facing control code
(arm/gripper/camera) is untouched — ONLY how we talk to the policy server changes.

openpi protocol                         DiT4DiT protocol (this file)
--------------------------------------  --------------------------------------------
stateless infer(obs) -> {"actions"}     stateless, but a structured envelope:
server un-normalizes (norm_stats live      request  {"type":"infer","examples":[ex]}
server-side), returns RAW-unit actions     response {"ok":True,"data":{"normalized_actions": ...}}
images sent as decoded RGB arrays        server does NOT normalize. It expects
                                         NORMALIZED state in and returns NORMALIZED
                                         actions out, so this client must do both
                                         halves of the normalization locally.

Because the DiT4DiT server returns *normalized* actions and expects *normalized*
state (DiT4DiT.predict_action -> {"normalized_actions": ...}), this wrapper holds
the checkpoint's dataset_statistics.json and reproduces the EXACT preprocessing of
examples/Real_G1/eval_files/eval_policy.py — the purpose-built Real-G1 inference
reference for this checkpoint family:

  * state:  min-max normalize to [-1, 1], pad to 64 (gripper dims sent live by
            default; --zero-gripper-state to zero them for constant-0 checkpoints).
  * image:  3 cameras [ego, left_wrist, right_wrist], each resized to 224x224,
            RGB uint8 HWC; the server concatenates them into the conditioning frame.
  * action: take the first real_action_dim cols, clip to [-1, 1], min-max
            un-normalize back to RAW joint radians.

CONTRACT TO VERIFY against the checkpoint (same silent-desync class the repo's
CLAUDE.md warns about — wrong here means the robot moves to wrong joint angles
with no error):
  * State preprocessing is min_max to [-1,1] + pad to 64, matching the
    UnitreeG1Dex1ThreeCamConfig training transform. Gripper-zeroing is OFF by
    default because this dataset's gripper state varies; pass --zero-gripper-state
    only for a checkpoint trained with constant-0 gripper state.
  * dataset_statistics.json must be the one paired with the served checkpoint
    (run_dir/dataset_statistics.json), with ["<key>"]["state"]/["action"]
    each carrying "min"/"max" lists.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from g1_client.policy_client import PolicyClient

log = logging.getLogger("g1_dit.policy")

# Matches the G1 training YAML (config/real_robot/dit4dit_g1.yaml):
#   image_size: [224, 224], max_state_dim: 64.
TRAIN_IMAGE_SIZE = (224, 224)  # (W, H) for cv2.resize
MAX_STATE_DIM = 64


def _load_norm_stats(path: str, unnorm_key: Optional[str]):
    """Read dataset_statistics.json and return (state_stats, action_stats).

    Mirrors eval_policy.py: a single-dataset checkpoint needs no key; otherwise
    --unnorm-key must name one of the blocks.
    """
    with open(path, "r") as f:
        norm_stats = json.load(f)
    if unnorm_key is None:
        assert len(norm_stats) == 1, (
            f"{Path(path).name} has multiple datasets; pass --unnorm-key from: "
            f"{list(norm_stats.keys())}"
        )
        unnorm_key = next(iter(norm_stats.keys()))
    assert unnorm_key in norm_stats, (
        f"--unnorm-key {unnorm_key!r} not in {list(norm_stats.keys())}"
    )
    block = norm_stats[unnorm_key]
    return block["state"], block["action"]


def _normalize_state(state_raw: np.ndarray, state_stats: dict,
                     zero_gripper: bool = True) -> np.ndarray:
    """Min-max normalize the 16-dim state to [-1, 1], pad to MAX_STATE_DIM.

    Verbatim port of eval_policy.get_action_from_model's state branch, including
    the gripper-zeroing hack: the two gripper dims were constant 0 in the
    collected data, so the model was conditioned on 0 there — feeding the live
    gripper q instead would be off-distribution. The min/max for those dims is
    degenerate (max==min -> nan), which the subsequent zero-write overwrites.
    """
    mn = np.asarray(state_stats["min"], dtype=np.float64)
    mx = np.asarray(state_stats["max"], dtype=np.float64)
    state = state_raw.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        state = (state - mn) / (mx - mn) * 2 - 1
    if zero_gripper:
        state[-1] = 0.0
        state[-2] = 0.0
    state = state.reshape(1, -1)  # (1, state_dim)
    if state.shape[1] < MAX_STATE_DIM:
        pad = np.zeros((1, MAX_STATE_DIM - state.shape[1]), dtype=state.dtype)
        state = np.concatenate([state, pad], axis=-1)
    return state.astype(np.float32)


def _unnormalize_actions(norm_actions: np.ndarray, action_stats: dict) -> np.ndarray:
    """Map normalized [-1, 1] actions back to raw units (joint radians).

    Verbatim port of eval_policy.unnormalize_actions: the model emits a padded
    action_dim (32); the real action is the first len(min) cols.
    """
    high = np.asarray(action_stats["max"], dtype=np.float64)
    low = np.asarray(action_stats["min"], dtype=np.float64)
    real_dim = len(high)
    a = np.clip(norm_actions[:, :real_dim], -1, 1)
    return 0.5 * (a + 1) * (high - low) + low


class DitPolicy:
    """Stateless wrapper over the native PolicyClient speaking the DiT4DiT
    envelope protocol, with eval_policy.py preprocessing folded in.

    infer(obs) takes a RAW observation and returns {"actions": ndarray[H, 16]} in
    raw joint radians, so the inference loop in main-dit.py is identical in shape
    to main_openpi.py's (which also reads result["actions"]).
    """

    def __init__(self, host: str, port: int, norm_stats_path: str,
                 unnorm_key: Optional[str] = None, zero_gripper_state: bool = False,
                 api_key: Optional[str] = None):
        self._state_stats, self._action_stats = _load_norm_stats(norm_stats_path, unnorm_key)
        self._zero_gripper = zero_gripper_state
        log.info(f"Loaded norm_stats from {norm_stats_path} "
                 f"(action dims={len(self._action_stats['max'])}, "
                 f"state dims={len(self._state_stats['max'])}, "
                 f"zero_gripper_state={zero_gripper_state})")
        log.info(f"Connecting to DiT4DiT policy server ws://{host}:{port}")
        self._client = PolicyClient(host=host, port=port, api_key=api_key)
        log.info(f"Server metadata: {self._client.get_server_metadata()}")

    @property
    def last_timing(self):
        # Delegate so main-dit.py's per-chunk latency logging works unchanged.
        return self._client.last_timing

    def get_server_metadata(self) -> dict:
        return self._client.get_server_metadata()

    def infer(self, obs: dict) -> dict:
        """obs = {"image": list of RGB uint8 HWC (one per camera), "state": raw
        float32 (16,), "prompt": str}.

        Returns {"actions": ndarray[H, real_action_dim]} in raw joint radians.
        """
        views_raw = obs["image"] if isinstance(obs["image"], (list, tuple)) else [obs["image"]]
        views = [cv2.resize(v, TRAIN_IMAGE_SIZE) for v in views_raw]  # each (224,224,3) RGB
        state2d = _normalize_state(obs["state"], self._state_stats, self._zero_gripper)

        # predict_action(**msg) requires an "examples" key; the server resizes each
        # view to the training resolution and concatenates them side-by-side into
        # the one conditioning frame (must arrive in training video_keys order).
        payload = {
            "type": "infer",
            "request_id": "g1-dit",
            "examples": [{
                "image": views,
                "lang": obs["prompt"],
                "state": state2d,
            }],
        }
        resp = self._client.infer(payload)
        if not isinstance(resp, dict) or not resp.get("ok", False):
            err = resp.get("error") if isinstance(resp, dict) else resp
            raise RuntimeError(f"DiT4DiT server returned an error: {err}")

        normalized = np.asarray(resp["data"]["normalized_actions"][0], dtype=np.float64)  # [H, 32]
        actions = _unnormalize_actions(normalized, self._action_stats)  # [H, real_dim]
        return {"actions": actions}

    def close(self):
        self._client.close()
