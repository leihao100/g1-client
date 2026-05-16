"""Camera wrapper for LingBot-VA inference.

Subscribes to the three G1 cameras via teleimager.ImageClient and returns each
as a JPEG-encoded byte string (256x320, quality 90) matching the training
format (training data was H.264 mp4, so q90 JPEG is well within distribution):
- cam_left_high: left half of the binocular head camera, resized, JPEG
- cam_left_wrist:  left wrist camera, resized, JPEG
- cam_right_wrist: right wrist camera, resized, JPEG

Resize: source 480x640 → 256x320 (preserves 4:3 aspect ratio).

COLOR ORDER CONTRACT: frames are JPEG-encoded straight from cv2's native BGR
(no BGR→RGB here anymore). The server must cv2.imdecode (→ BGR) and then do
the BGR→RGB conversion the client previously did. Sending ~15-25 KiB JPEG vs
240 KiB raw cuts the dominant kv_cache upload ~10-15x.
"""

import cv2
import numpy as np

try:
    from teleimager import ImageClient
except ImportError:
    # The installed teleimager package may not re-export ImageClient at the
    # package root — import from the submodule directly.
    from teleimager.image_client import ImageClient


CAM_HEIGHT = 256
CAM_WIDTH = 320
JPEG_QUALITY = 90


class CameraClient:
    def __init__(self, host: str = "192.168.123.164", request_port: int = 60000):
        # request_bgr=True so .bgr is decoded for us
        self._client = ImageClient(host=host, request_port=request_port, request_bgr=True)
        self._cfg = self._client.get_cam_config()

        if not self._cfg["head_camera"]["enable_zmq"]:
            raise RuntimeError("head_camera ZMQ not enabled on image server")
        if not self._cfg["left_wrist_camera"]["enable_zmq"]:
            raise RuntimeError("left_wrist_camera ZMQ not enabled on image server")
        if not self._cfg["right_wrist_camera"]["enable_zmq"]:
            raise RuntimeError("right_wrist_camera ZMQ not enabled on image server")

        self._head_binocular = bool(self._cfg["head_camera"].get("binocular", False))

    @staticmethod
    def _resize_jpeg(bgr: np.ndarray) -> bytes:
        """Resize to 256x320 and JPEG-encode in cv2's native BGR order.

        No BGR→RGB here — the server decodes the JPEG (→ BGR) and does the
        color conversion. Keep this in lockstep with the server decode path.
        """
        if bgr.shape[0] != CAM_HEIGHT or bgr.shape[1] != CAM_WIDTH:
            bgr = cv2.resize(bgr, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("cv2.imencode failed to JPEG-encode frame")
        return buf.tobytes()

    def _take_left_half(self, bgr: np.ndarray) -> np.ndarray:
        if not self._head_binocular:
            return bgr
        # Binocular feed is two views concatenated along width; take the left half.
        w = bgr.shape[1]
        return bgr[:, : w // 2, :]

    def get_obs_images(self) -> dict:
        """Return the three camera frames as JPEG bytes (no prompt key).

        Note: head/lw/rw are fetched sequentially — they are a few ms apart in
        time, not a true simultaneous snapshot.
        """
        head = self._client.get_head_frame()
        lw = self._client.get_left_wrist_frame()
        rw = self._client.get_right_wrist_frame()

        if head.bgr is None or lw.bgr is None or rw.bgr is None:
            raise RuntimeError("Got None from one or more cameras")

        head_bgr = self._take_left_half(head.bgr)
        return {
            "observation.images.cam_left_high":   self._resize_jpeg(head_bgr),
            "observation.images.cam_left_wrist":  self._resize_jpeg(lw.bgr),
            "observation.images.cam_right_wrist": self._resize_jpeg(rw.bgr),
        }

    def get_obs(self, prompt: str) -> dict:
        """Full inference-payload obs: images + task prompt."""
        obs = self.get_obs_images()
        obs["task"] = prompt
        return obs

    def close(self):
        self._client.close()
