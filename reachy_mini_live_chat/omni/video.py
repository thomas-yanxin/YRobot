"""Camera frame → downscaled base64 JPEG for the omni video stream.

We attach one *current* frame to each ~1 s audio chunk (continuous ~1 fps), which
matches the model's 1 Hz time-division design. Frames are downscaled to
``omni_video_max_edge`` and JPEG-compressed to keep the uplink light. Grabbing is
throttled to ``omni_video_fps`` and the last encoding is cached, so calling
:meth:`latest_b64` faster than the fps just re-uses the cached frame.

Encoding degrades gracefully: cv2 → PIL → (resize) numpy.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import numpy as np

from ..config import Config

log = logging.getLogger("live_chat.omni.video")


class VideoGrabber:
    def __init__(self, cfg: Config, mini) -> None:
        self.cfg = cfg
        self.mini = mini
        self._last_t = 0.0
        self._last_b64: Optional[str] = None
        self._min_dt = 1.0 / max(0.1, cfg.omni_video_fps)

    def latest_b64(self) -> Optional[str]:
        """Return the current frame as base64 JPEG (cached between fps ticks)."""
        if not self.cfg.omni_send_video:
            return None
        now = time.monotonic()
        if self._last_b64 is not None and now - self._last_t < self._min_dt:
            return self._last_b64
        frame = self._grab()
        if frame is None:
            return self._last_b64  # keep the last good frame rather than dropping video
        small = _resize_max_edge(frame, self.cfg.omni_video_max_edge)
        jpeg = _encode_jpeg(small, self.cfg.omni_video_jpeg_quality)
        if jpeg is None:
            return self._last_b64
        self._last_b64 = base64.b64encode(jpeg).decode("ascii")
        self._last_t = now
        return self._last_b64

    def _grab(self) -> Optional[np.ndarray]:
        try:
            return self.mini.media.get_frame()
        except Exception as e:
            log.debug("get_frame error: %s", e)
            return None


# ---- image helpers (cv2 -> PIL -> numpy fallbacks) --------------------------
def _resize_max_edge(frame: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_edge / max(h, w)
    if scale >= 1.0:
        return frame
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    try:
        import cv2

        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        ys = np.linspace(0, h - 1, nh).astype(np.int64)
        xs = np.linspace(0, w - 1, nw).astype(np.int64)
        return frame[ys][:, xs]


def _encode_jpeg(frame: np.ndarray, quality: int) -> Optional[bytes]:
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        return buf.tobytes() if ok else None
    except Exception:
        pass
    try:
        import io

        from PIL import Image

        rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame  # BGR -> RGB
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="JPEG", quality=int(quality))
        return bio.getvalue()
    except Exception as e:
        log.debug("jpeg encode failed: %s", e)
        return None
