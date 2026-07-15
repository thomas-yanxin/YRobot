"""Camera frame → downscaled base64 JPEG for the omni video stream.

We attach one *current* frame to each ~1 s audio chunk (continuous ~1 fps), which
matches the model's 1 Hz time-division design. Frames are downscaled to
``omni_video_max_edge`` and JPEG-compressed to keep the uplink light. Grabbing is
throttled to ``omni_video_fps`` and the last encoding is cached, so calling
:meth:`latest_b64` faster than the fps just re-uses the cached frame.

Encoding degrades gracefully: cv2/PIL downscaled JPEG → the SDK's own
``media.get_frame_jpeg()`` (full-res, zero extra deps). So a minimal on-robot
install needs no image library at all — pillow/opencv only buy smaller frames.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Optional

import numpy as np

from ..config import Config

log = logging.getLogger("live_chat.omni.video")


class VideoGrabber:
    """Grabs + encodes frames on a **background thread** so the omni audio uplink
    never blocks on a slow camera read. :meth:`latest_b64` just returns the most
    recent cached encoding (non-blocking)."""

    def __init__(self, cfg: Config, mini) -> None:
        self.cfg = cfg
        self.mini = mini
        self._last_b64: Optional[str] = None
        self._min_dt = 1.0 / max(0.1, cfg.omni_video_fps)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.cfg.omni_send_video or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def latest_b64(self) -> Optional[str]:
        """Non-blocking: return the most recently encoded frame (or None)."""
        return self._last_b64 if self.cfg.omni_send_video else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._refresh()
            except Exception as e:
                log.debug("video refresh error: %s", e)
            time.sleep(max(0.0, self._min_dt - (time.monotonic() - t0)))

    def _refresh(self) -> None:
        jpeg = None
        frame = self._grab()
        if frame is not None:
            small = _resize_max_edge(frame, self.cfg.omni_video_max_edge)
            jpeg = _encode_jpeg(small, self.cfg.omni_video_jpeg_quality)
        if jpeg is None:
            # No frame array or no encoder (pillow/opencv absent) → let the SDK
            # hand us a ready-made JPEG. Full resolution, but zero extra deps.
            jpeg = self._grab_jpeg()
        if jpeg is not None:
            self._last_b64 = base64.b64encode(jpeg).decode("ascii")

    def _grab(self) -> Optional[np.ndarray]:
        try:
            return self.mini.media.get_frame()
        except Exception as e:
            log.debug("get_frame error: %s", e)
            return None

    def _grab_jpeg(self) -> Optional[bytes]:
        get = getattr(self.mini.media, "get_frame_jpeg", None)
        if get is None:
            return None
        try:
            return get()
        except Exception as e:
            log.debug("get_frame_jpeg error: %s", e)
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
