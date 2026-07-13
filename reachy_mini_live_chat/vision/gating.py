"""Decide *when* to spend vision tokens, and compress the frame when we do.

Token-saving policy (see plan.md §6):
1. **Intent gate** — only consider a frame when the user actually asks something visual
   (bilingual keyword check). This is the primary lever.
2. **Scene-change dedup** — perceptual hash vs the last kept frame; skip near-duplicates
   (used by the optional always-on awareness mode).
3. **Compress** — resize long edge to ``VISION_MAX_EDGE`` and JPEG at ``VISION_JPEG_QUALITY``.
4. The LLM router keeps only **one** keyframe in context, so cost is capped per visual turn.

Everything degrades gracefully: cv2 → PIL → numpy for resize/encode; imagehash → average-hash.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Optional

import numpy as np

from ..config import Config

log = logging.getLogger("live_chat.vision")

# Bilingual cues that indicate the user is asking about what the robot sees.
_VISUAL_CUES = [
    # zh
    "看", "这是什么", "这个是", "那是什么", "手里", "拿的", "拿着", "什么颜色", "画面",
    "镜头", "你看到", "看到了", "面前", "前面", "长什么样", "认识", "识别", "读一下", "念一下",
    # en
    "what is this", "what's this", "what am i holding", "what do you see", "can you see",
    "look at", "what color", "in front of", "on the screen", "describe what", "read this",
    "read the", "recognize", "identify", "hold up", "holding",
]
_CUE_RE = re.compile("|".join(re.escape(c) for c in _VISUAL_CUES), re.IGNORECASE)


class VisionGate:
    def __init__(self, cfg: Config, mini) -> None:
        self.cfg = cfg
        self.mini = mini
        self._last_hash: Optional[int] = None

    # -- decision -----------------------------------------------------------
    def wants_frame(self, text: str) -> bool:
        if not self.cfg.enable_vision or not text:
            return False
        return bool(_CUE_RE.search(text))

    def maybe_keyframe(self, text: str) -> Optional[str]:
        """Return a base64 JPEG keyframe iff the turn warrants vision, else None."""
        if not self.wants_frame(text):
            return None
        return self.capture_keyframe(force=True)

    def capture_keyframe(self, force: bool = False) -> Optional[str]:
        frame = self._grab()
        if frame is None:
            return None
        h = _phash(frame)
        if not force and self._last_hash is not None and _hamming(h, self._last_hash) <= self.cfg.vision_phash_threshold:
            return None  # scene hasn't changed enough; save the tokens
        self._last_hash = h
        small = _resize_max_edge(frame, self.cfg.vision_max_edge)
        jpeg = _encode_jpeg(small, self.cfg.vision_jpeg_quality)
        if jpeg is None:
            return None
        log.info("vision: sending keyframe (%d bytes jpeg)", len(jpeg))
        return base64.b64encode(jpeg).decode("ascii")

    def scene_changed(self, threshold: Optional[int] = None) -> bool:
        """For optional always-on awareness: did the view change materially?"""
        frame = self._grab()
        if frame is None:
            return False
        h = _phash(frame)
        thr = self.cfg.vision_phash_threshold if threshold is None else threshold
        changed = self._last_hash is None or _hamming(h, self._last_hash) > thr
        if changed:
            self._last_hash = h
        return changed

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
    nw, nh = int(w * scale), int(h * scale)
    try:
        import cv2

        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        ys = (np.linspace(0, h - 1, nh)).astype(np.int64)
        xs = (np.linspace(0, w - 1, nw)).astype(np.int64)
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

        rgb = frame[:, :, ::-1]  # BGR -> RGB
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="JPEG", quality=int(quality))
        return bio.getvalue()
    except Exception as e:
        log.debug("jpeg encode failed: %s", e)
        return None


def _phash(frame: np.ndarray) -> int:
    """Perceptual hash. Uses imagehash if present, else a 8x8 average-hash."""
    try:
        import imagehash
        from PIL import Image

        rgb = frame[:, :, ::-1]
        return int(str(imagehash.phash(Image.fromarray(rgb))), 16)
    except Exception:
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame
        try:
            import cv2

            small = cv2.resize(gray.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
        except Exception:
            ys = np.linspace(0, gray.shape[0] - 1, 8).astype(np.int64)
            xs = np.linspace(0, gray.shape[1] - 1, 8).astype(np.int64)
            small = gray[ys][:, xs].astype(np.float32)
        bits = (small > small.mean()).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
