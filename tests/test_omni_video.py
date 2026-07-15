import base64

import numpy as np

from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.omni import video
from reachy_mini_live_chat.omni.video import VideoGrabber


class _Media:
    def __init__(self, frame):
        self._frame = frame

    def get_frame(self):
        return self._frame


class _Mini:
    def __init__(self, frame):
        self.media = _Media(frame)


def _cfg():
    c = Config()
    c.omni_send_video = True
    c.omni_video_fps = 1000.0  # effectively no throttle for the test
    return c


def test_resize_max_edge():
    frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
    small = video._resize_max_edge(frame, 448)
    assert max(small.shape[:2]) == 448


def test_latest_b64_returns_base64_jpeg():
    frame = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    g = VideoGrabber(_cfg(), _Mini(frame))
    g._refresh()  # background loop does this; drive it once directly for the test
    b64 = g.latest_b64()
    assert isinstance(b64, str) and len(b64) > 100
    raw = base64.b64decode(b64)
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_disabled_returns_none():
    c = _cfg()
    c.omni_send_video = False
    g = VideoGrabber(c, _Mini(np.zeros((10, 10, 3), dtype=np.uint8)))
    g._refresh()
    assert g.latest_b64() is None


def test_no_frame_returns_last_or_none():
    g = VideoGrabber(_cfg(), _Mini(None))
    g._refresh()
    assert g.latest_b64() is None
