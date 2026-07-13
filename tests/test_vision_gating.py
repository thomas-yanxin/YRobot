import numpy as np

from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.vision import gating
from reachy_mini_live_chat.vision.gating import VisionGate


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
    c.enable_vision = True
    return c


def test_wants_frame_visual_question_zh():
    g = VisionGate(_cfg(), _Mini(None))
    assert g.wants_frame("你能看到我手里拿的是什么吗")


def test_wants_frame_visual_question_en():
    g = VisionGate(_cfg(), _Mini(None))
    assert g.wants_frame("what am I holding right now?")


def test_no_frame_for_nonvisual():
    g = VisionGate(_cfg(), _Mini(None))
    assert not g.wants_frame("今天天气怎么样")
    assert g.maybe_keyframe("讲个笑话") is None


def test_keyframe_returned_and_base64():
    frame = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    g = VisionGate(_cfg(), _Mini(frame))
    b64 = g.maybe_keyframe("这是什么颜色")
    assert isinstance(b64, str) and len(b64) > 100


def test_resize_max_edge():
    frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
    small = gating._resize_max_edge(frame, 768)
    assert max(small.shape[:2]) == 768


def test_phash_hamming_identity():
    frame = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    h = gating._phash(frame)
    assert gating._hamming(h, h) == 0


def test_scene_change_detects_difference():
    # non-uniform, structurally different images (uniform frames hash degenerately)
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    a = np.stack([grad, grad, grad], axis=2)              # horizontal gradient
    b = np.stack([grad.T, grad.T, grad.T], axis=2)        # vertical gradient
    assert gating._hamming(gating._phash(a), gating._phash(b)) > 0
