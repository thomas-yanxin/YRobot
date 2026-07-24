"""Unit tests for app-level helpers (no hardware, no network)."""

import numpy as np
import pytest

from yrobot.main import FRAME_MAX_DIM, shrink_jpeg


def test_shrink_jpeg_downscales_to_model_vision_size():
    cv2 = pytest.importorskip("cv2")
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    jpeg = shrink_jpeg(frame)
    assert jpeg is not None
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == FRAME_MAX_DIM
    full = cv2.imencode(".jpg", frame)[1].tobytes()
    assert len(jpeg) < len(full) / 3  # meaningfully lighter on the uplink


def test_shrink_jpeg_keeps_small_frames():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    decoded = cv2.imdecode(np.frombuffer(shrink_jpeg(frame), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (240, 320)


def test_shrink_jpeg_none_frame():
    assert shrink_jpeg(None) is None
