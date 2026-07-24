"""Unit tests for protocol helpers (no network)."""

import base64
import json
import threading

import numpy as np
import pytest

from yrobot.config import Settings, normalize_url
from yrobot.realtime import RealtimeClient, ThinkFilter, _parse_delta


def test_normalize_url_variants():
    full = "wss://minicpmo45.modelbest.cn/v1/realtime?mode=audio"
    assert normalize_url("minicpmo45.modelbest.cn") == full
    assert (
        normalize_url("wss://minicpmo45.modelbest.cn/v1/realtime?mode=video")
        == "wss://minicpmo45.modelbest.cn/v1/realtime?mode=video"
    )
    assert normalize_url("wss://minicpmo45.modelbest.cn/v1/realtime?mode=video", "audio") == full
    assert (
        normalize_url("wss://10.0.16.184:8006") == "wss://10.0.16.184:8006/v1/realtime?mode=audio"
    )


def test_parse_audio_delta_roundtrip():
    pcm = np.array([0.0, 0.5, -0.5], dtype="<f4")
    event = {
        "type": "response.output.delta",
        "kind": "audio",
        "audio": base64.b64encode(pcm.tobytes()).decode(),
        "response_id": "resp-1",
        "input_id": "input-1",
        "metrics": {"kv_cache_length": 321},
    }
    delta = _parse_delta(event)
    assert delta.kind == "audio"
    assert np.allclose(delta.audio, pcm)
    assert delta.response_id == "resp-1"
    assert delta.input_id == "input-1"
    assert delta.metrics["kv_cache_length"] == 321


def test_parse_listen_and_text():
    listen = _parse_delta({"kind": "listen"})
    assert (listen.kind, listen.text, len(listen.audio)) == ("listen", "", 0)
    assert _parse_delta({"kind": "text", "text": "hi"}).text == "hi"


def test_send_chunk_carries_input_id_and_exact_model_unit():
    class FakeSocket:
        def __init__(self):
            self.messages = []

        def send(self, raw):
            self.messages.append(json.loads(raw))

    client = RealtimeClient(Settings(), lambda delta: None, lambda reason: None)
    client._ws = FakeSocket()
    client._send_lock = threading.Lock()
    client.send_chunk(
        np.zeros(16_000, np.float32),
        jpeg=None,
        force_listen=True,
        input_id="input-42",
    )
    sent = client._ws.messages[0]
    assert sent["input"]["input_id"] == "input-42"
    assert sent["input"]["force_listen"] is True


def test_send_chunk_rejects_partial_inference_unit():
    client = RealtimeClient(Settings(), lambda delta: None, lambda reason: None)
    with pytest.raises(ValueError, match="16000 samples"):
        client.send_chunk(
            np.zeros(8_000, np.float32),
            jpeg=None,
            force_listen=True,
            input_id="partial",
        )


def test_think_filter_strips_spans_across_deltas():
    f = ThinkFilter()
    assert f.feed("hello <thi") == "hello "
    assert f.feed("nk>secret plan</th") == ""
    assert f.feed("ink> world") == " world"


def test_think_filter_passes_plain_text():
    f = ThinkFilter()
    assert f.feed("你好，") + f.feed("今天天气不错。") == "你好，今天天气不错。"
