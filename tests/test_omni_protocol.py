import base64

import numpy as np

from reachy_mini_live_chat.omni import protocol


def test_pcm_b64_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.123], dtype=np.float32)
    b = protocol.pcm_f32_to_b64(x)
    y = protocol.b64_to_pcm_f32(b)
    assert np.allclose(x, y, atol=1e-6)
    assert y.dtype == np.float32


def test_b64_roundtrip_is_float32_le():
    # a known value encoded as little-endian float32 must decode back exactly
    x = np.array([0.25], dtype="<f4")
    b64 = base64.b64encode(x.tobytes()).decode()
    assert protocol.b64_to_pcm_f32(b64)[0] == 0.25


def test_b64_tolerates_junk_and_data_url():
    assert len(protocol.b64_to_pcm_f32("")) == 0
    assert len(protocol.b64_to_pcm_f32("!!!notbase64!!!")) == 0
    # data-url prefix is stripped
    x = np.array([0.5, 0.25], dtype=np.float32)
    b = protocol.pcm_f32_to_b64(x)
    y = protocol.b64_to_pcm_f32("data:audio/pcm;base64," + b)
    assert np.allclose(x, y)


def test_build_session_init_full_duplex():
    msg = protocol.build_session_init(
        mode="full_duplex", use_tts=True, system_prompt="hi",
        config={"temperature": 0.7},
    )
    assert msg["type"] == "session.init"
    p = msg["payload"]
    assert p["mode"] == "full_duplex" and p["use_tts"] is True
    assert p["system_prompt"] == "hi"
    assert p["config"] == {"temperature": 0.7}
    assert "voice" not in p  # omitted when no ref audio


def test_build_session_init_with_voice():
    msg = protocol.build_session_init(ref_audio_b64="QUJD")
    assert msg["payload"]["voice"]["ref_audio"] == "QUJD"


def test_build_input_append_shape():
    audio = np.zeros(16000, dtype=np.float32)
    msg = protocol.build_input_append(audio, frame_b64="ZmFrZQ==")
    assert msg["type"] == "input.append"
    inp = msg["input"]
    assert "audio" in inp and isinstance(inp["audio"], str)
    assert inp["video_frames"] == ["ZmFrZQ=="]
    assert "messages" not in inp  # must NOT carry turn_based messages
    # audio decodes back to the right length
    assert len(protocol.b64_to_pcm_f32(inp["audio"])) == 16000


def test_build_input_append_no_frame():
    msg = protocol.build_input_append(np.zeros(8, dtype=np.float32))
    assert "video_frames" not in msg["input"]


def test_parse_session_created():
    ev = protocol.parse_event({"type": "session.created", "session_id": "s1", "mode": "full_duplex"})
    assert ev.category == protocol.EV_CREATED
    assert ev.session_id == "s1" and ev.mode == "full_duplex"


def test_parse_text_delta():
    ev = protocol.parse_event({
        "type": "response.output.delta", "kind": "text",
        "session_id": "s", "response_id": "r", "text": "你好",
    })
    assert ev.category == protocol.EV_TEXT and ev.text == "你好"
    assert ev.response_id == "r"


def test_parse_audio_delta():
    pcm = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ev = protocol.parse_event({
        "type": "response.output.delta", "kind": "audio",
        "audio": protocol.pcm_f32_to_b64(pcm),
    })
    assert ev.category == protocol.EV_AUDIO
    assert ev.audio is not None and np.allclose(ev.audio, pcm)


def test_parse_listen_delta():
    ev = protocol.parse_event({"type": "response.output.delta", "kind": "listen", "session_id": "s"})
    assert ev.category == protocol.EV_LISTEN


def test_parse_response_done_with_and_without_audio():
    ev = protocol.parse_event({"type": "response.done", "text": "done", "reason": "turn_end", "audio": None})
    assert ev.category == protocol.EV_DONE and ev.reason == "turn_end" and ev.audio is None
    ev2 = protocol.parse_event({
        "type": "response.done", "text": "x", "reason": "turn_end",
        "audio": protocol.pcm_f32_to_b64(np.ones(4, dtype=np.float32)),
    })
    assert ev2.audio is not None and len(ev2.audio) == 4


def test_parse_session_closed_and_unknown():
    ev = protocol.parse_event({"type": "session.closed", "reason": "client_closed"})
    assert ev.category == protocol.EV_CLOSED and ev.reason == "client_closed"
    assert protocol.parse_event({"type": "whatever"}).category == protocol.EV_OTHER
    assert protocol.parse_event("not a dict").category == protocol.EV_OTHER


def test_parse_gateway_queued_status():
    ev = protocol.parse_event({"type": "session.queued", "position": 2, "estimated_wait_s": 8})
    assert ev.category == protocol.EV_STATUS and ev.status == "session.queued"
    assert ev.raw.get("position") == 2


def test_parse_error_event():
    ev = protocol.parse_event({"type": "error", "error": {"code": "queue_full", "message": "Queue full"}})
    assert ev.category == protocol.EV_ERROR and ev.reason == "queue_full"
    assert "Queue full" in ev.message
