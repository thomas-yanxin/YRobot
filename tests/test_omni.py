import base64

import numpy as np
import pytest

from yrobot.omni import (
    OmniClient,
    build_input_append,
    build_session_init,
    decode_pcm,
    encode_pcm,
)


def test_pcm_round_trip_is_little_endian_float32() -> None:
    samples = np.array([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=np.float32)
    encoded = encode_pcm(samples)
    assert base64.b64decode(encoded) == samples.astype("<f4").tobytes()
    np.testing.assert_array_equal(decode_pcm(encoded), samples)


def test_pcm_codec_rejects_invalid_audio() -> None:
    with pytest.raises(ValueError, match="mono"):
        encode_pcm(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        decode_pcm(base64.b64encode(b"abc").decode())


def test_protocol_messages_match_full_duplex_backend() -> None:
    init = build_session_init("hello")
    assert init == {
        "type": "session.init",
        "payload": {"mode": "full_duplex", "use_tts": True, "system_prompt": "hello"},
    }

    append = build_input_append(np.zeros(16_000, dtype=np.float32), b"jpeg")
    assert append["type"] == "input.append"
    assert set(append["input"]) == {
        "audio",
        "video_frames",
        "max_slice_nums",
    }
    assert append["input"]["max_slice_nums"] == 1


def test_event_parser_requires_object_and_type() -> None:
    assert OmniClient._parse_event('{"type":"session.created"}') == {"type": "session.created"}
    with pytest.raises(ValueError):
        OmniClient._parse_event("[]")
