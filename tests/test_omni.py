import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosed

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

    interrupted = build_input_append(
        np.zeros(16_000, dtype=np.float32),
        None,
        force_listen=True,
    )
    assert interrupted["input"]["force_listen"] is True


def test_event_parser_requires_object_and_type() -> None:
    assert OmniClient._parse_event('{"type":"session.created"}') == {"type": "session.created"}
    with pytest.raises(ValueError):
        OmniClient._parse_event("[]")


def test_transport_error_reports_last_event_and_received_audio() -> None:
    audio = encode_pcm(np.zeros(24_000, dtype=np.float32))

    class WebSocket:
        def __init__(self) -> None:
            self.events: list[str | Exception] = [
                json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "audio",
                        "audio": audio,
                    }
                ),
                ConnectionClosed(None, None),
            ]

        async def recv(self) -> str:
            event = self.events.pop(0)
            if isinstance(event, Exception):
                raise event
            return event

    class Robot:
        def __init__(self) -> None:
            self.audio: list[np.ndarray] = []

        def set_conversation_state(self, state: str) -> None:
            pass

        def play_omni_audio(self, samples: np.ndarray) -> bool:
            self.audio.append(samples)
            return True

    client = OmniClient.__new__(OmniClient)
    robot = Robot()
    with pytest.raises(
        ConnectionError,
        match=r"no close frame.*last_event=response.output.delta:audio.*received_audio=1.00s",
    ):
        asyncio.run(client._receive_loop(WebSocket(), robot))
    assert len(robot.audio) == 1


def test_sender_holds_force_listen_while_barge_in_is_active() -> None:
    stop_event = threading.Event()

    class WebSocket:
        def __init__(self) -> None:
            self.message: dict[str, object] | None = None

        async def send(self, raw: str) -> None:
            self.message = json.loads(raw)
            stop_event.set()

    class Robot:
        def __init__(self) -> None:
            self.noted = False

        def next_audio_chunk(self, timeout: float) -> np.ndarray:
            return np.zeros(16_000, dtype=np.float32)

        def force_listen_active(self) -> bool:
            return True

        def note_force_listen_sent(self, response_id: str) -> None:
            self.noted = response_id == "test_resp_1"

    websocket = WebSocket()
    robot = Robot()
    client = OmniClient.__new__(OmniClient)
    client.config = SimpleNamespace(send_video=False)
    asyncio.run(client._send_loop(websocket, robot, stop_event, "test"))

    assert websocket.message is not None
    assert websocket.message["input"]["force_listen"] is True
    assert robot.noted
