import asyncio
import base64
import json
import logging
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
        "payload": {
            "mode": "full_duplex",
            "use_tts": True,
            "system_prompt": "hello",
            "config": {"length_penalty": 1.1},
        },
    }

    append = build_input_append(np.zeros(16_000, dtype=np.float32), b"jpeg")
    assert append["type"] == "input.append"
    assert set(append["input"]) == {
        "audio",
        "video_frames",
        "max_slice_nums",
    }
    assert append["input"]["max_slice_nums"] == 1

    forced = build_input_append(
        np.zeros(16_000, dtype=np.float32),
        None,
        force_listen=True,
    )
    assert forced["input"]["force_listen"] is True


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

        def play_omni_audio(self, samples: np.ndarray, response_id: str) -> bool:
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


def test_tts_gap_survives_response_done_and_response_id_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    audio = encode_pcm(np.zeros(240, dtype=np.float32))

    class WebSocket:
        def __init__(self) -> None:
            self.index = 0

        async def recv(self) -> str:
            self.index += 1
            if self.index == 1:
                return json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "audio",
                        "response_id": "r1",
                        "audio": audio,
                    }
                )
            if self.index == 2:
                return json.dumps({"type": "response.done", "response_id": "r1"})
            if self.index == 3:
                await asyncio.sleep(0.08)
                return json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "audio",
                        "response_id": "r2",
                        "audio": audio,
                    }
                )
            raise ConnectionClosed(None, None)

    class Robot:
        def __init__(self) -> None:
            self.supply_gaps: list[float] = []

        def set_conversation_state(self, state: str) -> None:
            pass

        def play_omni_audio(self, samples: np.ndarray, response_id: str) -> bool:
            return True

        def handle_omni_done(self, response_id: str) -> None:
            pass

        def note_tts_supply_gap(self, gap_seconds: float) -> None:
            self.supply_gaps.append(gap_seconds)

    client = OmniClient.__new__(OmniClient)
    robot = Robot()
    with caplog.at_level(logging.WARNING, logger="yrobot.omni"):
        with pytest.raises(ConnectionError):
            asyncio.run(client._receive_loop(WebSocket(), robot))

    assert "TTS supply gap for r2" in caplog.text
    assert robot.supply_gaps and robot.supply_gaps[0] > 0.05


def test_sender_sends_silent_force_control_before_preserved_microphone() -> None:
    stop_event = threading.Event()
    microphone = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)

    class Robot:
        def __init__(self) -> None:
            self.force_listen = True

        def next_audio_chunk(self, timeout: float) -> np.ndarray:
            del timeout
            return microphone

        def get_frame_jpeg(self) -> None:
            return None

        def force_listen_active(self) -> bool:
            return self.force_listen

    robot = Robot()

    class WebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))
            if len(self.messages) == 1:
                robot.force_listen = False
            else:
                stop_event.set()

    websocket = WebSocket()
    client = OmniClient.__new__(OmniClient)
    client.config = SimpleNamespace(send_video=False)

    asyncio.run(client._send_loop(websocket, robot, stop_event))

    assert len(websocket.messages) == 2
    control, speech = websocket.messages
    assert control["type"] == "input.append"
    assert control["input"]["force_listen"] is True
    np.testing.assert_array_equal(decode_pcm(control["input"]["audio"]), np.zeros(16_000))
    assert "force_listen" not in speech["input"]
    np.testing.assert_array_equal(decode_pcm(speech["input"]["audio"]), microphone)


def test_video_yields_to_audio_cadence_after_a_slow_send() -> None:
    stop_event = threading.Event()

    class Robot:
        def next_audio_chunk(self, timeout: float) -> np.ndarray:
            del timeout
            return np.zeros(16_000, dtype=np.float32)

        def get_frame_jpeg(self) -> bytes:
            return b"jpeg"

        def force_listen_active(self) -> bool:
            return False

    class WebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))
            if len(self.messages) == 1:
                await asyncio.sleep(0.3)
            else:
                stop_event.set()

    websocket = WebSocket()
    client = OmniClient.__new__(OmniClient)
    client.config = SimpleNamespace(send_video=True)

    asyncio.run(client._send_loop(websocket, Robot(), stop_event))

    congested, recovering = websocket.messages
    assert "video_frames" in congested["input"]
    assert "video_frames" not in recovering["input"]


def test_normal_websocket_backpressure_does_not_flood_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stop_event = threading.Event()

    class WebSocket:
        def __init__(self) -> None:
            self.sends = 0

        async def send(self, raw: str) -> None:
            del raw
            await asyncio.sleep(0.06)
            self.sends += 1
            if self.sends == 2:
                stop_event.set()

    class Robot:
        def next_audio_chunk(self, timeout: float) -> np.ndarray:
            del timeout
            return np.zeros(16_000, dtype=np.float32)

        def force_listen_active(self) -> bool:
            return False

    websocket = WebSocket()
    client = OmniClient.__new__(OmniClient)
    client.config = SimpleNamespace(send_video=False)
    with caplog.at_level(logging.WARNING, logger="yrobot.omni"):
        asyncio.run(client._send_loop(websocket, Robot(), stop_event))

    assert websocket.sends == 2
    assert "Slow Omni input stage" not in caplog.text
