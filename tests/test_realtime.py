import asyncio
import base64
import io
import json
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from yrobot.config import AUDIO_UNIT_SAMPLES, OUTPUT_SAMPLE_RATE, Config
from yrobot.realtime import (
    RealtimeClient,
    RealtimeProtocolError,
    SessionOutcome,
    build_input_append,
    build_session_init,
    decode_output_audio,
    encode_input_audio,
    gateway_default_ref_audio_url,
    load_gateway_ref_audio,
)


def make_config(url: str, **overrides: Any) -> Config:
    values: dict[str, Any] = {
        "realtime_url": url,
        "tls_verify": True,
        "send_video": False,
        "system_prompt": "test prompt",
        "enable_tts": False,
        "handshake_timeout": 1.0,
        "session_rollover": 10.0,
    }
    values.update(overrides)
    return Config(**values)


class FakePort:
    def __init__(
        self,
        *,
        audio: np.ndarray | None = None,
        force_listen: bool = False,
        frame_delay: float = 0.0,
    ) -> None:
        self.audio = audio
        self.force_listen = force_listen
        self.frame_delay = frame_delay
        self.audio_sent = False
        self.audio_deltas: list[tuple[np.ndarray, str, Mapping[str, Any]]] = []
        self.listens: list[tuple[str, Mapping[str, Any]]] = []
        self.texts: list[tuple[str, str]] = []
        self.ready_count = 0
        self.invalidations: list[str] = []
        self.rollover_ready = True

    def next_audio_unit(self, timeout: float) -> tuple[np.ndarray, bool] | None:
        if self.audio is not None and not self.audio_sent:
            self.audio_sent = True
            return self.audio, self.force_listen
        time.sleep(min(timeout, 0.005))
        return None

    def latest_frame_jpeg(self) -> bytes:
        time.sleep(self.frame_delay)
        return b"jpeg"

    def handle_audio_delta(
        self,
        samples: np.ndarray,
        response_id: str,
        metrics: Mapping[str, Any],
    ) -> None:
        self.audio_deltas.append((samples, response_id, metrics))

    def handle_listen(
        self,
        response_id: str,
        metrics: Mapping[str, Any],
    ) -> None:
        self.listens.append((response_id, metrics))

    def handle_text(self, text: str, response_id: str) -> None:
        self.texts.append((text, response_id))

    def handle_session_ready(self) -> None:
        self.ready_count += 1

    def invalidate_session(self, reason: str) -> None:
        self.invalidations.append(reason)

    def ready_for_rollover(self) -> bool:
        return self.rollover_ready


def test_audio_codec_is_little_endian_float32_with_fixed_input_unit() -> None:
    samples = np.linspace(-1.0, 1.0, AUDIO_UNIT_SAMPLES, dtype=np.float32)
    encoded = encode_input_audio(samples)
    assert base64.b64decode(encoded) == samples.astype("<f4").tobytes()

    output = np.linspace(-0.5, 0.5, OUTPUT_SAMPLE_RATE // 10, dtype=np.float32)
    decoded = decode_output_audio(base64.b64encode(output.astype("<f4").tobytes()).decode("ascii"))
    np.testing.assert_array_equal(decoded, output)
    assert OUTPUT_SAMPLE_RATE == 24_000


def test_audio_codec_rejects_bad_units_and_payloads() -> None:
    with pytest.raises(ValueError, match="exactly 16000"):
        encode_input_audio(np.zeros(AUDIO_UNIT_SAMPLES - 1, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        encode_input_audio(np.full(AUDIO_UNIT_SAMPLES, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="base64"):
        decode_output_audio("not base64!")
    with pytest.raises(ValueError, match="float32"):
        decode_output_audio(base64.b64encode(b"abc").decode("ascii"))


def test_gateway_reference_voice_is_loaded_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.linspace(-0.2, 0.2, 160, dtype="<f4")
    voice = base64.b64encode(samples.tobytes()).decode("ascii")
    body = json.dumps({"base64": voice, "sample_rate": 16_000}).encode()
    captured: dict[str, Any] = {}

    class Response(io.BytesIO):
        def __enter__(self) -> io.BytesIO:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def fake_urlopen(request: Any, **options: Any) -> Response:
        captured["url"] = request.full_url
        captured.update(options)
        return Response(body)

    monkeypatch.setattr("yrobot.realtime.urlopen", fake_urlopen)
    config = make_config(
        "wss://brain.local:8006/v1/realtime?mode=video",
        enable_tts=True,
        tls_verify=False,
    )

    assert gateway_default_ref_audio_url(config.realtime_url) == (
        "https://brain.local:8006/api/default_ref_audio"
    )
    assert load_gateway_ref_audio(config) == voice
    assert captured["url"] == "https://brain.local:8006/api/default_ref_audio"
    assert captured["timeout"] == 20.0
    assert captured["context"].verify_mode == 0


def test_protocol_messages_match_video_duplex_gateway() -> None:
    assert build_session_init("hello", ref_audio_base64="voice") == {
        "type": "session.init",
        "payload": {
            "system_prompt": "hello",
            "config": {
                "generate_audio": True,
                "length_penalty": 1.1,
                "force_listen_count": 1,
            },
            "use_tts": True,
            "voice": {
                "ref_audio_base64": "voice",
                "tts_ref_audio_base64": "voice",
            },
        },
    }
    assert build_session_init("hello", enable_tts=False) == {
        "type": "session.init",
        "payload": {
            "system_prompt": "hello",
            "config": {
                "generate_audio": False,
                "length_penalty": 1.1,
                "force_listen_count": 1,
            },
            "use_tts": False,
        },
    }
    with pytest.raises(ValueError, match="ref_audio_base64"):
        build_session_init("hello")

    audio = np.linspace(-0.25, 0.25, AUDIO_UNIT_SAMPLES, dtype=np.float32)
    append = build_input_append(audio, b"jpeg", force_listen=True)
    assert append["type"] == "input.append"
    assert append["input"]["force_listen"] is True
    assert append["input"]["max_slice_nums"] == 1
    assert base64.b64decode(append["input"]["video_frames"][0]) == b"jpeg"
    np.testing.assert_array_equal(
        np.frombuffer(base64.b64decode(append["input"]["audio"]), dtype="<f4"),
        audio,
    )

    audio_only = build_input_append(audio, None, force_listen=False)
    assert audio_only["input"]["force_listen"] is False
    assert "video_frames" not in audio_only["input"]


def test_handshake_waits_for_queue_done_before_init_and_then_created() -> None:
    class WebSocket:
        def __init__(self) -> None:
            self.events = [
                {"type": "session.queued", "position": 1},
                {"type": "session.queue_update", "position": 0},
                {"type": "session.queue_done"},
                {"type": "session.created", "session_id": "session-1"},
            ]
            self.sent: list[dict[str, Any]] = []
            self.index = 0

        async def recv(self) -> str:
            event = self.events[self.index]
            self.index += 1
            if event["type"] == "session.queue_done":
                assert self.sent == []
            if event["type"] == "session.created":
                assert [message["type"] for message in self.sent] == ["session.init"]
            return json.dumps(event)

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    async def scenario() -> None:
        websocket = WebSocket()
        client = RealtimeClient(make_config("ws://unused.local"))
        session_id, assigned_at = await client._handshake(websocket)
        assert session_id == "session-1"
        assert assigned_at > 0
        assert websocket.sent[0] == build_session_init("test prompt", enable_tts=False)

    asyncio.run(scenario())


def test_handshake_rejects_created_before_queue_done() -> None:
    class WebSocket:
        async def recv(self) -> str:
            return json.dumps({"type": "session.created", "session_id": "bad"})

    async def scenario() -> None:
        client = RealtimeClient(make_config("ws://unused.local"))
        with pytest.raises(RealtimeProtocolError, match="before session.queue_done"):
            await client._handshake(WebSocket())

    asyncio.run(scenario())


def test_receiver_dispatches_all_duplex_kinds_and_ignores_response_done() -> None:
    output = np.linspace(-0.2, 0.2, 240, dtype=np.float32)
    encoded = base64.b64encode(output.astype("<f4").tobytes()).decode("ascii")

    class WebSocket:
        def __init__(self) -> None:
            self.events: list[str | Exception] = [
                json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "audio",
                        "audio": encoded,
                        "response_id": "r1",
                        "metrics": {"latency_ms": 12},
                    }
                ),
                json.dumps({"type": "response.done", "response_id": "r1"}),
                json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "text",
                        "text": "你好",
                        "response_id": "r1",
                    }
                ),
                json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "listen",
                        "response_id": "r2",
                        "metrics": {"kv_cache_length": 32},
                    }
                ),
                ConnectionClosed(None, None),
            ]

        async def recv(self) -> str:
            event = self.events.pop(0)
            if isinstance(event, Exception):
                raise event
            return event

    async def scenario() -> None:
        port = FakePort()
        client = RealtimeClient(make_config("ws://unused.local"))
        with pytest.raises(ConnectionError, match="transport closed"):
            await client._receive_loop(WebSocket(), port)
        assert len(port.audio_deltas) == 1
        np.testing.assert_allclose(port.audio_deltas[0][0], output)
        assert port.audio_deltas[0][1:] == ("r1", {"latency_ms": 12})
        assert port.texts == [("你好", "r1")]
        assert port.listens == [("r2", {"kv_cache_length": 32})]

    asyncio.run(scenario())


def test_receiver_waits_for_transport_cleanup_after_session_closed() -> None:
    class WebSocket:
        def __init__(self) -> None:
            self.events: list[str | Exception] = [
                json.dumps({"type": "session.closed", "reason": "client_rollover"}),
                ConnectionClosed(None, None),
            ]
            self.recv_count = 0

        async def recv(self) -> str:
            self.recv_count += 1
            event = self.events.pop(0)
            if isinstance(event, Exception):
                raise event
            return event

    async def scenario() -> None:
        websocket = WebSocket()
        client = RealtimeClient(make_config("ws://unused.local"))
        await client._receive_loop(websocket, FakePort())
        assert websocket.recv_count == 2

    asyncio.run(scenario())


def test_session_is_full_duplex_with_latest_only_video() -> None:
    asyncio.run(_full_duplex_session_scenario())


async def _full_duplex_session_scenario() -> None:
    stop_event = threading.Event()
    observed: list[dict[str, Any]] = []
    output = np.linspace(-0.2, 0.2, 240, dtype=np.float32)

    async def handler(websocket: Any) -> None:
        await websocket.send(json.dumps({"type": "session.queued", "position": 1}))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(websocket.recv(), timeout=0.03)

        await websocket.send(json.dumps({"type": "session.queue_done"}))
        init = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.5))
        observed.append(init)
        await websocket.send(json.dumps({"type": "session.created", "session_id": "session-1"}))

        append = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.25))
        observed.append(append)
        await websocket.send(
            json.dumps(
                {
                    "type": "response.output.delta",
                    "kind": "audio",
                    "audio": base64.b64encode(output.astype("<f4").tobytes()).decode("ascii"),
                    "response_id": "r1",
                    "metrics": {"latency_ms": 10},
                }
            )
        )
        await websocket.send(json.dumps({"type": "response.done", "response_id": "r1"}))
        await websocket.send(
            json.dumps(
                {
                    "type": "response.output.delta",
                    "kind": "text",
                    "text": "你好",
                    "response_id": "r1",
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "response.output.delta",
                    "kind": "listen",
                    "response_id": "r2",
                    "metrics": {"kv_cache_length": 64},
                }
            )
        )
        await asyncio.sleep(0.02)
        stop_event.set()
        close = json.loads(await asyncio.wait_for(websocket.recv(), timeout=1.0))
        observed.append(close)
        await websocket.send(json.dumps({"type": "session.closed", "reason": "user_stop"}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        server_port = server.sockets[0].getsockname()[1]
        config = make_config(
            f"ws://127.0.0.1:{server_port}/v1/realtime?mode=video",
            send_video=True,
        )
        microphone = np.linspace(-0.5, 0.5, AUDIO_UNIT_SAMPLES, dtype=np.float32)
        port = FakePort(audio=microphone, force_listen=True)
        outcome = await RealtimeClient(config).run_session(port, stop_event)

    assert outcome is SessionOutcome.STOP
    assert observed[0] == build_session_init("test prompt", enable_tts=False)
    assert observed[1]["type"] == "input.append"
    assert observed[1]["input"]["force_listen"] is True
    assert observed[1]["input"]["max_slice_nums"] == 1
    assert base64.b64decode(observed[1]["input"]["video_frames"][0]) == b"jpeg"
    np.testing.assert_array_equal(
        np.frombuffer(base64.b64decode(observed[1]["input"]["audio"]), dtype="<f4"),
        microphone,
    )
    assert observed[2] == {"type": "session.close", "reason": "user_stop"}
    assert port.texts == [("你好", "r1")]
    assert port.listens == [("r2", {"kv_cache_length": 64})]
    assert port.audio_deltas[0][1:] == ("r1", {"latency_ms": 10})
    np.testing.assert_allclose(port.audio_deltas[0][0], output)
    assert port.ready_count == 1
    assert port.invalidations == ["connecting", "stop"]


def test_session_rolls_proactively_and_invalidates_old_playback() -> None:
    asyncio.run(_rollover_scenario())


async def _rollover_scenario() -> None:
    observed: list[dict[str, Any]] = []
    elapsed = 0.0

    async def handler(websocket: Any) -> None:
        await websocket.send(json.dumps({"type": "session.queue_done"}))
        observed.append(json.loads(await websocket.recv()))
        await websocket.send(json.dumps({"type": "session.created", "session_id": "roll"}))
        observed.append(json.loads(await asyncio.wait_for(websocket.recv(), timeout=1.0)))
        await websocket.send(json.dumps({"type": "session.closed", "reason": "client_rollover"}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        server_port = server.sockets[0].getsockname()[1]
        config = make_config(
            f"ws://127.0.0.1:{server_port}/v1/realtime?mode=video",
            session_rollover=0.05,
        )
        port = FakePort()
        port.rollover_ready = False
        loop = asyncio.get_running_loop()
        loop.call_later(0.12, setattr, port, "rollover_ready", True)
        started = loop.time()
        outcome = await RealtimeClient(config).run_session(port, threading.Event())
        elapsed = loop.time() - started

    assert outcome is SessionOutcome.ROLLOVER
    assert elapsed >= 0.10
    assert observed[0]["type"] == "session.init"
    assert observed[1] == {"type": "session.close", "reason": "client_rollover"}
    assert port.ready_count == 1
    assert port.invalidations == ["connecting", "rollover"]


def test_stop_cancels_a_queued_handshake_without_waiting_for_server() -> None:
    asyncio.run(_stop_during_queue_scenario())


async def _stop_during_queue_scenario() -> None:
    stop_event = threading.Event()
    close_messages: list[dict[str, Any]] = []

    async def handler(websocket: Any) -> None:
        await websocket.send(json.dumps({"type": "session.queued", "position": 10}))
        close_messages.append(json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.5)))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        server_port = server.sockets[0].getsockname()[1]
        config = make_config(
            f"ws://127.0.0.1:{server_port}/v1/realtime?mode=video",
            handshake_timeout=5.0,
        )
        port = FakePort()
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, stop_event.set)
        started = loop.time()
        outcome = await RealtimeClient(config).run_session(port, stop_event)
        elapsed = loop.time() - started

    assert outcome is SessionOutcome.STOP
    assert elapsed < 0.3
    assert close_messages == [{"type": "session.close", "reason": "user_stop"}]
    assert port.invalidations == ["connecting", "stop"]


def test_stop_during_reference_voice_fetch_is_immediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_fetch = threading.Event()

    def blocked_fetch(_config: Config) -> str:
        release_fetch.wait(1.0)
        return "unused"

    monkeypatch.setattr("yrobot.realtime.load_gateway_ref_audio", blocked_fetch)

    async def scenario() -> tuple[SessionOutcome, float, FakePort]:
        stop_event = threading.Event()
        port = FakePort()
        config = make_config(
            "ws://unused.local/v1/realtime?mode=video",
            enable_tts=True,
        )
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, stop_event.set)
        started = loop.time()
        outcome = await RealtimeClient(config).run_session(port, stop_event)
        return outcome, loop.time() - started, port

    try:
        outcome, elapsed, port = asyncio.run(scenario())
    finally:
        release_fetch.set()

    assert outcome is SessionOutcome.STOP
    assert elapsed < 0.3
    assert port.invalidations == ["connecting", "stop"]


def test_sender_is_quiescent_before_close_and_playback_invalidates_first() -> None:
    asyncio.run(_close_order_scenario())


async def _close_order_scenario() -> None:
    stop_event = threading.Event()
    observed: list[dict[str, Any]] = []
    no_message_after_close = False
    close_seen_at = 0.0

    class StreamingPort(FakePort):
        def __init__(self) -> None:
            super().__init__()
            self.invalidation_times: dict[str, float] = {}

        def next_audio_unit(self, timeout: float) -> tuple[np.ndarray, bool] | None:
            time.sleep(min(timeout, 0.01))
            return np.zeros(AUDIO_UNIT_SAMPLES, dtype=np.float32), False

        def invalidate_session(self, reason: str) -> None:
            super().invalidate_session(reason)
            self.invalidation_times[reason] = time.monotonic()

    async def handler(websocket: Any) -> None:
        nonlocal no_message_after_close, close_seen_at
        await websocket.send(json.dumps({"type": "session.queue_done"}))
        await websocket.recv()
        await websocket.send(json.dumps({"type": "session.created", "session_id": "close-order"}))

        first = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.5))
        assert first["type"] == "input.append"
        observed.append(first)
        stop_event.set()

        while True:
            message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=1.0))
            observed.append(message)
            if message["type"] == "session.close":
                close_seen_at = time.monotonic()
                break
        try:
            await asyncio.wait_for(websocket.recv(), timeout=0.12)
        except TimeoutError:
            no_message_after_close = True
        await websocket.send(json.dumps({"type": "session.closed", "reason": "user_stop"}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        server_port = server.sockets[0].getsockname()[1]
        config = make_config(
            f"ws://127.0.0.1:{server_port}/v1/realtime?mode=video",
            close_ack_timeout=1.0,
        )
        port = StreamingPort()
        outcome = await RealtimeClient(config).run_session(port, stop_event)

    assert outcome is SessionOutcome.STOP
    assert no_message_after_close
    assert observed[-1] == {"type": "session.close", "reason": "user_stop"}
    assert port.invalidation_times["stop"] <= close_seen_at


def test_failed_session_invalidates_before_reconnect() -> None:
    asyncio.run(_failed_session_scenario())


async def _failed_session_scenario() -> None:
    async def handler(websocket: Any) -> None:
        await websocket.send(json.dumps({"type": "session.queue_done"}))
        await websocket.recv()
        await websocket.send(json.dumps({"type": "error", "error": {"message": "worker failed"}}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        server_port = server.sockets[0].getsockname()[1]
        config = make_config(f"ws://127.0.0.1:{server_port}/v1/realtime?mode=video")
        port = FakePort()
        with pytest.raises(RealtimeProtocolError, match="worker failed"):
            await RealtimeClient(config).run_session(port, threading.Event())

    assert port.invalidations == ["connecting", "reconnect"]


def test_reconnect_uses_bounded_exponential_backoff() -> None:
    class FailingClient(RealtimeClient):
        def __init__(self, config: Config) -> None:
            super().__init__(config)
            self.delays: list[float] = []

        async def run_session(
            self,
            port: FakePort,
            stop_event: threading.Event,
        ) -> SessionOutcome:
            del port, stop_event
            raise ConnectionError("offline")

        async def _wait_or_stop(
            self,
            stop_event: threading.Event,
            delay: float,
        ) -> None:
            self.delays.append(delay)
            if len(self.delays) == 4:
                stop_event.set()

    async def scenario() -> None:
        config = make_config(
            "ws://unused.local",
            reconnect_initial_delay=0.1,
            reconnect_max_delay=0.25,
        )
        client = FailingClient(config)
        await client.run(FakePort(), threading.Event())
        expected = [0.1, 0.2, 0.25, 0.25]
        assert len(client.delays) == len(expected)
        for delay, maximum in zip(client.delays, expected, strict=True):
            assert maximum * 0.75 <= delay <= maximum

    asyncio.run(scenario())
