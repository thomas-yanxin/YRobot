from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from websockets.asyncio.server import serve

from yrobot.config import Settings
from yrobot.protocol import Phase, ProtocolState, session_init
from yrobot.realtime import RealtimeClient, RealtimeError, _Lifecycle
from yrobot.state import InteractionPhase, TurnCoordinator


def make_client(
    coordinator: TurnCoordinator | None = None,
    *,
    latest_frame: Callable[[], bytes | None] = lambda: None,
    on_audio: Callable[[np.ndarray, int, str | None], None] = lambda *_: None,
    on_listen: Callable[[int], None] = lambda *_: None,
) -> RealtimeClient:
    return RealtimeClient(
        Settings(realtime_url="ws://brain.local/v1/realtime?mode=video"),
        coordinator or TurnCoordinator(),
        latest_frame=latest_frame,
        on_audio=on_audio,
        on_listen=on_listen,
        on_text=lambda *_: None,
        on_session=lambda *_: None,
    )


class FakeWebSocket:
    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        *,
        on_send: Callable[[], None] | None = None,
    ) -> None:
        self.events = [json.dumps(event) for event in events or []]
        self.sent: list[dict[str, Any]] = []
        self.on_send = on_send

    async def recv(self) -> str:
        await asyncio.sleep(0)
        return self.events.pop(0)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))
        if self.on_send:
            self.on_send()


def test_uplink_is_exact_and_latest_only() -> None:
    client = make_client()
    first = np.zeros(16_000, dtype=np.float32)
    second = np.full(16_000, 0.25, dtype=np.float32)

    client.submit_audio(first, sequence=1)
    client.submit_audio(second, sequence=2)

    retained = asyncio.run(client._next_input(threading.Event()))
    assert retained is not None
    assert retained.sequence == 2
    np.testing.assert_array_equal(
        np.frombuffer(retained.pcm_f32le, dtype="<f4"),
        second,
    )
    assert client.metrics.dropped_input_units == 1


def test_empty_uplink_poll_is_bounded_for_session_cancellation() -> None:
    client = make_client()
    started = time.monotonic()

    assert asyncio.run(client._next_input(threading.Event())) is None
    assert time.monotonic() - started < 0.2


def test_handshake_waits_for_queue_then_init_then_created() -> None:
    websocket = FakeWebSocket(
        [
            {"type": "session.queued", "position": 1},
            {"type": "session.queue_done"},
            {
                "type": "session.created",
                "session_id": "sess-1",
                "mode": "full_duplex",
                "metrics": {},
                "server_send_ts": 123.0,
            },
        ]
    )
    client = make_client()
    lifecycle = _Lifecycle()

    async def scenario() -> None:
        await client._wait_for_queue(
            websocket,  # type: ignore[arg-type]
            lifecycle,
            threading.Event(),
        )
        init = session_init("test", length_penalty=1.1)
        await lifecycle.send(websocket, init)  # type: ignore[arg-type]
        created = await client._wait_for_created(
            websocket,  # type: ignore[arg-type]
            lifecycle,
            threading.Event(),
            asyncio.get_running_loop().time() + 10,
        )
        assert created.session_id == "sess-1"

    asyncio.run(scenario())

    assert [event["type"] for event in websocket.sent] == ["session.init"]
    assert lifecycle.state.phase is Phase.CREATED


def test_session_created_wait_obeys_the_video_budget_when_server_is_silent() -> None:
    class SilentWebSocket:
        async def recv(self) -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    client = make_client()
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.INIT)

    async def scenario() -> None:
        with pytest.raises(RealtimeError, match="initialization exceeded"):
            await client._wait_for_created(
                SilentWebSocket(),  # type: ignore[arg-type]
                lifecycle,
                threading.Event(),
                asyncio.get_running_loop().time() + 0.05,
            )

    asyncio.run(asyncio.wait_for(scenario(), timeout=0.2))


def test_sender_carries_force_listen_and_latest_jpeg() -> None:
    stop = threading.Event()
    websocket = FakeWebSocket(on_send=stop.set)
    turns = TurnCoordinator()
    turns.new_session()
    turns.accept_audio("old")
    turns.interrupt()
    client = make_client(turns, latest_frame=lambda: b"\xff\xd8jpeg\xff\xd9")
    client.submit_audio(
        np.zeros(16_000, dtype=np.float32),
        captured_at=time.monotonic(),
        sequence=9,
    )
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")

    asyncio.run(
        client._sender(  # type: ignore[arg-type]
            websocket,
            lifecycle,
            stop,
        )
    )

    event = websocket.sent[0]
    assert event["type"] == "input.append"
    assert event["input"]["force_listen"] is True
    assert event["input"]["max_slice_nums"] == 1
    assert base64.b64decode(event["input"]["video_frames"][0]) == b"\xff\xd8jpeg\xff\xd9"
    assert len(base64.b64decode(event["input"]["audio"])) == 16_000 * 4
    assert turns.snapshot().force_listen_sent is True


def test_sender_never_compresses_two_one_second_units() -> None:
    stop = threading.Event()
    send_times: list[float] = []
    turns = TurnCoordinator()
    turns.new_session()
    turns.accept_audio("old")
    turns.interrupt()
    client = make_client(turns)

    def after_send() -> None:
        send_times.append(time.monotonic())
        if len(send_times) == 1:
            client.submit_audio(
                np.zeros(16_000, dtype=np.float32),
                captured_at=time.monotonic(),
                sequence=2,
            )
        else:
            stop.set()

    websocket = FakeWebSocket(on_send=after_send)
    client.submit_audio(
        np.zeros(16_000, dtype=np.float32),
        captured_at=time.monotonic(),
        sequence=1,
    )
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")

    asyncio.run(
        client._sender(  # type: ignore[arg-type]
            websocket,
            lifecycle,
            stop,
        )
    )

    assert len(send_times) == 2
    assert send_times[1] - send_times[0] >= 0.98
    assert all(event["input"]["force_listen"] for event in websocket.sent)
    assert client.metrics.capture_to_send.count == 2
    assert client.metrics.send_interval.minimum_ms >= 980


def test_sender_keeps_absolute_cadence_across_normal_write_latency() -> None:
    class DelayedWebSocket(FakeWebSocket):
        async def send(self, raw: str) -> None:
            await asyncio.sleep(0.2)
            await super().send(raw)

    stop = threading.Event()
    send_times: list[float] = []
    client = make_client()

    def after_send() -> None:
        send_times.append(time.monotonic())
        if len(send_times) == 1:
            client.submit_audio(
                np.zeros(16_000, dtype=np.float32),
                captured_at=time.monotonic(),
                sequence=2,
            )
        else:
            stop.set()

    websocket = DelayedWebSocket(on_send=after_send)
    client.submit_audio(
        np.zeros(16_000, dtype=np.float32),
        captured_at=time.monotonic(),
        sequence=1,
    )
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")

    asyncio.run(
        client._sender(  # type: ignore[arg-type]
            websocket,
            lifecycle,
            stop,
        )
    )

    assert 0.98 <= send_times[1] - send_times[0] < 1.1


def test_input_send_timeout_closes_and_reconnects_without_acknowledging_force() -> None:
    class StalledInputWebSocket:
        def __init__(self) -> None:
            self.close_called = False

        async def send(self, raw: str) -> None:
            assert json.loads(raw)["type"] == "input.append"
            await asyncio.Event().wait()

        async def close(self, *, code: int, reason: str) -> None:
            del code, reason
            self.close_called = True

    async def scenario() -> tuple[StalledInputWebSocket, TurnCoordinator]:
        turns = TurnCoordinator()
        turns.new_session()
        turns.accept_audio("old")
        turns.interrupt()
        client = make_client(turns)
        client._input_send_timeout = 0.05
        client.submit_audio(
            np.zeros(16_000, dtype=np.float32),
            captured_at=time.monotonic(),
        )
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        websocket = StalledInputWebSocket()
        with pytest.raises(RealtimeError, match="input.append send timed out"):
            await client._sender(  # type: ignore[arg-type]
                websocket,
                lifecycle,
                threading.Event(),
            )
        return websocket, turns

    websocket, turns = asyncio.run(asyncio.wait_for(scenario(), timeout=0.3))
    assert websocket.close_called
    assert turns.snapshot().force_listen_sent is False


def test_stream_stop_cleans_up_a_shielded_stalled_input_send() -> None:
    class StalledWebSocket:
        def __init__(self) -> None:
            self.input_started = asyncio.Event()

        async def send(self, raw: str) -> None:
            assert json.loads(raw)["type"] == "input.append"
            self.input_started.set()
            await asyncio.Event().wait()

        async def recv(self) -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def close(self, *, code: int, reason: str) -> None:
            del code, reason

    async def scenario() -> list[str]:
        client = make_client()
        client._input_send_timeout = 10.0
        client.submit_audio(
            np.zeros(16_000, dtype=np.float32),
            captured_at=time.monotonic(),
        )
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        stop = threading.Event()
        websocket = StalledWebSocket()
        streaming = asyncio.create_task(
            client._stream(
                websocket,  # type: ignore[arg-type]
                lifecycle,
                stop,
                asyncio.get_running_loop().time() + 10,
            )
        )
        await websocket.input_started.wait()
        stop.set()
        assert await asyncio.wait_for(streaming, timeout=1.5)
        await asyncio.sleep(0)
        return [
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]

    names = asyncio.run(scenario())
    assert "minicpmo-input-send" not in names


def test_stop_before_first_input_still_sends_session_close() -> None:
    class ClosingWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []
            self.events: asyncio.Queue[str] = asyncio.Queue()

        async def send(self, raw: str) -> None:
            event = json.loads(raw)
            self.sent.append(event)
            if event["type"] == "session.close":
                await self.events.put(
                    json.dumps(
                        {
                            "type": "session.closed",
                            "session_id": "sess-1",
                            "reason": "user_stop",
                        }
                    )
                )

        async def recv(self) -> str:
            return await self.events.get()

    client = make_client()
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
    stop = threading.Event()
    stop.set()

    async def scenario() -> tuple[bool, list[dict[str, Any]]]:
        websocket = ClosingWebSocket()
        result = await client._stream(
            websocket,  # type: ignore[arg-type]
            lifecycle,
            stop,
            asyncio.get_running_loop().time() + 10,
        )
        return result, websocket.sent

    stopped, sent = asyncio.run(scenario())

    assert stopped is True
    assert [event["type"] for event in sent] == ["session.close"]


def test_stop_lets_an_inflight_input_send_finish_before_session_close() -> None:
    class SlowClosingWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []
            self.events: asyncio.Queue[str] = asyncio.Queue()
            self.input_started = asyncio.Event()
            self.release_input = asyncio.Event()
            self.input_cancelled = False

        async def send(self, raw: str) -> None:
            event = json.loads(raw)
            self.sent.append(event)
            if event["type"] == "input.append":
                self.input_started.set()
                try:
                    await self.release_input.wait()
                except asyncio.CancelledError:
                    self.input_cancelled = True
                    raise
            elif event["type"] == "session.close":
                await self.events.put(
                    json.dumps(
                        {
                            "type": "session.closed",
                            "session_id": "sess-1",
                            "reason": event["reason"],
                        }
                    )
                )

        async def recv(self) -> str:
            return await self.events.get()

    async def scenario() -> tuple[bool, SlowClosingWebSocket]:
        client = make_client()
        client.submit_audio(
            np.zeros(16_000, dtype=np.float32),
            captured_at=time.monotonic(),
        )
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        stop = threading.Event()
        websocket = SlowClosingWebSocket()
        streaming = asyncio.create_task(
            client._stream(
                websocket,  # type: ignore[arg-type]
                lifecycle,
                stop,
                asyncio.get_running_loop().time() + 10,
            )
        )
        await websocket.input_started.wait()
        stop.set()
        await asyncio.sleep(0.06)
        websocket.release_input.set()
        return await streaming, websocket

    stopped, websocket = asyncio.run(scenario())

    assert stopped is True
    assert websocket.input_cancelled is False
    assert [event["type"] for event in websocket.sent] == [
        "input.append",
        "session.close",
    ]


def test_session_close_send_is_bounded_when_the_socket_stalls() -> None:
    class StalledCloseWebSocket:
        def __init__(self) -> None:
            self.close_called = False

        async def send(self, raw: str) -> None:
            assert json.loads(raw)["type"] == "session.close"
            await asyncio.Event().wait()

        async def recv(self) -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def close(self, *, code: int, reason: str) -> None:
            del code, reason
            self.close_called = True

    async def scenario() -> StalledCloseWebSocket:
        client = make_client()
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        stop = threading.Event()
        stop.set()
        websocket = StalledCloseWebSocket()
        assert await client._stream(
            websocket,  # type: ignore[arg-type]
            lifecycle,
            stop,
            asyncio.get_running_loop().time() + 10,
        )
        return websocket

    websocket = asyncio.run(asyncio.wait_for(scenario(), timeout=1.5))
    assert websocket.close_called


def test_interrupted_audio_is_dropped_until_listen_boundary() -> None:
    pcm = np.linspace(-0.1, 0.1, 240, dtype="<f4").tobytes()
    encoded = base64.b64encode(pcm).decode("ascii")
    websocket = FakeWebSocket(
        [
            {
                "type": "response.output.delta",
                "kind": "audio",
                "session_id": "sess-1",
                "response_id": "old",
                "audio": encoded,
                "metrics": {},
            },
            {
                "type": "response.output.delta",
                "kind": "listen",
                "session_id": "sess-1",
                "response_id": "old",
                "metrics": {},
            },
            {
                "type": "response.output.delta",
                "kind": "audio",
                "session_id": "sess-1",
                "response_id": "new",
                "audio": encoded,
                "metrics": {},
            },
            {
                "type": "session.closed",
                "session_id": "sess-1",
                "reason": "test",
            },
        ]
    )
    turns = TurnCoordinator()
    turns.new_session()
    turns.accept_audio("old")
    interrupted_epoch = turns.interrupt()
    assert interrupted_epoch is not None
    assert turns.force_listen_sent(interrupted_epoch)
    received: list[tuple[int, str | None]] = []
    listens: list[int] = []
    client = make_client(
        turns,
        on_audio=lambda _samples, epoch, response: received.append((epoch, response)),
        on_listen=listens.append,
    )
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.STREAMING, "sess-1")
    closed = asyncio.Event()

    asyncio.run(
        client._receiver(  # type: ignore[arg-type]
            websocket,
            lifecycle,
            closed,
        )
    )

    assert received == [(interrupted_epoch, "new")]
    assert listens == [interrupted_epoch]
    assert turns.snapshot().phase is InteractionPhase.SPEAKING
    assert lifecycle.state.phase is Phase.CLOSED


def test_real_websockets15_full_lifecycle() -> None:
    async def scenario() -> tuple[list[str], list[str]]:
        server_events: list[str] = []
        received_audio: list[str] = []
        stop = threading.Event()

        async def handler(websocket: Any) -> None:
            await websocket.send('{"type":"session.queue_done"}')
            init = json.loads(await websocket.recv())
            server_events.append(init["type"])
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "session_id": "sess-local",
                        "mode": "full_duplex",
                        "metrics": {},
                        "server_send_ts": time.time(),
                    }
                )
            )
            append = json.loads(await websocket.recv())
            server_events.append(append["type"])
            audio = base64.b64encode(np.zeros(240, dtype="<f4").tobytes()).decode("ascii")
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "audio",
                        "session_id": "sess-local",
                        "response_id": "resp-local",
                        "audio": audio,
                        "metrics": {},
                    }
                )
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "listen",
                        "session_id": "sess-local",
                        "response_id": "resp-local",
                        "metrics": {},
                    }
                )
            )
            close = json.loads(await websocket.recv())
            server_events.append(close["type"])
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.closed",
                        "session_id": "sess-local",
                        "reason": close["reason"],
                    }
                )
            )

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            turns = TurnCoordinator()
            client = RealtimeClient(
                Settings(realtime_url=(f"ws://127.0.0.1:{port}/v1/realtime?mode=video")),
                turns,
                latest_frame=lambda: None,
                on_audio=lambda _samples, _epoch, response: received_audio.append(str(response)),
                on_listen=lambda _epoch: stop.set(),
                on_text=lambda *_: None,
                on_session=lambda *_: None,
            )
            client.submit_audio(
                np.zeros(16_000, dtype=np.float32),
                captured_at=time.monotonic(),
            )
            await asyncio.wait_for(client.run(stop), timeout=3.0)
        return server_events, received_audio

    events, audio = asyncio.run(scenario())

    assert events == ["session.init", "input.append", "session.close"]
    assert audio == ["resp-local"]
