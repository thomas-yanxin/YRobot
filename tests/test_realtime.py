from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from websockets.asyncio.server import serve

from yrobot.config import Settings
from yrobot.protocol import Phase, ProtocolState, input_append, session_close, session_init
from yrobot.realtime import RealtimeClient, RealtimeError, _Lifecycle
from yrobot.state import InteractionPhase, TurnCoordinator


def make_client(
    coordinator: TurnCoordinator | None = None,
    *,
    latest_frame: Callable[[], bytes | None] = lambda: None,
    on_audio: Callable[[np.ndarray, int, str | None, float], None] = lambda *_: None,
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


def test_slow_uplink_write_does_not_block_downlink_lifecycle() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingWebSocket:
            async def send(self, _raw: str) -> None:
                started.set()
                await release.wait()

        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.STREAMING, "sess-1")
        sending = asyncio.create_task(
            lifecycle.send(
                BlockingWebSocket(),  # type: ignore[arg-type]
                input_append(np.zeros(16_000, dtype="<f4").tobytes()),
            )
        )
        await started.wait()
        raw = json.dumps(
            {
                "type": "response.output.delta",
                "kind": "text",
                "session_id": "sess-1",
                "response_id": "resp-1",
                "text": "still receiving",
                "metrics": {},
            }
        )
        received = await asyncio.wait_for(lifecycle.receive(raw), timeout=0.05)
        assert received.text == "still receiving"  # type: ignore[union-attr]
        release.set()
        await sending

    asyncio.run(scenario())


def test_session_close_is_skipped_when_server_closes_during_send_race() -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        async with lifecycle.send_lock:
            closing = asyncio.create_task(
                lifecycle.try_send_close(  # type: ignore[arg-type]
                    websocket,
                    session_close("test"),
                )
            )
            await asyncio.sleep(0)
            await lifecycle.receive(
                json.dumps(
                    {
                        "type": "session.closed",
                        "session_id": "sess-1",
                        "reason": "server_stop",
                    }
                )
            )
        assert await closing is False
        assert websocket.sent == []

    asyncio.run(scenario())


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


def test_sender_sends_each_fresh_unit_without_a_second_pacing_delay() -> None:
    async def scenario() -> tuple[list[float], float, RealtimeClient]:
        stop = threading.Event()
        send_times: list[float] = []
        client = make_client()

        def after_send() -> None:
            send_times.append(time.monotonic())
            if len(send_times) == 2:
                stop.set()

        websocket = FakeWebSocket(on_send=after_send)
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        client.submit_audio(
            np.zeros(16_000, dtype=np.float32),
            captured_at=time.monotonic(),
            sequence=1,
        )
        sender = asyncio.create_task(
            client._sender(  # type: ignore[arg-type]
                websocket,
                lifecycle,
                stop,
            )
        )
        while client.metrics.input_units < 1:
            await asyncio.sleep(0)

        # Simulate the next hardware-clocked one-second unit without making the
        # test sleep for a full second.
        client._last_wire_send_at = time.monotonic() - 1.0
        client._last_sent_capture_at = time.monotonic() - 1.0
        submitted_at = time.monotonic()
        client.submit_audio(
            np.full(16_000, 0.2, dtype=np.float32),
            captured_at=submitted_at,
            sequence=2,
        )
        await asyncio.wait_for(sender, timeout=0.5)
        return send_times, submitted_at, client

    send_times, submitted_at, client = asyncio.run(scenario())

    assert len(send_times) == 2
    assert send_times[1] - submitted_at < 0.3
    assert client.metrics.unit_ready_to_send.count == 2
    assert client.metrics.send_interval.minimum_ms >= 900


def test_sender_drops_audio_captured_behind_a_blocked_socket_write() -> None:
    class BlockedWebSocket(FakeWebSocket):
        def __init__(self, stop: threading.Event) -> None:
            super().__init__()
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.stop = stop

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))
            if len(self.sent) == 1:
                self.first_started.set()
                await self.release_first.wait()
            else:
                self.stop.set()

    async def scenario() -> tuple[BlockedWebSocket, RealtimeClient]:
        stop = threading.Event()
        client = make_client()
        websocket = BlockedWebSocket(stop)
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        client.submit_audio(
            np.full(16_000, 0.1, dtype=np.float32),
            captured_at=time.monotonic(),
            sequence=1,
        )
        sender = asyncio.create_task(
            client._sender(  # type: ignore[arg-type]
                websocket,
                lifecycle,
                stop,
            )
        )
        await websocket.first_started.wait()
        client.submit_audio(
            np.full(16_000, 0.2, dtype=np.float32),
            captured_at=time.monotonic(),
            sequence=2,
        )
        websocket.release_first.set()
        while client.metrics.dropped_input_units < 1:
            await asyncio.sleep(0)
        client._last_wire_send_at = time.monotonic() - 1.0
        client._last_sent_capture_at = time.monotonic() - 1.0
        client.submit_audio(
            np.full(16_000, 0.3, dtype=np.float32),
            captured_at=time.monotonic(),
            sequence=3,
        )
        await asyncio.wait_for(sender, timeout=0.5)
        return websocket, client

    websocket, client = asyncio.run(scenario())
    sent_means = [
        float(
            np.frombuffer(
                base64.b64decode(event["input"]["audio"]),
                dtype="<f4",
            ).mean()
        )
        for event in websocket.sent
    ]

    assert sent_means == pytest.approx([0.1, 0.3])
    assert client.metrics.input_units == 2
    assert client.metrics.dropped_input_units == 1


def test_sender_drops_unit_captured_immediately_after_previous_send() -> None:
    async def scenario() -> tuple[FakeWebSocket, RealtimeClient]:
        stop = threading.Event()
        client = make_client()
        websocket = FakeWebSocket()
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        first_capture = time.monotonic() - 1.0
        client.submit_audio(
            np.full(16_000, 0.1, dtype=np.float32),
            captured_at=first_capture,
            sequence=1,
        )
        sender = asyncio.create_task(
            client._sender(  # type: ignore[arg-type]
                websocket,
                lifecycle,
                stop,
            )
        )
        while client.metrics.input_units < 1:
            await asyncio.sleep(0)

        assert client._last_wire_send_at is not None
        second_capture = time.monotonic()
        since_wire_send = second_capture - client._last_wire_send_at
        assert 0 <= since_wire_send < client._minimum_uplink_interval
        assert second_capture - first_capture >= client._minimum_uplink_interval
        client.submit_audio(
            np.full(16_000, 0.2, dtype=np.float32),
            captured_at=second_capture,
            sequence=2,
        )

        async def wait_for_drop() -> None:
            while client.metrics.dropped_input_units < 1:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_drop(), timeout=0.2)
        stop.set()
        await asyncio.wait_for(sender, timeout=0.3)
        return websocket, client

    websocket, client = asyncio.run(scenario())

    assert len(websocket.sent) == 1
    assert client.metrics.input_units == 1
    assert client.metrics.dropped_input_units == 1


def test_first_force_listen_unit_bypasses_backpressure_drop() -> None:
    class BlockedWebSocket(FakeWebSocket):
        def __init__(self, stop: threading.Event) -> None:
            super().__init__()
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.stop = stop

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))
            if len(self.sent) == 1:
                self.first_started.set()
                await self.release_first.wait()
            else:
                self.stop.set()

    async def scenario() -> tuple[BlockedWebSocket, TurnCoordinator, RealtimeClient]:
        stop = threading.Event()
        turns = TurnCoordinator()
        turns.new_session()
        turns.accept_audio("old")
        client = make_client(turns)
        websocket = BlockedWebSocket(stop)
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        client.submit_audio(
            np.full(16_000, 0.1, dtype=np.float32),
            captured_at=time.monotonic(),
            sequence=1,
        )
        sender = asyncio.create_task(
            client._sender(  # type: ignore[arg-type]
                websocket,
                lifecycle,
                stop,
            )
        )
        await websocket.first_started.wait()
        assert turns.interrupt() is not None
        client.submit_audio(
            np.full(16_000, 0.2, dtype=np.float32),
            captured_at=time.monotonic(),
            sequence=2,
        )
        websocket.release_first.set()
        await asyncio.wait_for(sender, timeout=0.5)
        return websocket, turns, client

    websocket, turns, client = asyncio.run(scenario())

    assert len(websocket.sent) == 2
    assert websocket.sent[1]["input"]["force_listen"] is True
    assert turns.snapshot().force_listen_sent is True
    assert client.metrics.dropped_input_units == 0


def test_session_activation_discards_only_handshake_era_audio() -> None:
    async def scenario() -> tuple[FakeWebSocket, RealtimeClient]:
        client = make_client()
        stop = threading.Event()
        websocket = FakeWebSocket(on_send=stop.set)
        lifecycle = _Lifecycle()
        lifecycle.state = ProtocolState(Phase.CREATED, "sess-1")
        activation = time.monotonic()
        client._session_input_cutoff = activation
        client._discard_input_captured_through(activation)

        # This emulates a capture callback that obtained its timestamp before
        # activation but submitted just after the one-time slot purge.
        client.submit_audio(
            np.full(16_000, -0.2, dtype=np.float32),
            captured_at=activation - 0.1,
            sequence=1,
        )
        sender = asyncio.create_task(
            client._sender(  # type: ignore[arg-type]
                websocket,
                lifecycle,
                stop,
            )
        )
        while client.metrics.dropped_input_units < 1:
            await asyncio.sleep(0)
        client.submit_audio(
            np.full(16_000, 0.4, dtype=np.float32),
            captured_at=time.monotonic(),
            sequence=2,
        )
        await asyncio.wait_for(sender, timeout=0.5)
        return websocket, client

    websocket, client = asyncio.run(scenario())

    assert len(websocket.sent) == 1
    sent = np.frombuffer(
        base64.b64decode(websocket.sent[0]["input"]["audio"]),
        dtype="<f4",
    )
    assert float(sent.mean()) == pytest.approx(0.4)
    assert client.metrics.dropped_input_units == 1


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
        on_audio=lambda _samples, epoch, response, _received_at: received.append((epoch, response)),
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


def test_audio_callback_receives_raw_timestamp_before_packet_diagnostics() -> None:
    encoded = base64.b64encode(np.zeros(240, dtype="<f4").tobytes()).decode("ascii")
    websocket = FakeWebSocket(
        [
            {
                "type": "response.output.delta",
                "kind": "audio",
                "session_id": "sess-1",
                "response_id": "resp-1",
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
    order: list[str] = []
    timestamps: list[float] = []

    def on_audio(
        _samples: np.ndarray,
        _epoch: int,
        _response: str | None,
        received_at: float,
    ) -> None:
        timestamps.append(received_at)
        order.append("audio")

    client = make_client(turns, on_audio=on_audio)
    client._log_first_response_packet = lambda *_args, **_kwargs: order.append(  # type: ignore[method-assign]
        "diagnostics"
    )
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.STREAMING, "sess-1")
    before = time.monotonic()

    asyncio.run(
        client._receiver(  # type: ignore[arg-type]
            websocket,
            lifecycle,
            asyncio.Event(),
        )
    )

    assert order == ["audio", "diagnostics"]
    assert before <= timestamps[0] <= time.monotonic()


def test_response_first_packet_logs_once_per_kind_with_server_segments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pcm = np.zeros(240, dtype="<f4").tobytes()
    encoded = base64.b64encode(pcm).decode("ascii")
    server_now = time.time() - 0.05
    websocket = FakeWebSocket(
        [
            {
                "type": "response.output.delta",
                "kind": "listen",
                "session_id": "sess-1",
                "metrics": {},
                "server_send_ts": server_now,
            },
            {
                "type": "response.output.delta",
                "kind": "text",
                "session_id": "sess-1",
                "response_id": "resp-1",
                "text": "first",
                "metrics": {"prefill_ms": 12.5, "wall_clock_ms": 31},
                "server_send_ts": server_now + 0.01,
            },
            {
                "type": "response.output.delta",
                "kind": "text",
                "session_id": "sess-1",
                "response_id": "resp-1",
                "text": "second",
                "metrics": {"prefill_ms": 999},
                "server_send_ts": server_now + 0.02,
            },
            {
                "type": "response.output.delta",
                "kind": "audio",
                "session_id": "sess-1",
                "response_id": "resp-1",
                "audio": encoded,
                "metrics": {"vision_slices": 1, "kv_cache_length": 384},
                "server_send_ts": server_now + 0.03,
            },
            {
                "type": "response.output.delta",
                "kind": "audio",
                "session_id": "sess-1",
                "response_id": "resp-1",
                "audio": encoded,
                "metrics": {"kv_cache_length": 999},
                "server_send_ts": server_now + 0.04,
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
    client = make_client(turns)
    client._last_input_send_at = time.monotonic() - 0.1
    lifecycle = _Lifecycle()
    lifecycle.state = ProtocolState(Phase.STREAMING, "sess-1")
    caplog.set_level(logging.DEBUG, logger="yrobot.realtime")

    asyncio.run(
        client._receiver(  # type: ignore[arg-type]
            websocket,
            lifecycle,
            asyncio.Event(),
        )
    )

    first_packet_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "yrobot.realtime"
        and record.getMessage().startswith("MiniCPM-o first ")
        and " packet:" in record.getMessage()
    ]
    assert len(first_packet_logs) == 2
    assert first_packet_logs[0].startswith("MiniCPM-o first text packet:")
    assert "time_since_last_uplink_ms=" in first_packet_logs[0]
    assert "downlink_excess_ms=" in first_packet_logs[0]
    assert "server_prefill_ms=12.5" in first_packet_logs[0]
    assert "server_wall_clock_ms=31" in first_packet_logs[0]
    assert "server_prefill_ms=999" not in " ".join(first_packet_logs)
    assert first_packet_logs[1].startswith("MiniCPM-o first audio packet:")
    assert "e2e" not in " ".join(first_packet_logs).lower()
    diagnostic_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "yrobot.realtime"
        and record.getMessage().startswith("MiniCPM-o first ")
        and "diagnostics:" in record.getMessage()
    ]
    assert "server_send_ts=" in diagnostic_logs[0]
    assert "server_to_client_excess_ms=" in diagnostic_logs[0]
    assert "server_vision_slices=1" in diagnostic_logs[1]
    assert "server_kv_cache_length=384" in diagnostic_logs[1]
    assert "server_kv_cache_length=999" not in " ".join(diagnostic_logs)
    assert client.metrics.first_audio_downlink_excess.count == 1


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
            client_ref: list[RealtimeClient] = []

            def on_session(ready: bool, _epoch: int) -> None:
                if ready:
                    client_ref[0].submit_audio(
                        np.zeros(16_000, dtype=np.float32),
                        captured_at=time.monotonic(),
                    )

            client = RealtimeClient(
                Settings(realtime_url=(f"ws://127.0.0.1:{port}/v1/realtime?mode=video")),
                turns,
                latest_frame=lambda: None,
                on_audio=lambda _samples, _epoch, response, _received_at: received_audio.append(
                    str(response)
                ),
                on_listen=lambda _epoch: stop.set(),
                on_text=lambda *_: None,
                on_session=on_session,
            )
            client_ref.append(client)
            client.submit_audio(
                np.full(16_000, -0.5, dtype=np.float32),
                captured_at=time.monotonic(),
            )
            await asyncio.wait_for(client.run(stop), timeout=3.0)
        return server_events, received_audio

    events, audio = asyncio.run(scenario())

    assert events == ["session.init", "input.append", "session.close"]
    assert audio == ["resp-local"]
