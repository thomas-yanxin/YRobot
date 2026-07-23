"""Reconnectable MiniCPM-o Realtime transport with a latest-only uplink."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import ssl
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .config import Settings
from .protocol import (
    Phase,
    ProtocolState,
    QueueDone,
    QueueStatus,
    ResponseDelta,
    ServerError,
    SessionClosed,
    SessionCreated,
    input_append,
    parse_server_event,
    serialize_client_event,
    session_close,
    session_init,
    transition_client,
    transition_server,
    validate_video_url,
)
from .state import TurnCoordinator

log = logging.getLogger(__name__)


class RealtimeError(RuntimeError):
    """The Realtime peer or transport cannot continue the current session."""


class _StopRequested(Exception):
    pass


@dataclass(frozen=True, slots=True)
class UplinkUnit:
    sequence: int
    captured_at: float
    pcm_f32le: bytes


@dataclass(frozen=True, slots=True)
class TimingSummary:
    count: int
    minimum_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    maximum_ms: float | None


@dataclass(frozen=True, slots=True)
class RealtimeMetrics:
    connections: int
    reconnects: int
    input_units: int
    dropped_input_units: int
    audio_deltas: int
    text_deltas: int
    listen_deltas: int
    unit_ready_to_send: TimingSummary
    send_interval: TimingSummary
    latest_input_to_first_audio: TimingSummary


class _Counters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.connections = 0
        self.reconnects = 0
        self.input_units = 0
        self.dropped_input_units = 0
        self.audio_deltas = 0
        self.text_deltas = 0
        self.listen_deltas = 0
        self.timings: dict[str, deque[float]] = {
            "unit_ready_to_send": deque(maxlen=256),
            "send_interval": deque(maxlen=256),
            "latest_input_to_first_audio": deque(maxlen=256),
        }

    def add(self, name: str, amount: int = 1) -> None:
        with self.lock:
            setattr(self, name, getattr(self, name) + amount)

    def observe(self, name: str, milliseconds: float) -> None:
        if milliseconds < 0 or not np.isfinite(milliseconds):
            return
        with self.lock:
            self.timings[name].append(float(milliseconds))

    def snapshot(self) -> RealtimeMetrics:
        with self.lock:
            return RealtimeMetrics(
                connections=self.connections,
                reconnects=self.reconnects,
                input_units=self.input_units,
                dropped_input_units=self.dropped_input_units,
                audio_deltas=self.audio_deltas,
                text_deltas=self.text_deltas,
                listen_deltas=self.listen_deltas,
                unit_ready_to_send=_timing_summary(self.timings["unit_ready_to_send"]),
                send_interval=_timing_summary(self.timings["send_interval"]),
                latest_input_to_first_audio=_timing_summary(
                    self.timings["latest_input_to_first_audio"]
                ),
            )


def _timing_summary(values: deque[float]) -> TimingSummary:
    if not values:
        return TimingSummary(0, None, None, None, None)
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return round(ordered[index], 1)

    return TimingSummary(
        count=len(ordered),
        minimum_ms=round(ordered[0], 1),
        p50_ms=percentile(0.50),
        p95_ms=percentile(0.95),
        maximum_ms=round(ordered[-1], 1),
    )


class _Lifecycle:
    """Serialize wire sends and lifecycle transitions across async tasks."""

    def __init__(self) -> None:
        self.state = ProtocolState()
        self.lock = asyncio.Lock()

    async def send(self, websocket: ClientConnection, event: dict[str, Any]) -> None:
        await self.send_then(websocket, event)

    async def send_then(
        self,
        websocket: ClientConnection,
        event: dict[str, Any],
        after_send: Callable[[], None] | None = None,
    ) -> None:
        async with self.lock:
            next_state = transition_client(self.state, event)  # type: ignore[arg-type]
            await websocket.send(serialize_client_event(event))  # type: ignore[arg-type]
            self.state = next_state
            if after_send is not None:
                after_send()

    async def receive(self, raw: str | bytes) -> object:
        event = parse_server_event(raw)  # type: ignore[arg-type]
        async with self.lock:
            # A best-effort close may race already-buffered output. It is
            # neither a new turn nor a lifecycle violation; callers discard it.
            if self.state.phase is Phase.CLOSE and isinstance(event, ResponseDelta):
                return event
            self.state = transition_server(self.state, event)
        return event


class RealtimeClient:
    """Own the official MiniCPM-o video Realtime session lifecycle.

    Microphone workers submit exact one-second units synchronously. Only the
    newest unsent unit is retained, so a stalled socket can never replay a
    burst of stale microphone audio after reconnecting.
    """

    def __init__(
        self,
        settings: Settings,
        coordinator: TurnCoordinator,
        *,
        latest_frame: Callable[[], bytes | None],
        on_audio: Callable[[np.ndarray, int, str | None], None],
        on_listen: Callable[[int], None],
        on_text: Callable[[str, int, str | None], None],
        on_session: Callable[[bool, int], None],
    ) -> None:
        validate_video_url(settings.realtime_url)
        self.settings = settings
        self.coordinator = coordinator
        self.latest_frame = latest_frame
        self.on_audio = on_audio
        self.on_listen = on_listen
        self.on_text = on_text
        self.on_session = on_session
        self._input_lock = threading.Lock()
        self._latest_input: UplinkUnit | None = None
        self._input_loop: asyncio.AbstractEventLoop | None = None
        self._input_event: asyncio.Event | None = None
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._metrics = _Counters()
        self._last_wire_send_at: float | None = None
        self._last_sent_capture_at: float | None = None
        self._last_input_send_at: float | None = None
        self._minimum_uplink_interval = settings.input_unit_ms / 1_000 * 0.8
        self._session_input_cutoff: float | None = None
        self._response_segment = 0
        self._logged_first_packets: set[tuple[str, str]] = set()
        self._server_client_offset_floor: float | None = None
        self._input_send_timeout = 2.0

    @property
    def metrics(self) -> RealtimeMetrics:
        return self._metrics.snapshot()

    def submit_audio(
        self,
        samples: np.ndarray | bytes | bytearray | memoryview,
        *,
        captured_at: float | None = None,
        sequence: int | None = None,
    ) -> None:
        """Publish one exact 16 kHz, mono, one-second F32LE input unit."""

        if isinstance(samples, np.ndarray):
            pcm = np.asarray(samples, dtype="<f4")
            if pcm.ndim != 1 or pcm.size != self.settings.input_sample_rate:
                raise ValueError("uplink audio must be exactly one second of mono 16 kHz")
            if not np.all(np.isfinite(pcm)):
                raise ValueError("uplink audio contains non-finite samples")
            payload = np.ascontiguousarray(pcm).tobytes()
        else:
            payload = bytes(samples)
            expected = self.settings.input_sample_rate * np.dtype("<f4").itemsize
            if len(payload) != expected:
                raise ValueError("uplink F32LE payload must contain exactly 16000 samples")
            if not np.all(np.isfinite(np.frombuffer(payload, dtype="<f4"))):
                raise ValueError("uplink audio contains non-finite samples")

        if sequence is None:
            with self._sequence_lock:
                self._sequence += 1
                sequence = self._sequence
        unit = UplinkUnit(
            sequence=sequence,
            captured_at=time.monotonic() if captured_at is None else captured_at,
            pcm_f32le=payload,
        )
        with self._input_lock:
            if self._latest_input is not None:
                self._metrics.add("dropped_input_units")
            self._latest_input = unit
            loop = self._input_loop
            event = self._input_event
        if loop is not None and event is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(event.set)

    async def run(self, stop_event: threading.Event) -> None:
        """Reconnect until the app stops; sessions roll before the 300 s limit."""

        self._bind_input_loop()
        backoff = 0.25
        reconnecting = False
        try:
            while not stop_event.is_set():
                if reconnecting:
                    self._metrics.add("reconnects")
                try:
                    should_stop = await self._run_session(stop_event)
                    if should_stop:
                        return
                    backoff = 0.25
                except asyncio.CancelledError:
                    raise
                except _StopRequested:
                    return
                except Exception as exc:
                    if stop_event.is_set():
                        return
                    delay = random.uniform(backoff * 0.75, backoff)
                    log.warning(
                        "Realtime session failed (%s); retrying in %.2fs",
                        exc,
                        delay,
                    )
                    await self._wait_or_stop(stop_event, delay)
                    backoff = min(self.settings.reconnect_max_seconds, backoff * 2.0)
                reconnecting = True
        finally:
            self._unbind_input_loop()

    async def _run_session(self, stop_event: threading.Event) -> bool:
        activated = False
        lifecycle = _Lifecycle()
        ssl_context = self._ssl_context()
        options: dict[str, Any] = {
            "open_timeout": 10,
            "close_timeout": 3,
            "max_size": 16 * 1024 * 1024,
            "max_queue": 4,
            "compression": None,
            "ping_interval": 20,
            "ping_timeout": 20,
            "proxy": None,
        }
        if ssl_context is not None:
            options["ssl"] = ssl_context

        try:
            async with connect(self.settings.realtime_url, **options) as websocket:
                self._metrics.add("connections")
                await self._wait_for_queue(websocket, lifecycle, stop_event)
                session_deadline = asyncio.get_running_loop().time() + self.settings.session_seconds
                init = session_init(
                    self.settings.system_prompt,
                    length_penalty=self.settings.length_penalty,
                )
                await self._send_init(
                    websocket,
                    lifecycle,
                    init,
                    stop_event,
                    session_deadline,
                )
                created = await self._wait_for_created(
                    websocket,
                    lifecycle,
                    stop_event,
                    session_deadline,
                )

                # Audio captured while waiting for a GPU/session is no longer
                # live context. Drop it at the activation boundary instead of
                # replaying it as the first unit of the new session.
                activation = time.monotonic()
                self._session_input_cutoff = activation
                self._discard_input_captured_through(activation)
                self._last_wire_send_at = None
                self._last_sent_capture_at = None
                self._last_input_send_at = None
                self._response_segment = 0
                self._logged_first_packets.clear()
                self._server_client_offset_floor = None
                epoch = self.coordinator.new_session()
                self.on_session(True, epoch)
                activated = True
                log.info("MiniCPM-o session ready: %s", created.session_id)
                return await self._stream(
                    websocket,
                    lifecycle,
                    stop_event,
                    session_deadline,
                )
        except ConnectionClosed as exc:
            if stop_event.is_set():
                return True
            raise RealtimeError(
                f"WebSocket closed ({exc.code}: {exc.reason or 'no reason'})"
            ) from exc
        finally:
            if activated:
                epoch = self.coordinator.session_lost()
                self.on_session(False, epoch)

    async def _wait_for_queue(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        stop_event: threading.Event,
    ) -> None:
        while True:
            event = await self._receive_one(websocket, lifecycle, stop_event)
            if isinstance(event, QueueDone):
                return
            if isinstance(event, QueueStatus):
                log.info(
                    "MiniCPM-o queue position %d%s",
                    event.position,
                    (
                        f", estimated {event.estimated_wait_s:.1f}s"
                        if event.estimated_wait_s is not None
                        else ""
                    ),
                )
                continue
            if isinstance(event, ServerError):
                raise RealtimeError(f"MiniCPM-o queue error: {event.error}")
            raise RealtimeError(f"unexpected event before queue_done: {event!r}")

    async def _wait_for_created(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        stop_event: threading.Event,
        session_deadline: float,
    ) -> SessionCreated:
        while True:
            if asyncio.get_running_loop().time() >= session_deadline:
                raise RealtimeError("session initialization exceeded its video budget")
            try:
                event = await self._receive_one(
                    websocket,
                    lifecycle,
                    stop_event,
                    deadline=session_deadline,
                )
            except TimeoutError as exc:
                raise RealtimeError("session initialization exceeded its video budget") from exc
            if isinstance(event, SessionCreated):
                return event
            if isinstance(event, ServerError):
                raise RealtimeError(f"MiniCPM-o init error: {event.error}")
            raise RealtimeError(f"unexpected event before session.created: {event!r}")

    async def _send_init(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        event: dict[str, Any],
        stop_event: threading.Event,
        session_deadline: float,
    ) -> None:
        """Bound the initialization write without cancelling an active send first."""

        loop = asyncio.get_running_loop()
        write_deadline = min(session_deadline, loop.time() + 5.0)
        sending = asyncio.create_task(
            lifecycle.send(websocket, event),
            name="minicpmo-session-init",
        )
        failure: Exception | None = None
        try:
            while not sending.done():
                if stop_event.is_set():
                    failure = _StopRequested()
                    break
                remaining = write_deadline - loop.time()
                if remaining <= 0:
                    failure = RealtimeError("session.init send timed out")
                    break
                await asyncio.wait({sending}, timeout=min(0.05, remaining))
        except asyncio.CancelledError:
            await self._abort_send(
                websocket,
                sending,
                code=1000,
                reason="init_cancelled",
            )
            raise
        if failure is None:
            await sending
            return

        await self._abort_send(
            websocket,
            sending,
            code=1000,
            reason="init_aborted",
        )
        raise failure

    async def _receive_one(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        stop_event: threading.Event,
        *,
        deadline: float | None = None,
    ) -> object:
        loop = asyncio.get_running_loop()
        while not stop_event.is_set():
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                receive_timeout = min(0.25, remaining)
            else:
                receive_timeout = 0.25
            try:
                raw = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=receive_timeout,
                )
            except TimeoutError:
                if deadline is not None and loop.time() >= deadline:
                    raise
                continue
            return await lifecycle.receive(raw)
        raise _StopRequested

    async def _stream(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        stop_event: threading.Event,
        session_deadline: float,
    ) -> bool:
        closed = asyncio.Event()
        sender_stop = asyncio.Event()
        sender = asyncio.create_task(
            self._sender(websocket, lifecycle, stop_event, sender_stop),
            name="minicpmo-uplink",
        )
        receiver = asyncio.create_task(
            self._receiver(websocket, lifecycle, closed),
            name="minicpmo-downlink",
        )
        stopper = asyncio.create_task(
            self._stop_waiter(stop_event),
            name="minicpmo-stop",
        )
        rollover = asyncio.create_task(
            asyncio.sleep(max(0.0, session_deadline - asyncio.get_running_loop().time())),
            name="minicpmo-rollover",
        )
        tasks = {sender, receiver, stopper, rollover}
        reason = "session_rollover"
        should_stop = False

        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                error = receiver.exception()
                if error is not None:
                    raise error
                return stop_event.is_set()
            if stopper in done or (sender in done and stop_event.is_set()):
                reason = "user_stop"
                should_stop = True
            elif sender in done:
                error = sender.exception()
                if error is not None:
                    raise error
                raise RealtimeError("microphone sender stopped unexpectedly")

            sender_stop.set()
            if self._input_event is not None:
                self._input_event.set()
            try:
                await asyncio.wait_for(asyncio.shield(sender), timeout=1.0)
            except TimeoutError:
                log.warning("uplink did not stop cooperatively; closing the WebSocket")
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        websocket.close(code=1000, reason=reason),
                        timeout=1.0,
                    )
                if not sender.done():
                    sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender
                return should_stop

            if lifecycle.state.phase in {Phase.CREATED, Phase.STREAMING} and not receiver.done():
                close_send = asyncio.create_task(
                    lifecycle.send(websocket, session_close(reason)),
                    name="minicpmo-session-close",
                )
                try:
                    await asyncio.wait_for(asyncio.shield(close_send), timeout=1.0)
                except asyncio.CancelledError:
                    await self._abort_send(
                        websocket,
                        close_send,
                        code=1000,
                        reason=reason,
                    )
                    raise
                except TimeoutError:
                    log.warning("session.close send timed out; closing the WebSocket")
                    await self._abort_send(
                        websocket,
                        close_send,
                        code=1000,
                        reason=reason,
                    )
                    return should_stop
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(closed.wait(), timeout=2.0)
            return should_stop
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _sender(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        stop_event: threading.Event,
        session_stop: asyncio.Event | None = None,
    ) -> None:
        self._bind_input_loop()
        if session_stop is None:
            session_stop = asyncio.Event()
        while not stop_event.is_set() and not session_stop.is_set():
            unit = await self._next_input(stop_event, session_stop)
            if unit is None:
                continue
            snapshot = self.coordinator.snapshot()
            age = time.monotonic() - unit.captured_at
            if age > 1.5:
                self._metrics.add("dropped_input_units")
                continue
            if (
                self._session_input_cutoff is not None
                and unit.captured_at <= self._session_input_cutoff
            ):
                self._metrics.add("dropped_input_units")
                continue
            self._session_input_cutoff = None
            urgent_force = snapshot.force_listen and not snapshot.force_listen_sent
            # A recovering appsink/socket may release complete one-second units
            # in a burst. Drop catch-up audio instead of accelerating the model
            # timeline, except for the first packet that must carry barge-in.
            if not urgent_force and (
                (
                    self._last_wire_send_at is not None
                    and unit.captured_at - self._last_wire_send_at < self._minimum_uplink_interval
                )
                or (
                    self._last_sent_capture_at is not None
                    and unit.captured_at - self._last_sent_capture_at
                    < self._minimum_uplink_interval
                )
            ):
                self._metrics.add("dropped_input_units")
                continue
            frame = self.latest_frame()
            if stop_event.is_set() or session_stop.is_set():
                return
            event = input_append(
                unit.pcm_f32le,
                video_frames=(frame,) if frame else (),
                force_listen=snapshot.force_listen,
                max_slice_nums=1,
            )
            after_send = (
                partial(self.coordinator.force_listen_sent, snapshot.epoch)
                if snapshot.force_listen
                else None
            )
            await self._send_input(
                websocket,
                lifecycle,
                event,
                after_send,
            )
            sent_at = time.monotonic()
            self._metrics.observe(
                "unit_ready_to_send",
                (sent_at - unit.captured_at) * 1_000,
            )
            if self._last_wire_send_at is not None:
                self._metrics.observe(
                    "send_interval",
                    (sent_at - self._last_wire_send_at) * 1_000,
                )
            self._last_wire_send_at = sent_at
            self._last_sent_capture_at = unit.captured_at
            self._last_input_send_at = sent_at
            self._metrics.add("input_units")

    async def _send_input(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        event: dict[str, Any],
        after_send: Callable[[], None] | None,
    ) -> None:
        sending = asyncio.create_task(
            lifecycle.send_then(websocket, event, after_send),
            name="minicpmo-input-send",
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(sending),
                timeout=self._input_send_timeout,
            )
        except asyncio.CancelledError:
            await self._abort_send(
                websocket,
                sending,
                code=1000,
                reason="input_send_cancelled",
            )
            raise
        except TimeoutError as exc:
            log.warning("input.append send timed out; closing the WebSocket")
            await self._abort_send(
                websocket,
                sending,
                code=1011,
                reason="input_send_timeout",
            )
            raise RealtimeError("input.append send timed out") from exc

    async def _abort_send(
        self,
        websocket: ClientConnection,
        sending: asyncio.Task[None],
        *,
        code: int,
        reason: str,
    ) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                websocket.close(code=code, reason=reason),
                timeout=1.0,
            )
        if not sending.done():
            sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)

    async def _next_input(
        self,
        stop_event: threading.Event,
        session_stop: asyncio.Event | None = None,
    ) -> UplinkUnit | None:
        if stop_event.is_set() or (session_stop is not None and session_stop.is_set()):
            return None
        self._bind_input_loop()
        assert self._input_event is not None
        self._input_event.clear()
        with self._input_lock:
            unit = self._latest_input
            self._latest_input = None
        if unit is not None:
            return unit
        try:
            await asyncio.wait_for(self._input_event.wait(), timeout=0.1)
        except TimeoutError:
            return None
        if stop_event.is_set() or (session_stop is not None and session_stop.is_set()):
            return None
        with self._input_lock:
            unit = self._latest_input
            self._latest_input = None
        return unit

    async def _receiver(
        self,
        websocket: ClientConnection,
        lifecycle: _Lifecycle,
        closed: asyncio.Event,
    ) -> None:
        while True:
            raw = await websocket.recv()
            received_monotonic = time.monotonic()
            received_wall = time.time()
            last_input_send_at = self._last_input_send_at
            event = await lifecycle.receive(raw)
            if isinstance(event, SessionClosed):
                closed.set()
                return
            if isinstance(event, ServerError):
                raise RealtimeError(f"MiniCPM-o server error: {event.error}")
            if not isinstance(event, ResponseDelta):
                raise RealtimeError(f"unexpected streaming event: {event!r}")
            if lifecycle.state.phase is Phase.CLOSE:
                continue

            downlink_excess_ms = self._server_to_client_excess_ms(
                event.server_send_ts,
                received_wall,
            )
            if event.kind == "listen":
                epoch = self.coordinator.model_listening()
                if epoch is None:
                    continue
                self._response_segment += 1
                self._metrics.add("listen_deltas")
                self.on_listen(epoch)
                continue
            if event.kind == "text":
                self._metrics.add("text_deltas")
                snapshot = self.coordinator.snapshot()
                if not snapshot.drop_output and event.text is not None:
                    self._log_first_response_packet(
                        event,
                        received_monotonic=received_monotonic,
                        received_wall=received_wall,
                        last_input_send_at=last_input_send_at,
                        server_to_client_excess_ms=downlink_excess_ms,
                    )
                    self.on_text(event.text, snapshot.epoch, event.response_id)
                continue

            assert event.audio is not None
            epoch = self.coordinator.accept_audio(event.response_id)
            if epoch is None:
                continue
            self._log_first_response_packet(
                event,
                received_monotonic=received_monotonic,
                received_wall=received_wall,
                last_input_send_at=last_input_send_at,
                server_to_client_excess_ms=downlink_excess_ms,
            )
            samples = np.frombuffer(event.audio.pcm_f32le, dtype="<f4").copy()
            if not np.all(np.isfinite(samples)):
                raise RealtimeError("MiniCPM-o output contains non-finite samples")
            self._metrics.add("audio_deltas")
            self.on_audio(samples, epoch, event.response_id)

    def _log_first_response_packet(
        self,
        event: ResponseDelta,
        *,
        received_monotonic: float,
        received_wall: float,
        last_input_send_at: float | None,
        server_to_client_excess_ms: float | None,
    ) -> None:
        response_key = (
            f"id:{event.response_id}"
            if event.response_id is not None
            else f"anonymous-segment:{self._response_segment}"
        )
        marker = (response_key, event.kind)
        if marker in self._logged_first_packets:
            return
        self._logged_first_packets.add(marker)

        fields = [
            f"response_id={event.response_id or '-'}",
            f"client_receive_ts={received_wall:.6f}",
        ]
        if last_input_send_at is not None:
            latest_input_age_ms = (received_monotonic - last_input_send_at) * 1_000
            fields.append(f"latest_input_send_age_ms={latest_input_age_ms:.1f}")
            if event.kind == "audio":
                self._metrics.observe(
                    "latest_input_to_first_audio",
                    latest_input_age_ms,
                )
        if event.server_send_ts is not None:
            fields.append(f"server_send_ts={event.server_send_ts:.6f}")
        if server_to_client_excess_ms is not None:
            fields.append(f"server_to_client_excess_ms={server_to_client_excess_ms:.1f}")
        for name in (
            "prefill_ms",
            "generate_ms",
            "wall_clock_ms",
            "cost_llm_ms",
            "cost_tts_prep_ms",
            "cost_tts_ms",
            "cost_token2wav_ms",
            "vision_slices",
            "vision_tokens",
            "kv_cache_length",
        ):
            value = _finite_metric(event.metrics.get(name))
            if value is not None:
                fields.append(f"server_{name}={value:g}")
        log.info(
            "MiniCPM-o first %s packet: %s",
            event.kind,
            " ".join(fields),
        )

    def _server_to_client_excess_ms(
        self,
        server_send_ts: float | None,
        client_receive_ts: float,
    ) -> float | None:
        if server_send_ts is None:
            return None
        current_offset = client_receive_ts - server_send_ts
        if (
            self._server_client_offset_floor is None
            or current_offset < self._server_client_offset_floor
        ):
            self._server_client_offset_floor = current_offset
        # Relative excess over the best observed path. It is not one-way
        # latency because client and server wall clocks need not be synchronized.
        return max(0.0, (current_offset - self._server_client_offset_floor) * 1_000)

    def _discard_input_captured_through(self, cutoff: float) -> None:
        dropped = False
        with self._input_lock:
            unit = self._latest_input
            if unit is not None and unit.captured_at <= cutoff:
                self._latest_input = None
                dropped = True
                if self._input_event is not None:
                    self._input_event.clear()
        if dropped:
            self._metrics.add("dropped_input_units")

    async def _stop_waiter(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(0.05)

    async def _wait_or_stop(
        self,
        stop_event: threading.Event,
        seconds: float,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + seconds
        while not stop_event.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

    def _bind_input_loop(self) -> None:
        loop = asyncio.get_running_loop()
        with self._input_lock:
            if self._input_loop is loop and self._input_event is not None:
                return
            if self._input_loop is not None and not self._input_loop.is_closed():
                raise RuntimeError("RealtimeClient is already bound to another event loop")
            event = asyncio.Event()
            self._input_loop = loop
            self._input_event = event
            pending = self._latest_input is not None
        if pending:
            event.set()

    def _unbind_input_loop(self) -> None:
        loop = asyncio.get_running_loop()
        with self._input_lock:
            if self._input_loop is loop:
                self._input_loop = None
                self._input_event = None

    def _ssl_context(self) -> ssl.SSLContext | None:
        if urlsplit(self.settings.realtime_url).scheme != "wss":
            return None
        context = ssl.create_default_context()
        if not self.settings.tls_verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context


def _finite_metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if np.isfinite(number) else None
