"""MiniCPM-o Realtime Gateway protocol and bounded reconnecting client."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import queue
import random
import threading
import time
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from .config import AUDIO_UNIT_SAMPLES, Config

log = logging.getLogger(__name__)

SESSION_CLOSE_START_SECONDS = 294.0
SENDER_STOP_TIMEOUT = 0.5


class RealtimePort(Protocol):
    """Fast synchronous boundary implemented by the robot audio/runtime layer."""

    def next_audio_unit(
        self,
        timeout: float,
    ) -> tuple[np.ndarray, bool] | None:
        """Return exactly one 16 kHz/one-second unit and its force-listen latch."""

    def latest_frame_jpeg(self) -> bytes | None:
        """Return an already-encoded latest-frame snapshot without blocking."""

    def handle_audio_delta(
        self,
        samples: np.ndarray,
        response_id: str,
        metrics: Mapping[str, Any],
    ) -> None: ...

    def handle_listen(
        self,
        response_id: str,
        metrics: Mapping[str, Any],
    ) -> None: ...

    def handle_text(self, text: str, response_id: str) -> None: ...

    def handle_session_ready(self) -> None:
        """Begin accepting microphone input after the backend is initialized."""

    def invalidate_session(self, reason: str) -> None:
        """Atomically invalidate stale playback and interruption state."""

    def ready_for_rollover(self) -> bool:
        """Return true at a locally idle conversation boundary."""


class SessionOutcome(StrEnum):
    STOP = "stop"
    ROLLOVER = "rollover"


class RealtimeProtocolError(RuntimeError):
    """The peer violated the documented Realtime Gateway protocol."""


def gateway_default_ref_audio_url(realtime_url: str) -> str:
    """Return the HTTP companion endpoint for a WS Realtime URL."""

    parsed = urlsplit(realtime_url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme)
    if scheme is None or not parsed.netloc:
        raise ValueError("Realtime URL must use ws:// or wss://")
    return urlunsplit((scheme, parsed.netloc, "/api/default_ref_audio", "", ""))


def load_gateway_ref_audio(config: Config) -> str:
    """Load and validate the Gateway's default 16 kHz float32 voice prompt."""

    url = gateway_default_ref_audio_url(config.realtime_url)
    request = Request(url, headers={"Accept": "application/json"})
    options: dict[str, Any] = {"timeout": 20.0}
    if url.startswith("https://"):
        options["context"] = config.ssl_context()
    with urlopen(request, **options) as response:  # noqa: S310 - URL is operator config.
        payload = json.load(response)

    value = payload.get("base64") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value:
        raise RealtimeProtocolError("Gateway default reference audio is missing base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise RealtimeProtocolError("Gateway default reference audio is invalid base64") from exc
    if len(raw) == 0 or len(raw) % 4 or len(raw) > 16 * 1024 * 1024:
        raise RealtimeProtocolError("Gateway reference audio must be bounded float32 PCM")
    sample_rate = payload.get("sample_rate")
    if sample_rate not in {None, 16_000}:
        raise RealtimeProtocolError("Gateway reference audio must be 16 kHz")
    samples = np.frombuffer(raw, dtype="<f4")
    if not np.all(np.isfinite(samples)):
        raise RealtimeProtocolError("Gateway reference audio contains non-finite samples")
    return value


def encode_input_audio(samples: np.ndarray) -> str:
    pcm = np.asarray(samples, dtype="<f4")
    if pcm.ndim != 1 or pcm.shape != (AUDIO_UNIT_SAMPLES,):
        raise ValueError(f"Realtime input must contain exactly {AUDIO_UNIT_SAMPLES} mono samples")
    if not np.all(np.isfinite(pcm)):
        raise ValueError("Realtime input contains non-finite samples")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def decode_output_audio(value: str) -> np.ndarray:
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Realtime audio payload is not valid base64") from exc
    if not raw or len(raw) % 4:
        raise ValueError("Realtime audio payload is not float32 PCM")
    samples = np.frombuffer(raw, dtype="<f4").copy()
    if not np.all(np.isfinite(samples)):
        raise ValueError("Realtime audio payload contains non-finite samples")
    return samples


def build_session_init(
    system_prompt: str,
    *,
    length_penalty: float = 1.1,
    force_listen_count: int = 1,
    enable_tts: bool = True,
    ref_audio_base64: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "system_prompt": system_prompt,
        "config": {
            "generate_audio": enable_tts,
            "length_penalty": length_penalty,
            "force_listen_count": force_listen_count,
        },
        # Use the canonical public protocol shape. The current comni parser
        # reads reference audio only from payload.voice; its web frontend's
        # historical root-level alias is not accepted by that backend.
        "use_tts": enable_tts,
    }
    if enable_tts:
        if not ref_audio_base64:
            raise ValueError("ref_audio_base64 is required when TTS is enabled")
        payload["voice"] = {
            "ref_audio_base64": ref_audio_base64,
            "tts_ref_audio_base64": ref_audio_base64,
        }
    return {
        "type": "session.init",
        "payload": payload,
    }


def build_input_append(
    audio: np.ndarray,
    frame_jpeg: bytes | None,
    *,
    force_listen: bool,
) -> dict[str, Any]:
    model_input: dict[str, Any] = {
        "audio": encode_input_audio(audio),
        "force_listen": bool(force_listen),
        "max_slice_nums": 1,
    }
    if frame_jpeg:
        model_input["video_frames"] = [base64.b64encode(frame_jpeg).decode("ascii")]
    return {"type": "input.append", "input": model_input}


def _parse_event(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    event = json.loads(raw)
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise RealtimeProtocolError("Realtime event must be an object with a string type")
    return event


def _metrics(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("metrics")
    metrics = dict(value) if isinstance(value, Mapping) else {}
    for field in ("input_id", "server_send_ts"):
        if field in event:
            metrics[field] = event[field]
    return metrics


class RealtimeClient:
    """Maintain MiniCPM-o video sessions with bounded tasks and reconnects."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._ref_audio_base64: str | None = None

    async def run(
        self,
        port: RealtimePort,
        stop_event: threading.Event,
    ) -> None:
        backoff = self.config.reconnect_initial_delay
        while not stop_event.is_set():
            started_at = time.monotonic()
            try:
                outcome = await self.run_session(port, stop_event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if stop_event.is_set():
                    return
                lived_for = time.monotonic() - started_at
                if lived_for >= self.config.reconnect_reset_after:
                    backoff = self.config.reconnect_initial_delay
                delay = random.uniform(backoff * 0.75, backoff)
                log.warning(
                    "Realtime session failed (%s); reconnecting in %.1fs",
                    exc,
                    delay,
                )
                await self._wait_or_stop(stop_event, delay)
                backoff = min(self.config.reconnect_max_delay, backoff * 2.0)
                continue

            if outcome is SessionOutcome.STOP:
                return

            # A planned rollover is healthy and should not inherit an error
            # backoff. The backend has no resume API, so the new session still
            # resets context and incurs its normal initialization window.
            backoff = self.config.reconnect_initial_delay
            log.info("Realtime session reached rollover deadline; reconnecting")

    async def run_session(
        self,
        port: RealtimePort,
        stop_event: threading.Event,
    ) -> SessionOutcome:
        invalidation_reason = "reconnect"
        port_invalidated = False
        websocket: Any | None = None
        sender: asyncio.Task[Any] | None = None
        receiver: asyncio.Task[Any] | None = None
        tasks: set[asyncio.Task[Any]] = set()
        closing = asyncio.Event()

        try:
            if stop_event.is_set():
                invalidation_reason = "stop"
                return SessionOutcome.STOP

            port.invalidate_session("connecting")
            stopper = asyncio.create_task(
                self._stop_waiter(stop_event),
                name="realtime-stop",
            )
            tasks.add(stopper)
            ref_audio_base64 = await self._ensure_ref_audio(stop_event)
            if stop_event.is_set():
                invalidation_reason = "stop"
                port.invalidate_session(invalidation_reason)
                port_invalidated = True
                return SessionOutcome.STOP

            connect = asyncio.ensure_future(
                websockets.connect(
                    self.config.realtime_url,
                    ssl=self.config.ssl_context(),
                    open_timeout=10,
                    close_timeout=3,
                    max_size=self.config.max_message_size,
                    max_queue=8,
                    compression=None,
                    ping_interval=20,
                    ping_timeout=20,
                )
            )
            connect_started_at = asyncio.get_running_loop().time()
            connect.set_name("realtime-connect")
            tasks.add(connect)
            connected, _ = await asyncio.wait(
                {connect, stopper},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopper in connected or stop_event.is_set():
                invalidation_reason = "stop"
                port.invalidate_session(invalidation_reason)
                port_invalidated = True
                connect.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await connect
                return SessionOutcome.STOP
            websocket = connect.result()
            connected_at = asyncio.get_running_loop().time()

            handshake = asyncio.create_task(
                self._handshake(websocket, ref_audio_base64),
                name="realtime-handshake",
            )
            tasks.update({stopper, handshake})
            ready, _ = await asyncio.wait(
                {handshake, stopper},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopper in ready or stop_event.is_set():
                invalidation_reason = "stop"
                port.invalidate_session(invalidation_reason)
                port_invalidated = True
                handshake.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await handshake
                await self._send_close(websocket, "user_stop")
                return SessionOutcome.STOP

            session_id, assigned_at = handshake.result()
            created_at = asyncio.get_running_loop().time()
            port.handle_session_ready()
            log.info(
                "Realtime session ready: %s (connect %.1f ms, queue %.1f ms, init %.1f ms)",
                session_id,
                (connected_at - connect_started_at) * 1000.0,
                (assigned_at - connected_at) * 1000.0,
                (created_at - assigned_at) * 1000.0,
            )

            sender = asyncio.create_task(
                self._send_loop(websocket, port, stop_event, closing),
                name="realtime-send",
            )
            receiver = asyncio.create_task(
                self._receive_loop(websocket, port),
                name="realtime-receive",
            )
            elapsed_since_assignment = asyncio.get_running_loop().time() - assigned_at
            rollover = asyncio.create_task(
                asyncio.sleep(max(0.0, self.config.session_rollover - elapsed_since_assignment)),
                name="realtime-rollover",
            )
            tasks.update({sender, receiver, stopper, rollover})

            watched = {sender, receiver, stopper, rollover}
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)

            if stopper in done or stop_event.is_set():
                invalidation_reason = "stop"
                port.invalidate_session(invalidation_reason)
                port_invalidated = True
                await self._stop_sender(sender, closing)
                await self._close_and_wait(websocket, receiver, "user_stop")
                return SessionOutcome.STOP
            if rollover in done:
                hard_deadline = assigned_at + SESSION_CLOSE_START_SECONDS
                while not port.ready_for_rollover():
                    remaining = hard_deadline - asyncio.get_running_loop().time()
                    if remaining <= 0.0:
                        log.warning("Forcing realtime rollover at the hard deadline")
                        break
                    lifecycle_done, _ = await asyncio.wait(
                        {sender, receiver, stopper},
                        timeout=min(0.05, remaining),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stopper in lifecycle_done or stop_event.is_set():
                        invalidation_reason = "stop"
                        port.invalidate_session(invalidation_reason)
                        port_invalidated = True
                        await self._stop_sender(sender, closing)
                        await self._close_and_wait(websocket, receiver, "user_stop")
                        return SessionOutcome.STOP
                    completed_io = lifecycle_done.intersection({sender, receiver})
                    if completed_io:
                        completed = next(iter(completed_io))
                        error = completed.exception()
                        if error is not None:
                            raise error
                        raise ConnectionError(f"{completed.get_name()} ended unexpectedly")

                invalidation_reason = "rollover"
                port.invalidate_session(invalidation_reason)
                port_invalidated = True
                await self._stop_sender(sender, closing)
                await self._close_and_wait(websocket, receiver, "client_rollover")
                return SessionOutcome.ROLLOVER

            completed = next(task for task in done if task in {sender, receiver})
            error = completed.exception()
            if error is not None:
                raise error
            raise ConnectionError(f"{completed.get_name()} ended unexpectedly")
        except asyncio.CancelledError:
            invalidation_reason = "cancelled"
            port.invalidate_session(invalidation_reason)
            port_invalidated = True
            await self._stop_sender(sender, closing)
            if websocket is not None:
                await self._close_and_wait(websocket, receiver, "client_cancelled")
            raise
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            if websocket is not None:
                with contextlib.suppress(Exception):
                    await websocket.close()
            if not port_invalidated:
                try:
                    port.invalidate_session(invalidation_reason)
                except Exception:
                    log.exception("Realtime port session invalidation failed")

    async def _ensure_ref_audio(self, stop_event: threading.Event) -> str | None:
        if not self.config.enable_tts:
            return None
        if self._ref_audio_base64 is None:
            result: queue.SimpleQueue[tuple[bool, str | BaseException]] = queue.SimpleQueue()

            def load() -> None:
                try:
                    result.put((True, load_gateway_ref_audio(self.config)))
                except BaseException as exc:
                    result.put((False, exc))

            threading.Thread(
                target=load,
                name="yrobot-reference-voice",
                daemon=True,
            ).start()
            while not stop_event.is_set():
                try:
                    succeeded, value = result.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                if not succeeded:
                    assert isinstance(value, BaseException)
                    raise value
                assert isinstance(value, str)
                self._ref_audio_base64 = value
                break
            if stop_event.is_set():
                return None
            log.info("Loaded and cached the Gateway default reference voice")
        return self._ref_audio_base64

    async def _handshake(
        self,
        websocket: Any,
        ref_audio_base64: str | None = None,
    ) -> tuple[str, float]:
        queue_done = False
        init_sent = False
        assigned_at = 0.0

        async with asyncio.timeout(self.config.handshake_timeout):
            while True:
                event = _parse_event(await websocket.recv())
                event_type = event["type"]

                if event_type in {"session.queued", "session.queue_update"}:
                    position = event.get("position")
                    eta = event.get("estimated_wait_seconds")
                    log.info("Realtime queue: position=%s eta=%s", position, eta)
                    continue
                if event_type == "session.queue_done":
                    if not init_sent:
                        queue_done = True
                        assigned_at = asyncio.get_running_loop().time()
                        await websocket.send(
                            json.dumps(
                                build_session_init(
                                    self.config.system_prompt,
                                    length_penalty=self.config.length_penalty,
                                    force_listen_count=self.config.force_listen_count,
                                    enable_tts=self.config.enable_tts,
                                    ref_audio_base64=ref_audio_base64,
                                ),
                                separators=(",", ":"),
                            )
                        )
                        init_sent = True
                    continue
                if event_type == "session.created":
                    if not queue_done or not init_sent:
                        raise RealtimeProtocolError(
                            "session.created arrived before session.queue_done"
                        )
                    return str(event.get("session_id") or "unknown"), assigned_at
                if event_type == "error":
                    raise RealtimeProtocolError(json.dumps(event, ensure_ascii=False))
                if event_type == "session.closed":
                    raise ConnectionError(
                        str(event.get("reason") or "session closed during handshake")
                    )
                log.debug("Ignoring handshake event: %s", event_type)

    async def _send_loop(
        self,
        websocket: Any,
        port: RealtimePort,
        stop_event: threading.Event,
        closing: asyncio.Event,
    ) -> None:
        while not stop_event.is_set() and not closing.is_set():
            item = await asyncio.to_thread(port.next_audio_unit, 0.25)
            if item is None or closing.is_set():
                continue
            audio, force_listen = item
            # The media layer owns camera capture/JPEG encoding and exposes one
            # immutable latest-only slot, so this read cannot backpressure audio.
            frame = port.latest_frame_jpeg() if self.config.send_video else None
            if frame is not None and not isinstance(frame, bytes):
                raise TypeError("latest_frame_jpeg() must return bytes or None")
            message = await asyncio.to_thread(
                build_input_append,
                audio,
                frame,
                force_listen=force_listen,
            )
            if closing.is_set():
                return
            await websocket.send(json.dumps(message, separators=(",", ":")))

    @staticmethod
    async def _stop_sender(
        sender: asyncio.Task[Any] | None,
        closing: asyncio.Event,
    ) -> None:
        """Finish or cancel the sole input producer before session.close."""

        closing.set()
        if sender is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(sender),
                timeout=SENDER_STOP_TIMEOUT,
            )
        except TimeoutError:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
        except Exception:
            # A shutdown path still owns transport cleanup; input must remain
            # stopped even if its final send failed.
            return

    async def _receive_loop(self, websocket: Any, port: RealtimePort) -> None:
        session_closed_seen = False
        try:
            while True:
                event = _parse_event(await websocket.recv())
                event_type = event["type"]

                if event_type == "response.output.delta":
                    kind = event.get("kind")
                    response_id = str(event.get("response_id") or "")
                    if kind == "audio":
                        audio = event.get("audio")
                        if isinstance(audio, str) and audio:
                            port.handle_audio_delta(
                                decode_output_audio(audio),
                                response_id,
                                _metrics(event),
                            )
                    elif kind == "listen":
                        port.handle_listen(response_id, _metrics(event))
                    elif kind == "text":
                        text = event.get("text")
                        if isinstance(text, str) and text:
                            port.handle_text(text, response_id)
                    else:
                        log.debug("Ignoring Realtime output kind: %s", kind)
                    continue

                if event_type == "error":
                    raise RealtimeProtocolError(json.dumps(event, ensure_ascii=False))
                if event_type == "session.closed":
                    # This Gateway releases its worker only after closing the
                    # internal Worker WebSocket, then closes this transport.
                    # Waiting for that close avoids the immutable deployment's
                    # early-session.closed / next-session HTTP 403 race.
                    session_closed_seen = True
                    log.debug(
                        "Realtime session.closed received; waiting for transport cleanup: %s",
                        event.get("reason"),
                    )
                    continue
                if event_type == "response.done":
                    # Full-duplex turn boundaries are kind=listen. Some backend
                    # versions still emit response.done; it has no state effect.
                    log.debug("Ignoring chat-only response.done in duplex session")
                    continue
                log.debug("Ignoring Realtime event: %s", event_type)
        except ConnectionClosed as exc:
            if session_closed_seen:
                return
            raise ConnectionError("Realtime WebSocket transport closed") from exc

    @staticmethod
    async def _send_close(websocket: Any, reason: str) -> None:
        with contextlib.suppress(Exception):
            await websocket.send(
                json.dumps(
                    {"type": "session.close", "reason": reason},
                    separators=(",", ":"),
                )
            )

    async def _close_and_wait(
        self,
        websocket: Any,
        receiver: asyncio.Task[Any] | None,
        reason: str,
    ) -> None:
        """Ask every server layer to finish cleanup before closing transport."""

        await self._send_close(websocket, reason)
        if receiver is None or receiver.done():
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(receiver),
                timeout=self.config.close_ack_timeout,
            )
        except (TimeoutError, ConnectionError):
            log.warning(
                "Realtime close acknowledgement timed out or disconnected after %.1fs",
                self.config.close_ack_timeout,
            )

    @staticmethod
    async def _stop_waiter(stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(0.05)

    @staticmethod
    async def _wait_or_stop(stop_event: threading.Event, delay: float) -> None:
        deadline = asyncio.get_running_loop().time() + delay
        while not stop_event.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
