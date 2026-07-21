"""llama-omni-server protocol codec and reconnecting WebSocket client."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import threading
import time
from typing import Any, Protocol

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from .config import OUTPUT_SAMPLE_RATE, Config

log = logging.getLogger(__name__)


class RobotPort(Protocol):
    def next_audio_chunk(self, timeout: float) -> np.ndarray | None: ...

    def flush_audio_input(self) -> None: ...

    def get_frame_jpeg(self) -> bytes | None: ...

    def play_omni_audio(self, samples: np.ndarray, response_id: str) -> bool: ...

    def force_listen_active(self) -> bool: ...

    def handle_omni_listen(self, response_id: str) -> None: ...

    def set_conversation_state(self, state: str) -> None: ...


def encode_pcm(samples: np.ndarray) -> str:
    pcm = np.asarray(samples, dtype="<f4")
    if pcm.ndim != 1:
        raise ValueError("Omni input audio must be mono")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def decode_pcm(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    if len(raw) % 4:
        raise ValueError("Omni audio payload is not float32 PCM")
    samples = np.frombuffer(raw, dtype="<f4").copy()
    if not np.all(np.isfinite(samples)):
        raise ValueError("Omni audio payload contains non-finite samples")
    return samples


def build_session_init(system_prompt: str, length_penalty: float = 1.1) -> dict[str, Any]:
    return {
        "type": "session.init",
        "payload": {
            "mode": "full_duplex",
            "use_tts": True,
            "system_prompt": system_prompt,
            "config": {"length_penalty": length_penalty},
        },
    }


def build_input_append(
    audio: np.ndarray,
    frame_jpeg: bytes | None,
    *,
    force_listen: bool = False,
) -> dict[str, Any]:
    model_input: dict[str, Any] = {"audio": encode_pcm(audio)}
    if force_listen:
        model_input["force_listen"] = True
    if frame_jpeg:
        model_input["video_frames"] = [base64.b64encode(frame_jpeg).decode("ascii")]
        model_input["max_slice_nums"] = 1
    return {"type": "input.append", "input": model_input}


def serialize_input_append(
    audio: np.ndarray,
    frame_jpeg: bytes | None,
    *,
    force_listen: bool = False,
) -> str:
    """Encode one compact input message away from the receive event loop."""
    return json.dumps(
        build_input_append(audio, frame_jpeg, force_listen=force_listen),
        separators=(",", ":"),
    )


class OmniClient:
    """Maintain one bounded full-duplex session and reconnect after failures."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def run(self, robot: RobotPort, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.run_session(robot, stop_event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if stop_event.is_set():
                    break
                log.warning(
                    "Omni session ended: %s; reconnecting (received playback is retained)",
                    exc,
                )
                robot.set_conversation_state("idle")
                await self._wait_or_stop(stop_event, self.config.reconnect_delay)

    async def run_session(self, robot: RobotPort, stop_event: threading.Event) -> None:
        if not self.config.tls_verify and self.config.omni_url.startswith("wss://"):
            log.warning("TLS certificate verification is disabled for the Omni server")

        async with websockets.connect(
            self.config.omni_url,
            ssl=self.config.ssl_context(),
            open_timeout=10,
            close_timeout=3,
            max_size=self.config.max_message_size,
            # llama-omni-server supplies the heartbeat.  websockets still
            # answers server pings automatically when its own periodic ping is
            # disabled, avoiding two independent heartbeat loops.
            ping_interval=None,
            # PCM and JPEG are already dense; permessage-deflate adds CM4 CPU
            # and event-loop jitter for little bandwidth reduction on the LAN.
            compression=None,
            max_queue=64,
        ) as websocket:
            await websocket.send(
                json.dumps(
                    build_session_init(
                        self.config.system_prompt,
                        self.config.length_penalty,
                    )
                )
            )
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.config.session_timeout)
            event = self._parse_event(raw)
            if event.get("type") != "session.created":
                raise RuntimeError(f"expected session.created, got {event.get('type')!r}")

            session_id = str(event.get("session_id") or "unknown")
            log.info("Omni session ready: %s", session_id)
            robot.flush_audio_input()
            robot.set_conversation_state("listening")

            sender = asyncio.create_task(self._send_loop(websocket, robot, stop_event))
            receiver = asyncio.create_task(self._receive_loop(websocket, robot))
            stopper = asyncio.create_task(self._stop_waiter(stop_event))
            tasks = {sender, receiver, stopper}
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if stop_event.is_set():
                    return
                for task in done:
                    if task is stopper:
                        continue
                    error = task.exception()
                    if error is not None:
                        raise error
                raise ConnectionError("Omni WebSocket closed")
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    async def _send_loop(
        self,
        websocket: Any,
        robot: RobotPort,
        stop_event: threading.Event,
    ) -> None:
        last_send_at: float | None = None
        while not stop_event.is_set():
            audio = await asyncio.to_thread(robot.next_audio_chunk, 0.25)
            if audio is None:
                continue
            frame = robot.get_frame_jpeg() if self.config.send_video else None
            force_listen = robot.force_listen_active()
            encode_started = time.perf_counter()
            message = await asyncio.to_thread(
                serialize_input_append,
                audio,
                frame,
                force_listen=force_listen,
            )
            encode_ms = (time.perf_counter() - encode_started) * 1_000
            send_started = time.perf_counter()
            await websocket.send(message)
            sent_at = time.perf_counter()
            send_ms = (sent_at - send_started) * 1_000
            if last_send_at is not None and sent_at - last_send_at > 1.15:
                log.warning(
                    "Slow Omni input cadence: %.0f ms (encode=%.1f ms, send=%.1f ms)",
                    (sent_at - last_send_at) * 1_000,
                    encode_ms,
                    send_ms,
                )
            elif encode_ms > 50.0 or send_ms > 50.0:
                # websocket.send() applies TCP flow control. A 50--100 ms
                # wait is normal for an audio + JPEG message on Wi-Fi and
                # doesn't hurt a one-second stream cadence, so keep the stage
                # timings available for diagnosis without flooding warnings.
                log.debug(
                    "Slow Omni input stage: encode=%.1f ms, send=%.1f ms",
                    encode_ms,
                    send_ms,
                )
            last_send_at = sent_at

    async def _receive_loop(self, websocket: Any, robot: RobotPort) -> None:
        transcript: dict[str, str] = {}
        # The raw full-duplex backend may relabel asynchronous Token2Wav
        # callbacks with a later response id, and response.done can precede the
        # final audio delta. Track the physical stream globally so diagnostics
        # cannot hide a real starvation gap at either boundary.
        audio_supply_until: float | None = None
        audio_stream_active = False
        last_event = "none"
        audio_samples = 0
        try:
            while True:
                event = self._parse_event(await websocket.recv())
                event_type = event.get("type")
                last_event = str(event_type)
                if event_type == "response.output.delta":
                    kind = event.get("kind")
                    last_event = f"{event_type}:{kind}"
                    if kind == "audio" and event.get("audio"):
                        response_id = str(event.get("response_id") or "current")
                        arrived_at = time.perf_counter()
                        samples = decode_pcm(event["audio"])
                        audio_samples += samples.size
                        duration = samples.size / OUTPUT_SAMPLE_RATE
                        if audio_stream_active and audio_supply_until is not None:
                            supply_gap = arrived_at - audio_supply_until
                            if supply_gap > 0.05:
                                log.warning(
                                    "TTS supply gap for %s: %.0f ms beyond buffered audio",
                                    response_id,
                                    supply_gap * 1_000,
                                )
                            else:
                                log.debug(
                                    "TTS supply margin for %s: %.0f ms",
                                    response_id,
                                    -supply_gap * 1_000,
                                )
                        audio_supply_until = max(
                            arrived_at,
                            audio_supply_until or arrived_at,
                        ) + duration
                        audio_stream_active = True
                        # RobotIO only enqueues here; its dedicated playback
                        # worker preserves order and survives a reconnect.
                        if robot.play_omni_audio(samples, response_id):
                            robot.set_conversation_state("speaking")
                    elif kind == "text":
                        response_id = str(event.get("response_id") or "current")
                        transcript[response_id] = transcript.get(response_id, "") + str(
                            event.get("text") or ""
                        )
                    elif kind == "listen":
                        robot.handle_omni_listen(str(event.get("response_id") or ""))
                        robot.set_conversation_state("listening")
                        # Silence after an explicit listen boundary is expected
                        # and must not be reported as TTS starvation.
                        audio_supply_until = None
                        audio_stream_active = False
                elif event_type == "response.done":
                    response_id = str(event.get("response_id") or "current")
                    # Text decoding can finish before Token2Wav's background
                    # callback emits its final audio delta.  Playback uses its
                    # real queue/drain state instead of resetting at this event.
                    partial_text = transcript.pop(response_id, "")
                    text = str(event.get("text") or partial_text).strip()
                    if text:
                        log.info("Reachy: %s", text)
                    robot.set_conversation_state("listening")
                elif event_type == "session.closed":
                    raise ConnectionError(str(event.get("reason") or "server closed session"))
                elif event_type == "error":
                    raise RuntimeError(json.dumps(event, ensure_ascii=False))
                else:
                    log.debug("Ignoring Omni event: %s", event_type)
        except ConnectionClosed as exc:
            duration = audio_samples / OUTPUT_SAMPLE_RATE
            raise ConnectionError(
                "WebSocket transport lost "
                f"({self._close_details(exc)}; last_event={last_event}; "
                f"received_audio={duration:.2f}s)"
            ) from exc

    @staticmethod
    def _parse_event(raw: str | bytes) -> dict[str, Any]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("Omni event must be an object with a string type")
        return event

    @staticmethod
    def _close_details(exc: ConnectionClosed) -> str:
        received = exc.rcvd
        sent = exc.sent
        if received is None and sent is None:
            return "no close frame"
        details = []
        if received is not None:
            details.append(f"server={received.code}:{received.reason or '-'}")
        if sent is not None:
            details.append(f"client={sent.code}:{sent.reason or '-'}")
        return ", ".join(details)

    @staticmethod
    async def _stop_waiter(stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

    @staticmethod
    async def _wait_or_stop(stop_event: threading.Event, delay: float) -> None:
        deadline = asyncio.get_running_loop().time() + delay
        while not stop_event.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
