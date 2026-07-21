"""llama-omni-server protocol codec and reconnecting WebSocket client."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import threading
from typing import Any, Protocol

import numpy as np
import websockets

from .config import Config

log = logging.getLogger(__name__)


class RobotPort(Protocol):
    def next_audio_chunk(self, timeout: float) -> np.ndarray | None: ...

    def flush_audio_input(self) -> None: ...

    def get_frame_jpeg(self) -> bytes | None: ...

    def play_omni_audio(self, samples: np.ndarray) -> None: ...

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


def build_session_init(system_prompt: str) -> dict[str, Any]:
    return {
        "type": "session.init",
        "payload": {
            "mode": "full_duplex",
            "use_tts": True,
            "system_prompt": system_prompt,
        },
    }


def build_input_append(audio: np.ndarray, frame_jpeg: bytes | None) -> dict[str, Any]:
    model_input: dict[str, Any] = {"audio": encode_pcm(audio)}
    if frame_jpeg:
        model_input["video_frames"] = [base64.b64encode(frame_jpeg).decode("ascii")]
        model_input["max_slice_nums"] = 1
    return {"type": "input.append", "input": model_input}


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
                log.warning("Omni session ended: %s; reconnecting", exc)
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
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            await websocket.send(json.dumps(build_session_init(self.config.system_prompt)))
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.config.session_timeout)
            event = self._parse_event(raw)
            if event.get("type") != "session.created":
                raise RuntimeError(f"expected session.created, got {event.get('type')!r}")

            log.info("Omni session ready: %s", event.get("session_id", "unknown"))
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
        while not stop_event.is_set():
            audio = await asyncio.to_thread(robot.next_audio_chunk, 0.25)
            if audio is None:
                continue
            frame = None
            if self.config.send_video:
                frame = await asyncio.to_thread(robot.get_frame_jpeg)
            await websocket.send(json.dumps(build_input_append(audio, frame)))

    async def _receive_loop(self, websocket: Any, robot: RobotPort) -> None:
        transcript: dict[str, str] = {}
        while True:
            event = self._parse_event(await websocket.recv())
            event_type = event.get("type")
            if event_type == "response.output.delta":
                kind = event.get("kind")
                if kind == "audio" and event.get("audio"):
                    robot.set_conversation_state("speaking")
                    await asyncio.to_thread(robot.play_omni_audio, decode_pcm(event["audio"]))
                elif kind == "text":
                    response_id = str(event.get("response_id") or "current")
                    transcript[response_id] = transcript.get(response_id, "") + str(
                        event.get("text") or ""
                    )
                elif kind == "listen":
                    robot.set_conversation_state("listening")
            elif event_type == "response.done":
                response_id = str(event.get("response_id") or "current")
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

    @staticmethod
    def _parse_event(raw: str | bytes) -> dict[str, Any]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("Omni event must be an object with a string type")
        return event

    @staticmethod
    async def _stop_waiter(stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

    @staticmethod
    async def _wait_or_stop(stop_event: threading.Event, delay: float) -> None:
        deadline = asyncio.get_running_loop().time() + delay
        while not stop_event.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
