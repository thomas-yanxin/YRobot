"""MiniCPM-o 4.5 realtime gateway client.

Protocol (verified against https://minicpmo45.modelbest.cn/docs/en/realtime-api/):

    connect  ?mode=audio
      <- session.queued / session.queue_update      (position in line)
      <- session.queue_done                          (worker assigned)
      -> session.init {system_prompt, config}
      <- session.created                             (~14 s: server model reset)
      -> input.append {audio, video_frames?, force_listen?}   every chunk_ms
      <- response.output.delta kind in {listen,text,audio}
      -> session.close / <- session.closed

Uplink audio is base64 float32 PCM 16 kHz mono; downlink audio deltas are
24 kHz. Only a ``listen`` delta marks an utterance boundary — text and audio
deltas are independent streams. A synchronous ``websockets`` client plus one
receiver thread keeps the whole app thread-based like the Reachy SDK.
"""

from __future__ import annotations

import base64
import json
import logging
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from websockets.sync.client import connect

from yrobot.config import Settings

logger = logging.getLogger(__name__)

DOWNLINK_RATE = 24_000
UPLINK_RATE = 16_000


@dataclass(frozen=True)
class Delta:
    """One ``response.output.delta`` server event."""

    kind: str  # "listen" | "text" | "audio"
    text: str = ""
    audio: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))


class ThinkFilter:
    """Drop ``<think>…</think>`` spans that leak across text deltas.

    The Qwen3 base occasionally emits reasoning tags on the duplex path;
    they are noise for captions and must never be logged as speech.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._thinking = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            tag = "</think>" if self._thinking else "<think>"
            pos = self._buf.find(tag)
            if pos >= 0:
                if not self._thinking:
                    out.append(self._buf[:pos])
                self._buf = self._buf[pos + len(tag) :]
                self._thinking = not self._thinking
                continue
            tail = _partial_tag_suffix(self._buf, tag)
            if not self._thinking:
                out.append(self._buf[: len(self._buf) - tail])
            self._buf = self._buf[len(self._buf) - tail :]
            return "".join(out)


def _partial_tag_suffix(buf: str, tag: str) -> int:
    """Length of the longest ``buf`` suffix that is a proper prefix of ``tag``."""
    for size in range(min(len(tag) - 1, len(buf)), 0, -1):
        if tag.startswith(buf[-size:]):
            return size
    return 0


class RealtimeClient:
    """One gateway session: open() → send_chunk()* → close().

    ``on_delta`` is invoked from the receiver thread; ``on_closed`` fires
    exactly once when the session ends for any reason.
    """

    def __init__(
        self,
        settings: Settings,
        on_delta: Callable[[Delta], None],
        on_closed: Callable[[str], None],
    ) -> None:
        self._settings = settings
        self._on_delta = on_delta
        self._on_closed = on_closed
        self._ws = None
        self._send_lock = threading.Lock()
        self._queue_done = threading.Event()
        self._created = threading.Event()
        self._closed_once = threading.Event()
        self.session_id = ""

    def open(self, timeout: float = 120.0) -> None:
        """Connect, pass the queue, init the session and await creation."""
        s = self._settings
        ssl_ctx: ssl.SSLContext | None = None
        if s.url.startswith("wss://"):
            ssl_ctx = ssl.create_default_context()
            if not s.tls_verify:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
        t0 = time.monotonic()
        self._ws = connect(s.url, ssl=ssl_ctx, open_timeout=30, max_size=32 * 1024 * 1024)
        threading.Thread(target=self._receiver, name="yrobot-recv", daemon=True).start()

        if not self._wait_queue(timeout):
            raise TimeoutError("gateway queue timeout")
        self._send(
            {
                "type": "session.init",
                "payload": {
                    "system_prompt": s.system_prompt,
                    "config": {"length_penalty": s.length_penalty},
                },
            }
        )
        if not self._created.wait(timeout):
            raise TimeoutError("session.created timeout")
        logger.info("session %s ready in %.1f s", self.session_id, time.monotonic() - t0)

    def send_chunk(self, audio_16k: np.ndarray, jpeg: bytes | None, force_listen: bool) -> None:
        """Send one uplink unit: mono float32 16 kHz audio + optional frame."""
        payload: dict = {"audio": base64.b64encode(audio_16k.astype("<f4").tobytes()).decode()}
        if jpeg is not None:
            payload["video_frames"] = [base64.b64encode(jpeg).decode()]
        if force_listen:
            payload["force_listen"] = True
        self._send({"type": "input.append", "input": payload})

    def close(self, reason: str = "user_stop") -> None:
        """Best-effort graceful close; safe to call from any thread, twice."""
        try:
            self._send({"type": "session.close", "reason": reason})
            # Give the worker a moment to acknowledge and recycle cleanly;
            # an abrupt transport close tends to wedge the next connect.
            self._closed_once.wait(3.0)
        except Exception:
            pass
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    # -- internals ---------------------------------------------------------

    def _send(self, message: dict) -> None:
        if self._ws is None:
            raise ConnectionError("session not open")
        with self._send_lock:
            self._ws.send(json.dumps(message))

    def _wait_queue(self, timeout: float) -> bool:
        return self._queue_done.wait(timeout)

    def _receiver(self) -> None:
        reason = "connection_lost"
        try:
            assert self._ws is not None
            for raw in self._ws:
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype == "response.output.delta":
                    self._on_delta(_parse_delta(event))
                elif etype in ("session.queued", "session.queue_update"):
                    logger.info(
                        "queued: position %s, ~%ss wait",
                        event.get("position"),
                        event.get("estimated_wait_s"),
                    )
                elif etype == "session.queue_done":
                    self._queue_done.set()
                elif etype == "session.created":
                    self.session_id = event.get("session_id", "")
                    self._created.set()
                elif etype == "session.closed":
                    reason = event.get("reason", "closed")
                    break
                elif etype == "error":
                    logger.error("gateway error: %s", event.get("error"))
        except Exception as exc:  # noqa: BLE001 — any transport failure ends the session
            logger.info("receiver ended: %s", exc)
        if not self._closed_once.is_set():
            self._closed_once.set()
            self._on_closed(reason)


def _parse_delta(event: dict) -> Delta:
    kind = event.get("kind", "")
    if kind == "audio":
        pcm = np.frombuffer(base64.b64decode(event.get("audio", "")), dtype="<f4")
        return Delta(kind="audio", audio=pcm)
    return Delta(kind=kind, text=event.get("text", ""))
