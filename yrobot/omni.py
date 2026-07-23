"""Client for the MiniCPM-o 4.5 realtime gateway (llama.cpp-omni backend).

Protocol (verified against the live server):
- uplink `input.append`: base64 raw float32-LE PCM, mono 16 kHz, one chunk per
  message, optional `video_frames` (base64 JPEG) and `force_listen`.
- downlink `response.output.delta` with kind ∈ listen/text/audio; audio is
  float32-LE 24 kHz. `response_id`/`response.done` are per one-second slice,
  NOT per utterance — only a `listen` delta is an utterance boundary.
- The kv cache degrades past 8192 tokens and vision costs ~64 tokens/frame,
  so sessions are rotated proactively (kv- and age-aware). Session creation
  takes ~14 s of model reset server-side and an immediate reconnect is
  rejected with backend_error — hence the paced retry loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import ssl
import threading
import time
from typing import Optional, Protocol

import numpy as np
import websockets

from yrobot.config import Config

logger = logging.getLogger(__name__)


class Sink(Protocol):
    def on_ready(self, ready: bool) -> None: ...
    def on_listen(self) -> None: ...
    def on_model_audio(self, pcm24k: np.ndarray) -> None: ...
    def on_text(self, text: str) -> None: ...
    def quiet(self) -> bool: ...


def encode_append(audio: np.ndarray, frame_jpeg: Optional[bytes], force_listen: bool) -> str:
    inp: dict = {
        "audio": base64.b64encode(audio.astype(np.float32).tobytes()).decode(),
        "force_listen": force_listen,
    }
    if frame_jpeg is not None:
        inp["video_frames"] = [base64.b64encode(frame_jpeg).decode()]
        inp["max_slice_nums"] = 1
    return json.dumps({"type": "input.append", "input": inp})


def should_rotate(cfg: Config, kv: int, age_s: float, quiet: bool) -> Optional[str]:
    if kv >= cfg.kv_hard:
        return f"kv={kv}"
    if age_s >= cfg.session_budget_s:
        return f"age={age_s:.0f}s"
    if quiet and (kv >= cfg.kv_soft or age_s >= cfg.session_budget_s - 30):
        return f"quiet kv={kv} age={age_s:.0f}s"
    return None


class OmniClient:
    """Owns the websocket in a dedicated asyncio thread; feeds/serves the app."""

    def __init__(self, cfg: Config, sink: Sink):
        self._cfg = cfg
        self._sink = sink
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._q: Optional[asyncio.Queue] = None
        self._stopping = threading.Event()
        self._ready = False
        self._thread = threading.Thread(target=self._thread_main, name="yrobot-omni", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)  # wake the loop
        self._thread.join(timeout=10.0)

    def submit(self, audio: np.ndarray, frame_jpeg: Optional[bytes], force_listen: bool) -> None:
        """Thread-safe uplink; drops when no session is ready (model is deaf anyway)."""
        loop, q = self._loop, self._q
        if loop is None or q is None or not self._ready:
            return
        msg = encode_append(audio, frame_jpeg, force_listen)
        loop.call_soon_threadsafe(self._enqueue, q, msg)

    @staticmethod
    def _enqueue(q: asyncio.Queue, msg: str) -> None:
        if q.full():  # drop-oldest keeps the model on real time, never backlog
            q.get_nowait()
        q.put_nowait(msg)

    # -- session loop ----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        backoff = self._cfg.reconnect_initial_s
        while not self._stopping.is_set():
            try:
                await self._session()
                backoff = self._cfg.reconnect_initial_s
            except Exception as e:
                logger.warning("session error: %s", e)
                backoff = min(backoff * 2, self._cfg.reconnect_max_s)
            finally:
                self._set_ready(False)
            if self._stopping.is_set():
                break
            await asyncio.sleep(backoff)

    async def _session(self) -> None:
        cfg = self._cfg
        ssl_ctx: ssl.SSLContext | None = None
        if cfg.full_url.startswith("wss"):
            ssl_ctx = (ssl.create_default_context() if cfg.tls_verify
                       else ssl._create_unverified_context())
        self._q = asyncio.Queue(maxsize=4)
        kv = 0
        t_created = 0.0
        rotating = False
        sender: Optional[asyncio.Task] = None
        logger.info("connecting %s", cfg.full_url)
        async with websockets.connect(
            cfg.full_url, ssl=ssl_ctx, open_timeout=15,
            ping_interval=10, ping_timeout=15, max_size=128 * 1024 * 1024,
        ) as ws:
            deadline = time.monotonic() + 120  # created can take ~15 s (model reset)
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type", "")
                    if mtype == "response.output.delta":
                        kind = msg.get("kind")
                        metrics = msg.get("metrics") or {}
                        kv = metrics.get("kv_cache_length", kv)
                        if kind == "audio":
                            pcm = np.frombuffer(base64.b64decode(msg["audio"]), dtype=np.float32)
                            self._sink.on_model_audio(pcm)
                        elif kind == "text":
                            self._sink.on_text(msg.get("text", ""))
                        elif kind == "listen":
                            self._sink.on_listen()
                    elif mtype in ("session.queue_done", "queue_done"):
                        await ws.send(json.dumps({
                            "type": "session.init",
                            "payload": {"system_prompt": cfg.system_prompt,
                                        "config": cfg.session_config()},
                        }))
                    elif mtype == "session.created":
                        t_created = time.monotonic()
                        sender = asyncio.create_task(self._sender(ws))
                        self._set_ready(True)
                        logger.info("session ready (server mode=%s)", msg.get("mode"))
                    elif mtype in ("session.queued", "session.queue_update"):
                        logger.info("queued at position %s", msg.get("position"))
                    elif mtype == "session.closed":
                        logger.info("session closed: %s (kv=%d)", msg.get("reason"), kv)
                        return
                    elif mtype == "error":
                        raise RuntimeError(f"server error: {msg.get('error')}")

                    if not t_created and time.monotonic() > deadline:
                        raise TimeoutError("no session.created within 120 s")
                    if t_created and not rotating:
                        reason = should_rotate(cfg, kv, time.monotonic() - t_created,
                                               self._sink.quiet())
                        if reason:
                            rotating = True
                            self._set_ready(False)  # stop uplink before closing
                            logger.info("rotating session (%s)", reason)
                            await ws.send(json.dumps(
                                {"type": "session.close", "reason": "rotation"}))
            finally:
                if sender is not None:
                    sender.cancel()

    async def _sender(self, ws) -> None:
        assert self._q is not None
        while True:
            msg = await self._q.get()
            await ws.send(msg)

    def _set_ready(self, ready: bool) -> None:
        if ready != self._ready:
            self._ready = ready
            self._sink.on_ready(ready)
