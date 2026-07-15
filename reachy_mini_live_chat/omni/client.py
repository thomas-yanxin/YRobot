"""``OmniClient`` — full-duplex WebSocket client for the remote omni brain.

Runs an asyncio event loop on its own thread. The capture thread pushes 1 s float32
16 kHz chunks via :meth:`submit_audio_chunk`; a **sender** task drains them, attaches
the current video frame, and sends ``input.append``. A **receiver** task reads server
events and calls the ``sink`` (text / audio / listen / turn-done). The connection
self-heals: on any drop it reports ``on_disconnected`` and reconnects with backoff.

The sink is duck-typed — the orchestrator implements:
``on_connected() · on_disconnected(reason) · on_text(str) · on_audio(np.ndarray) ·
on_listen() · on_turn_done(full_text)``.
"""
from __future__ import annotations

import json
import logging
import queue
import ssl
import threading
from typing import Callable, Optional

import numpy as np

from ..config import Config
from . import protocol

log = logging.getLogger("live_chat.omni.client")

FrameSource = Callable[[], Optional[str]]


class OmniClient:
    def __init__(self, cfg: Config, bus, sink) -> None:
        self.cfg = cfg
        self.bus = bus
        self.sink = sink
        self._frame_source: Optional[FrameSource] = None

        # 1 s chunks from the capture thread → sender task.
        self._audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self._thread: Optional[threading.Thread] = None
        self._loop = None
        self._ws = None
        # per-response accumulated text, keyed by response_id
        self._turn_text: dict[str, str] = {}

    # -- public API (called from other threads) -----------------------------
    def set_frame_source(self, fn: FrameSource) -> None:
        self._frame_source = fn

    def submit_audio_chunk(self, pcm: np.ndarray) -> None:
        """Enqueue a ~1 s mono float32 16 kHz chunk (drops oldest if backed up)."""
        try:
            self._audio_q.put_nowait(pcm)
        except queue.Full:
            try:
                self._audio_q.get_nowait()
                self._audio_q.put_nowait(pcm)
            except queue.Empty:
                pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._thread_main, name="omni", daemon=True)
        self._thread.start()

    def join(self, timeout: float = 1.5) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    # -- thread / event loop ------------------------------------------------
    def _thread_main(self) -> None:
        import asyncio

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as e:  # pragma: no cover - defensive
            log.exception("omni client crashed: %s", e)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    def _ssl_context(self):
        url = self.cfg.omni_backend_url
        if not url.startswith("wss://"):
            return None
        ctx = ssl.create_default_context()
        if self.cfg.omni_tls_insecure:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def _run(self) -> None:
        import asyncio

        try:
            import websockets
        except Exception as e:  # pragma: no cover - dependency guard
            log.error("`websockets` not installed (%s); omni brain unavailable", e)
            return

        url = self.cfg.omni_backend_url
        ssl_ctx = self._ssl_context()
        backoff = self.cfg.omni_reconnect_s

        while not self.bus.stop_event.is_set():
            try:
                log.info("omni: connecting to %s", url)
                async with websockets.connect(
                    url, ssl=ssl_ctx, max_size=None, open_timeout=10,
                    ping_interval=20, ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    await self._session(ws)
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as e:
                log.warning("omni: connection error: %s", e)
                try:
                    self.sink.on_disconnected(str(e))
                except Exception:
                    pass
            self._ws = None
            if self.bus.stop_event.is_set():
                break
            await asyncio.sleep(backoff)

    async def _session(self, ws) -> None:
        import asyncio

        init = protocol.build_session_init(
            mode=self.cfg.omni_mode,
            use_tts=self.cfg.omni_use_tts,
            system_prompt=self.cfg.omni_system_prompt,
            ref_audio_b64=_load_ref_audio_b64(self.cfg.omni_voice_ref),
            config=self.cfg.omni_sampling_config(),
        )
        await ws.send(json.dumps(init))
        self._turn_text.clear()
        ready = asyncio.Event()

        sender = asyncio.create_task(self._sender(ws, ready))
        receiver = asyncio.create_task(self._receiver(ws, ready))
        try:
            # Whichever finishes first (sender returns on stop; receiver on ws close)
            # tears the session down; the other is cancelled below.
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (sender, receiver):
                t.cancel()
            for t in (sender, receiver):
                try:
                    await t
                except Exception:
                    pass

    async def _sender(self, ws, ready) -> None:
        import asyncio

        # Wait for session.created (or a short grace) before streaming audio.
        try:
            await asyncio.wait_for(ready.wait(), timeout=10)
        except asyncio.TimeoutError:
            log.warning("omni: no session.created within 10s; sending anyway")

        only_fd = self.cfg.omni_mode == "full_duplex"
        while not self.bus.stop_event.is_set():
            chunk = await self._next_chunk()
            if chunk is None:
                continue
            if not only_fd:
                # turn_based has no continuous mic push path here.
                continue
            frame_b64 = self._frame_source() if (self._frame_source and self.cfg.omni_send_video) else None
            msg = protocol.build_input_append(chunk, frame_b64=frame_b64)
            await ws.send(json.dumps(msg))

    async def _next_chunk(self):
        """Await the next audio chunk, coalescing a backlog to stay real-time."""
        import asyncio

        while not self.bus.stop_event.is_set():
            try:
                chunk = self._audio_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            # If more than a couple are queued we're behind: keep only the newest.
            dropped = 0
            while self._audio_q.qsize() > 1:
                try:
                    chunk = self._audio_q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped:
                log.debug("omni: dropped %d stale audio chunk(s) to catch up", dropped)
            return chunk
        return None

    async def _receiver(self, ws, ready) -> None:
        async for raw in ws:
            if not isinstance(raw, (str, bytes)):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            evt = protocol.parse_event(msg)
            self._dispatch(evt, ready)
            if evt.category == protocol.EV_CLOSED:
                log.info("omni: session.closed (%s)", evt.reason)
                break

    # -- event → sink -------------------------------------------------------
    def _dispatch(self, evt: "protocol.OmniEvent", ready) -> None:
        cat = evt.category
        try:
            if cat == protocol.EV_CREATED:
                ready.set()
                log.info("omni: session.created (mode=%s)", evt.mode)
                self.sink.on_connected()
            elif cat == protocol.EV_TEXT:
                rid = evt.response_id or "_"
                self._turn_text[rid] = self._turn_text.get(rid, "") + evt.text
                self.sink.on_text(evt.text)
            elif cat == protocol.EV_AUDIO:
                if evt.audio is not None and len(evt.audio):
                    self.sink.on_audio(evt.audio)
            elif cat == protocol.EV_LISTEN:
                self.sink.on_listen()
            elif cat == protocol.EV_DONE:
                rid = evt.response_id or "_"
                full = self._turn_text.pop(rid, "") or evt.text
                if evt.audio is not None and len(evt.audio):
                    self.sink.on_audio(evt.audio)  # turn_based TTS returns audio here
                self.sink.on_turn_done(full)
        except Exception as e:  # pragma: no cover - sink must never kill the loop
            log.debug("omni sink error on %s: %s", cat, e)


def _load_ref_audio_b64(path: str) -> str:
    """Load a reference WAV for voice-cloning as base64 float32 PCM (best-effort)."""
    if not path:
        return ""
    try:
        import wave

        with wave.open(path, "rb") as w:
            n = w.getnframes()
            sw = w.getsampwidth()
            ch = w.getnchannels()
            raw = w.readframes(n)
        if sw == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        else:
            log.warning("omni voice ref: unsupported sample width %d; ignoring", sw)
            return ""
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        return protocol.pcm_f32_to_b64(data)
    except Exception as e:
        log.warning("omni voice ref: could not load %s (%s); ignoring", path, e)
        return ""
