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
        # rolling 5 s telemetry so "no response" is diagnosable from one run:
        # is the mic reaching us (uplink rms), and what is the server sending back?
        self._stats = {"up": 0, "up_rms": 0.0, "up_rms_max": 0.0, "drop": 0, "text": 0, "audio": 0, "listen": 0, "done": 0, "barge": 0}
        self._seen_other: set = set()  # unrecognized event (type, kind) — logged once each
        self._clean_close = False      # last session ended via a server session.closed
        # Diagnostics: dump exactly what we send to the model (raw s16le mono 16 kHz)
        # so "the robot doesn't reply" can be split into server-side vs our-audio-side.
        self._dump_f = None
        if getattr(cfg, "omni_dump_uplink", ""):
            try:
                self._dump_f = open(cfg.omni_dump_uplink, "ab")
                log.info("omni: dumping uplink audio to %s — play with: "
                         "ffplay -f s16le -ar 16000 -i %s",
                         cfg.omni_dump_uplink, cfg.omni_dump_uplink)
            except Exception as e:
                log.warning("omni: cannot open OMNI_DUMP_UPLINK file: %s", e)

    # -- public API (called from other threads) -----------------------------
    def set_frame_source(self, fn: FrameSource) -> None:
        self._frame_source = fn

    def submit_audio_chunk(self, pcm: np.ndarray) -> None:
        """Enqueue a ~1 s mono float32 16 kHz chunk (drops oldest if backed up)."""
        if self._dump_f is not None:
            try:
                self._dump_f.write(
                    (np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0) * 32767.0)
                    .astype("<i2").tobytes()
                )
            except Exception:
                pass
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
            clean = False
            try:
                log.info("omni: connecting to %s", url)
                async with websockets.connect(
                    url, ssl=ssl_ctx, max_size=None, open_timeout=10,
                    ping_interval=20, ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    clean = await self._session(ws)
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as e:
                log.warning("omni: connection error: %s", e)
                try:
                    self.sink.on_disconnected(str(e))
                except Exception:
                    pass
            self._ws = None
            if clean:
                # A clean server close (e.g. the gateway's ~5 min session cap) is expected;
                # reset turn state and reconnect right away so the gap is barely noticeable.
                try:
                    self.sink.on_disconnected("reconnecting")
                except Exception:
                    pass
            if self.bus.stop_event.is_set():
                break
            await asyncio.sleep(0.1 if clean else backoff)

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
        self._clean_close = False
        ready = asyncio.Event()

        sender = asyncio.create_task(self._sender(ws, ready))
        receiver = asyncio.create_task(self._receiver(ws, ready))
        stats = asyncio.create_task(self._stats_loop())
        try:
            # Whichever finishes first (sender returns on stop; receiver on ws close)
            # tears the session down; the others are cancelled below.
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (sender, receiver, stats):
                t.cancel()
            for t in (sender, receiver, stats):
                try:
                    await t
                except Exception:
                    pass
        return self._clean_close

    async def _stats_loop(self) -> None:
        """Every 5 s, log uplink chunk count + mean RMS and the downlink event mix."""
        import asyncio

        while not self.bus.stop_event.is_set():
            await asyncio.sleep(5.0)
            s = self._stats
            n = s["up"]
            playq = self.bus.tts_audio.qsize()  # unplayed audio backlog (→ choppy if it grows)
            if n or s["text"] or s["audio"] or s["listen"] or s["done"]:
                log.info(
                    "omni 5s: uplink=%d chunks (mic rms~%.4f peak~%.4f, dropped=%d, force_listen=%d) | downlink text=%d audio=%d listen=%d done=%d | playq=%d",
                    n, (s["up_rms"] / n if n else 0.0), s["up_rms_max"], s["drop"], s["barge"], s["text"], s["audio"], s["listen"], s["done"], playq,
                )
                if playq > 25:
                    # Long replies stream in bursts (generation outruns real-time playback),
                    # so a transient backlog is expected; it is dropped wholesale on a
                    # barge-in. Only a backlog that keeps GROWING across windows would mean
                    # playback is starved.
                    log.info("omni: %d reply chunks queued ahead of playback (long reply "
                             "streaming in a burst — normal; dropped instantly on barge-in)", playq)
            for k in s:
                s[k] = 0

    async def _sender(self, ws, ready) -> None:
        import asyncio

        # Wait for session.created before streaming audio. The gateway queues sessions
        # and the backend lazy-loads the model on first use (10–60 s), so wait generously.
        ready_s = max(1.0, float(self.cfg.omni_session_ready_s))
        try:
            await asyncio.wait_for(ready.wait(), timeout=ready_s)
        except asyncio.TimeoutError:
            log.warning("omni: no session.created within %.0fs; sending anyway", ready_s)

        only_fd = self.cfg.omni_mode == "full_duplex"
        every_n = max(1, int(self.cfg.omni_video_every_n))
        send_video = self.cfg.omni_video_active
        sent = 0
        while not self.bus.stop_event.is_set():
            chunk = await self._next_chunk()
            if chunk is None:
                continue
            if not only_fd:
                # turn_based has no continuous mic push path here.
                continue
            # Attach a frame only every Nth chunk — sending vision every second can push
            # the server past real time (→ backlog / choppy speech). Never in audio mode.
            want_frame = self._frame_source and send_video and (sent % every_n == 0)
            frame_b64 = self._frame_source() if want_frame else None
            # Barge-in: tell the SERVER to stop its current turn and listen. Keyed on
            # interrupt_event (set only on a real barge-in, never at idle) so it can NEVER
            # stick on — keying it on user_speaking made force_listen permanent whenever the
            # local VAD got stuck "in speech", which forced the model to listen forever and it
            # never replied.
            force_listen = self.bus.interrupt_event.is_set()
            msg = protocol.build_input_append(chunk, frame_b64=frame_b64, force_listen=force_listen)
            await ws.send(json.dumps(msg))
            sent += 1
            self._stats["up"] += 1
            if force_listen:
                self._stats["barge"] += 1
            if len(chunk):
                rms = float(np.sqrt(np.mean(np.asarray(chunk, dtype=np.float64) ** 2)))
                self._stats["up_rms"] += rms
                self._stats["up_rms_max"] = max(self._stats["up_rms_max"], rms)

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
                self._stats["drop"] += dropped
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
                # Only an expected close (the gateway's ~5 min session cap = "timeout", or a
                # graceful close) is "clean" → fast reconnect. An error close (backend_error,
                # etc.) means the server side is unhealthy; fast-reconnecting would just hammer
                # a failing backend, so treat it as non-clean and back off (omni_reconnect_s).
                if evt.reason in ("timeout", "closed", "session_ended", None, ""):
                    self._clean_close = True
                    log.info("omni: gateway session cap reached (%s) — reconnecting", evt.reason)
                else:
                    self._clean_close = False
                    log.warning("omni: session.closed (%s) — backing off before reconnect", evt.reason)
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
                self._stats["text"] += 1
                self.sink.on_text(evt.text)
            elif cat == protocol.EV_AUDIO:
                if evt.audio is not None and len(evt.audio):
                    self._stats["audio"] += 1
                    self.sink.on_audio(evt.audio)
            elif cat == protocol.EV_LISTEN:
                self._stats["listen"] += 1
                self.sink.on_listen()
            elif cat == protocol.EV_DONE:
                self._stats["done"] += 1
                rid = evt.response_id or "_"
                full = self._turn_text.pop(rid, "") or evt.text
                if evt.audio is not None and len(evt.audio):
                    self.sink.on_audio(evt.audio)  # turn_based TTS returns audio here
                self.sink.on_turn_done(full)
            elif cat == protocol.EV_STATUS:
                # gateway control frame (e.g. session.queued) — surface, keep waiting
                if evt.status == "session.queued":
                    pos = evt.raw.get("position")
                    wait = evt.raw.get("estimated_wait_s")
                    log.info("omni: queued at the gateway (position=%s, ~%ss)", pos, wait)
                else:
                    log.debug("omni: status %s", evt.status)
            elif cat == protocol.EV_ERROR:
                log.warning("omni: server error (%s): %s", evt.reason, evt.message)
            elif cat == protocol.EV_OTHER:
                # Surface unrecognized frames once each — if the audio path replies with
                # a differently-named event, this reveals it instead of silently dropping.
                key = (evt.raw.get("type"), evt.raw.get("kind"))
                if key not in self._seen_other:
                    self._seen_other.add(key)
                    log.info("omni: unhandled server event type=%r kind=%r (keys=%s)",
                             key[0], key[1], list(evt.raw.keys()))
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
