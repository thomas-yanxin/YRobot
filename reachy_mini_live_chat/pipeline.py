"""Orchestrator: wires the mic, camera, remote omni brain, and motion via the Bus.

This is deliberately thin — the omni model does ASR + reasoning + TTS end-to-end, so the
robot side just moves bytes and bodies:

* :class:`~reachy_mini_live_chat.audio.io.AudioEngine` streams 1 s mic chunks up and plays
  the returned speech.
* :class:`~reachy_mini_live_chat.omni.video.VideoGrabber` attaches one current frame per chunk.
* :class:`~reachy_mini_live_chat.omni.client.OmniClient` runs the full-duplex WebSocket and
  calls back into this object (the *sink*) with text / audio / listen / turn-done events.
* :class:`~reachy_mini_live_chat.motion.MotionController` (unchanged) owns ``set_target`` at
  100 Hz: DOA head-turn while the user talks, conversation-state moods, and emotion gestures
  inferred from the transcript — every command clamped to the safe joint range.

The conversation state is a blend of a **local VAD** (``user_speaking`` → LISTENING, DOA) and
the **omni events** (text/audio → SPEAKING; listen/done → IDLE).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .audio.io import AudioEngine
from .bus import Bus, ConvState, MotionIntent
from .config import Config
from .motion import MotionController
from .motion.emotions import map_intent
from .omni.client import OmniClient
from .omni.video import VideoGrabber
from .text_utils import clean_spoken, detect_lang

log = logging.getLogger("live_chat.pipeline")


class Pipeline:
    def __init__(self, mini, cfg: Config, bus: Optional[Bus] = None) -> None:
        self.mini = mini
        self.cfg = cfg
        self.bus = bus or Bus()

        self.motion = MotionController(mini, cfg, bus=self.bus)
        self.video = VideoGrabber(cfg, mini)
        self.omni = OmniClient(cfg, self.bus, sink=self)
        self.omni.set_frame_source(self.video.latest_b64)
        self.audio = AudioEngine(mini, cfg, self.bus, on_audio_chunk=self.omni.submit_audio_chunk)

        self._speaking = False          # is the robot mid-turn (speaking)?
        self._turn_text = ""            # transcript of the current spoken turn
        # Barge-in cancels the WHOLE reply, not just "audio while the user talks":
        # the omni server streams long replies in bursts (often 10+ s in flight), and
        # interrupt_event clears as soon as the user stops talking — without this
        # latch the rest of the old reply would resume playing a beat later.
        self._discard_turn = False
        self.bus.subscribe(self._on_bus_event)

    def _on_bus_event(self, evt: dict) -> None:
        if evt.get("kind") == "interrupt":
            self._discard_turn = True

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        try:
            self.mini.wake_up()
        except Exception:
            pass
        self.motion.start()
        self.video.start()
        self.audio.start()
        self.omni.start()
        log.info("pipeline started (omni @ %s)", self.cfg.omni_backend_url)

    def shutdown(self) -> None:
        self.bus.stop_event.set()
        self.bus.tts_audio.put(None)
        self.omni.join()
        self.audio.join()
        self.video.stop()
        self.motion.join()
        try:
            self.mini.goto_sleep()
        except Exception:
            pass

    # -- web UI: type-to-talk is voice-driven in full-duplex omni -----------
    def inject_text(self, text: str) -> None:
        """Full-duplex omni is a live voice channel — there's no text-turn injection.

        We surface the typed text in the transcript so the UI stays useful, but the robot
        replies to *spoken* input only.
        """
        text = (text or "").strip()
        if not text:
            return
        self.bus.emit("system", {"text": "文本输入在全双工语音模式下不发送给模型（请直接说话）。"})
        self.bus.emit("user", {"text": text, "lang": detect_lang(text)})

    # ======================================================================
    # OmniClient sink — called from the omni asyncio thread
    # ======================================================================
    def on_connected(self) -> None:
        self.bus.emit("system", {"text": "omni connected"})
        # a small welcoming gesture (no spoken greeting: TTS is the model's job)
        self._enqueue_emotion("greeting")

    def on_disconnected(self, reason: str) -> None:
        self.bus.emit("system", {"text": f"omni disconnected: {reason}"})
        self._discard_turn = False
        self._end_turn(reset_state=True)

    def on_text(self, text: str) -> None:
        if self._discard_turn or self.bus.interrupt_event.is_set():
            return  # barge-interrupted reply (or barge still in progress) — drop it
        self._begin_speaking()
        shown = clean_spoken(text)
        if shown:
            self._turn_text += text
            self.bus.emit("assistant", {"text": shown, "lang": detect_lang(self._turn_text)})

    def on_audio(self, pcm: np.ndarray) -> None:
        # Queue raw omni audio (float32 at omni_out_sr); the playback thread resamples to
        # the device rate. Keeping the WS receive thread free of resampling avoids choppy
        # speech on the CM4.
        if self._discard_turn or self.bus.interrupt_event.is_set():
            return  # barge-interrupted reply (or barge still in progress) — drop it
        self._begin_speaking()
        lat = self.bus.lat
        if "t_end" in lat and "recv" not in lat:
            lat["recv"] = time.monotonic()  # first reply audio back from the server
        arr = np.asarray(pcm, dtype=np.float32)
        if len(arr):
            self.bus.tts_audio.put(arr)

    def on_listen(self) -> None:
        # This omni server emits listen/done at its ~1 Hz step cadence, NOT once per
        # conversational turn (verified on hardware: done≈1/s while speaking). During
        # an active barge-in every force_listen chunk produces a listen step — if those
        # cleared the discard latch, the old reply's still-streaming audio would resume
        # the moment one slipped in ("pauses, then keeps talking"). Only a boundary
        # seen AFTER the user finished their barge utterance ends the discard.
        if self.bus.interrupt_event.is_set():
            return
        self._discard_turn = False
        if self._speaking:
            self._end_turn()

    def on_turn_done(self, full_text: str) -> None:
        if self.bus.interrupt_event.is_set():
            # step boundary in the middle of a barge-in: keep discarding
            self._end_turn()
            return
        discarded = self._discard_turn
        self._discard_turn = False
        if discarded:
            # The turn was barge-interrupted: no gesture for words never spoken.
            self._end_turn()
            return
        # Match a body gesture to what was just said (bilingual keyword cues).
        text = (full_text or self._turn_text).strip()
        if text:
            emo = map_intent(text, detect_lang(text))
            if emo:
                self._enqueue_emotion(emo)
        self._end_turn()

    # -- state helpers ------------------------------------------------------
    def _begin_speaking(self) -> None:
        # Only reached with interrupt_event clear (on_text/on_audio gate on it), so a
        # new reply never has to clear the flag itself — clearing here would race a
        # barge-in that fired between the gate and this call. Stale-flag recovery
        # lives in the playback loop's safety net + the VAD's speech-end hook.
        if not self._speaking:
            self._speaking = True
            self._turn_text = ""
            self.bus.set_state(ConvState.SPEAKING)

    def _end_turn(self, reset_state: bool = False) -> None:
        was_speaking = self._speaking
        self._speaking = False
        self._turn_text = ""
        if was_speaking:
            self.bus.tts_audio.put(None)  # flush sentinel → playback marks end-of-turn
        # Return to LISTENING if the human is mid-sentence, else IDLE.
        if reset_state or self.bus.state in (ConvState.SPEAKING, ConvState.INTERRUPTED):
            self.bus.set_state(
                ConvState.LISTENING if self.bus.user_speaking.is_set() else ConvState.IDLE
            )

    def _enqueue_emotion(self, name: str) -> None:
        if self.cfg.enable_motion:
            self.bus.motion_intents.put(MotionIntent(kind="emotion", emotion=name))
