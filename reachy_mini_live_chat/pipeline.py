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
from typing import Optional

import numpy as np

from .audio.io import TARGET_SR, AudioEngine, _resample
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
        self._enqueue_emotion("welcoming1")

    def on_disconnected(self, reason: str) -> None:
        self.bus.emit("system", {"text": f"omni disconnected: {reason}"})
        self._end_turn(reset_state=True)

    def on_text(self, text: str) -> None:
        self._begin_speaking()
        shown = clean_spoken(text)
        if shown:
            self._turn_text += text
            self.bus.emit("assistant", {"text": shown, "lang": detect_lang(self._turn_text)})

    def on_audio(self, pcm: np.ndarray) -> None:
        self._begin_speaking()
        pcm16k = _resample(np.asarray(pcm, dtype=np.float32), self.cfg.omni_out_sr, TARGET_SR)
        if len(pcm16k):
            self.bus.tts_audio.put(pcm16k)

    def on_listen(self) -> None:
        # Model chose to listen this step: end any current spoken turn.
        if self._speaking:
            self._end_turn()

    def on_turn_done(self, full_text: str) -> None:
        # Match a body gesture to what was just said (bilingual keyword cues).
        text = (full_text or self._turn_text).strip()
        if text:
            emo = map_intent(text, detect_lang(text))
            if emo:
                self._enqueue_emotion(emo)
        self._end_turn()

    # -- state helpers ------------------------------------------------------
    def _begin_speaking(self) -> None:
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
