"""Full-duplex orchestrator: wires VAD/ASR/LLM/TTS/vision/motion via the Bus.

Threads:
* AudioEngine's capture + playback threads (mic <-> speaker).
* MotionController's 100 Hz control loop.
* **brain** — this module's worker: consumes finalized utterances, runs ASR (if needed),
  gates vision, streams the LLM into clauses, synthesizes TTS, and drives motion moods.
  One turn at a time; a barge-in sets ``interrupt_event`` and the brain abandons the reply
  and picks up the next turn.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .asr import Asr
from .audio.io import AudioEngine
from .bus import Bus, ConvState, MotionIntent
from .config import Config
from .llm import LlmEngine
from .motion import MotionController
from .text_utils import detect_lang
from .tts import TtsEngine
from .vision import VisionGate

log = logging.getLogger("live_chat.pipeline")


class Pipeline:
    def __init__(self, mini, cfg: Config, bus: Optional[Bus] = None) -> None:
        self.mini = mini
        self.cfg = cfg
        self.bus = bus or Bus()

        self.asr = Asr(cfg)
        self.llm = LlmEngine(cfg)
        self.tts = TtsEngine(cfg)
        self.vision = VisionGate(cfg, mini)
        self.motion = MotionController(mini, cfg, bus=self.bus)
        self.audio = AudioEngine(mini, cfg, self.bus, on_utterance_pcm=self._on_utterance)

        # ('audio', pcm, t_end) or ('text', text, t_end)
        self._turns: "queue.Queue[Tuple[str, object, float]]" = queue.Queue()
        self._brain: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        try:
            self.mini.wake_up()
        except Exception:
            pass
        self.motion.start()
        self.audio.start()
        self._brain = threading.Thread(target=self._brain_loop, name="brain", daemon=True)
        self._brain.start()
        self._greet()
        log.info("pipeline started")

    def shutdown(self) -> None:
        self.bus.stop_event.set()
        self.bus.tts_audio.put(None)
        self.audio.join()
        self.motion.join()
        if self._brain:
            self._brain.join(timeout=1.0)
        try:
            self.mini.goto_sleep()
        except Exception:
            pass

    # -- inputs -------------------------------------------------------------
    def _on_utterance(self, pcm: np.ndarray, t_end: float) -> None:
        self._turns.put(("audio", pcm, t_end))

    def inject_text(self, text: str) -> None:
        """Feed a typed message as if it were a spoken turn (web UI / stub demo)."""
        if text and text.strip():
            self._turns.put(("text", text.strip(), time.monotonic()))

    # -- brain --------------------------------------------------------------
    def _brain_loop(self) -> None:
        while not self.bus.stop_event.is_set():
            try:
                kind, payload, t_end = self._turns.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._handle_turn(kind, payload, t_end)
            except Exception as e:
                log.exception("turn failed: %s", e)
                self.bus.set_state(ConvState.IDLE)

    def _handle_turn(self, kind: str, payload, t_end: float) -> None:
        self.bus.clear_interrupt()
        self.bus.next_turn()
        self.bus.set_state(ConvState.THINKING)

        # 1) transcribe
        if kind == "audio":
            text, lang = self.asr.transcribe(payload)
        else:
            text = str(payload)
            lang = self.cfg.lang if self.cfg.lang != "auto" else detect_lang(text)
        text = (text or "").strip()
        if not text:
            self.bus.set_state(ConvState.IDLE)
            return
        self.bus.emit("user", {"text": text, "lang": lang})
        log.info("user(%s): %s", lang, text)

        # 2) latency clock starts now (end of user speech)
        self.bus.pending_t_end = t_end
        self.bus.pending_measured = False

        # 3) vision (token-gated)
        keyframe = self.vision.maybe_keyframe(text)

        # 4) stream reply -> clauses -> TTS; drive motion
        def stop_check() -> bool:
            return self.bus.interrupt_event.is_set() or self.bus.stop_event.is_set()

        emitted_emotion = False
        first_clause = True
        full_text = ""
        for evt in self.llm.respond(text, lang, keyframe_b64=keyframe, stop_check=stop_check):
            if stop_check():
                break
            if evt["type"] == "emotion":
                self._enqueue_emotion(evt["name"])
                emitted_emotion = True
            elif evt["type"] == "clause":
                if first_clause:
                    self.bus.set_state(ConvState.SPEAKING)
                    self.bus.motion_intents.put(MotionIntent(kind="mood", mood="speaking"))
                    if not emitted_emotion:
                        # no explicit tag from the model -> infer from content/language
                        from .motion.emotions import map_intent

                        inferred = map_intent(full_text or text, lang)
                        if inferred:
                            self._enqueue_emotion(inferred)
                    first_clause = False
                self.bus.emit("assistant", {"text": evt["text"], "lang": lang})
                self._speak_clause(evt["text"], lang, stop_check)
            elif evt["type"] == "final":
                full_text = evt["text"]

        # 5) end of answer
        self.bus.tts_audio.put(None)  # sentinel
        if not stop_check():
            self._wait_until_quiet()
            self.bus.set_state(ConvState.IDLE)

    def _speak_clause(self, clause: str, lang: str, stop_check) -> None:
        for chunk in self.tts.synthesize(clause, lang, stop_check=stop_check):
            if stop_check():
                return
            self.bus.tts_audio.put(chunk)

    def _enqueue_emotion(self, name: str) -> None:
        if self.cfg.enable_motion:
            self.bus.motion_intents.put(MotionIntent(kind="emotion", emotion=name))

    def _wait_until_quiet(self, timeout: float = 15.0) -> None:
        t0 = time.monotonic()
        # give playback a moment to start, then wait for it to finish
        time.sleep(0.1)
        while self.bus.robot_speaking.is_set() and not self.bus.interrupt_event.is_set():
            if time.monotonic() - t0 > timeout:
                break
            time.sleep(0.05)

    # -- greeting -----------------------------------------------------------
    def _greet(self) -> None:
        lang = "en" if self.cfg.lang == "en" else "zh"
        text = "你好！我在听，随时可以聊。" if lang == "zh" else "Hi! I'm listening — talk to me anytime."
        self.bus.emit("assistant", {"text": text, "lang": lang})
        self._enqueue_emotion("welcoming1")
        self.bus.set_state(ConvState.SPEAKING)

        def _run():
            for chunk in self.tts.synthesize(text, lang):
                if self.bus.stop_event.is_set():
                    return
                self.bus.tts_audio.put(chunk)
            self.bus.tts_audio.put(None)
            self._wait_until_quiet()
            self.bus.set_state(ConvState.IDLE)

        threading.Thread(target=_run, name="greet", daemon=True).start()
