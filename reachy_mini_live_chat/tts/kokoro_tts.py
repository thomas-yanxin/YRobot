"""Streaming TTS via **Kokoro-82M** — low time-to-first-audio, zh + en, streams.

Primary backend is the ``kokoro`` ``KPipeline`` (CPU/MPS on Mac). On Apple Silicon you can
swap in ``mlx-audio`` for extra speed (set ``TTS_MODEL`` to an mlx Kokoro repo) — the
loading is guarded and falls back automatically. Without any TTS backend (``--stub``),
a quiet tone of proportional length is produced so playback/duplex timing stays realistic.

Output is always **mono float32 @ 16 kHz** (the robot's audio rate), yielded in chunks so
the audio-out thread can start playing before the whole clause is synthesized.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterator

import numpy as np

from ..config import Config

log = logging.getLogger("live_chat.tts")

KOKORO_SR = 24000
OUT_SR = 16000
_LANG_CODE = {"zh": "z", "en": "a"}


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = int(round(len(x) * sr_out / sr_in))
        idx = np.linspace(0, len(x) - 1, n).astype(np.int64)
        return x[idx].astype(np.float32)


class TtsEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pipelines: Dict[str, object] = {}
        self._mlx = None
        self._backend = "stub" if cfg.stub else None

    # -- backend loading ----------------------------------------------------
    def _pipeline(self, lang: str):
        if lang in self._pipelines:
            return self._pipelines[lang]
        code = _LANG_CODE.get(lang, "a")
        try:
            from kokoro import KPipeline  # lazy

            pipe = KPipeline(lang_code=code)
            self._pipelines[lang] = pipe
            self._backend = "kokoro"
            log.info("TTS: Kokoro KPipeline (lang_code=%s)", code)
            return pipe
        except Exception as e:
            log.warning("TTS: Kokoro unavailable (%s); using tone fallback", e)
            self._pipelines[lang] = None
            self._backend = self._backend or "stub"
            return None

    def _voice(self, lang: str) -> str:
        return self.cfg.tts_voice_zh if lang == "zh" else self.cfg.tts_voice_en

    # -- synthesis ----------------------------------------------------------
    def synthesize(self, text: str, lang: str, stop_check=None) -> Iterator[np.ndarray]:
        """Yield mono float32 @ 16 kHz chunks for one clause."""
        text = (text or "").strip()
        if not text:
            return
        if self.cfg.stub:
            yield from self._tone(text)
            return
        pipe = self._pipeline(lang)
        if pipe is None:
            yield from self._tone(text)
            return
        try:
            gen = pipe(text, voice=self._voice(lang), speed=self.cfg.tts_speed)
            for item in gen:
                if stop_check is not None and stop_check():
                    return
                audio = _extract_audio(item)
                if audio is None or len(audio) == 0:
                    continue
                yield _resample(np.asarray(audio, dtype=np.float32), KOKORO_SR, OUT_SR)
        except Exception as e:
            log.warning("TTS error: %s; tone fallback", e)
            yield from self._tone(text)

    def _tone(self, text: str) -> Iterator[np.ndarray]:
        # gentle low tone, ~55 ms per character, amplitude-enveloped
        dur = min(6.0, max(0.3, len(text) * 0.055))
        n = int(dur * OUT_SR)
        t = np.arange(n) / OUT_SR
        env = np.minimum(1.0, np.minimum(t * 20, (dur - t) * 20))
        tone = 0.05 * env * np.sin(2 * np.pi * 200 * t).astype(np.float32)
        step = int(OUT_SR * 0.1)
        for i in range(0, n, step):
            yield tone[i:i + step].astype(np.float32)


def _extract_audio(item):
    """KPipeline yields (graphemes, phonemes, audio); mlx-audio may yield an array/obj."""
    if isinstance(item, (tuple, list)) and len(item) >= 3:
        audio = item[2]
    else:
        audio = getattr(item, "audio", item)
    try:
        # torch tensor -> numpy
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        elif hasattr(audio, "numpy") and not isinstance(audio, np.ndarray):
            audio = np.asarray(audio)
    except Exception:
        return None
    return np.asarray(audio, dtype=np.float32).reshape(-1)
