"""ASR via FunASR **SenseVoiceSmall** — fast, zh/en/code-switch, RTF ~0.007.

Loaded lazily so the package imports without ``funasr``. In ``--stub`` mode (no model)
``transcribe`` returns empty text; type-to-talk in the web UI drives the pipeline instead.
"""
from __future__ import annotations

import logging
import re
from typing import Tuple

import numpy as np

from ..config import Config
from ..text_utils import detect_lang

log = logging.getLogger("live_chat.asr")


class Asr:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None
        if not cfg.stub:
            self._load()

    def _load(self) -> None:
        try:
            from funasr import AutoModel  # lazy

            self._model = AutoModel(
                model=self.cfg.asr_model,
                device=self.cfg.asr_device,
                disable_update=True,
                log_level="ERROR",
            )
            log.info("ASR: loaded %s", self.cfg.asr_model)
        except Exception as e:
            log.warning("ASR: could not load %s (%s); transcription disabled", self.cfg.asr_model, e)
            self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, pcm: np.ndarray, sr: int = 16000) -> Tuple[str, str]:
        """Return (text, lang) for a mono float32 utterance."""
        if self._model is None:
            return "", self.cfg.lang if self.cfg.lang != "auto" else "zh"
        try:
            lang = "auto" if self.cfg.lang == "auto" else self.cfg.lang
            res = self._model.generate(
                input=pcm.astype(np.float32),
                cache={},
                language=lang,
                use_itn=True,
                batch_size_s=60,
            )
            text = _clean_sensevoice(res[0]["text"]) if res else ""
        except Exception as e:
            log.warning("ASR error: %s", e)
            return "", "zh"
        detected = self.cfg.lang if self.cfg.lang != "auto" else detect_lang(text)
        return text, detected


_TAG = re.compile(r"<\|[^|]*\|>")  # SenseVoice emits <|zh|><|EMO|>... markup


def _clean_sensevoice(text: str) -> str:
    return _TAG.sub("", text or "").strip()
