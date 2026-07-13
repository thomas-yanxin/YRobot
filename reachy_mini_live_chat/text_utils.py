"""Small, dependency-free text helpers shared across stages.

* ``detect_lang`` — cheap zh/en decision (drives voice choice + motion language).
* ``ClauseAccumulator`` — turns a stream of LLM token deltas into speakable clauses,
  flushing at sentence/clause boundaries (or a soft length cap) so TTS can start
  before the full reply is generated — the key to low first-audio latency.
"""
from __future__ import annotations

import re
from typing import List, Optional

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ebef]")
# terminal + strong clause punctuation in both languages
_BOUNDARY = "。！？!?…\n；;：:，,"
_HARD_BOUNDARY = "。！？!?…\n"


def detect_lang(text: str) -> str:
    """Return 'zh' if the text is meaningfully Chinese, else 'en'."""
    if not text:
        return "en"
    cjk = len(_CJK.findall(text))
    letters = sum(ch.isalpha() and ord(ch) < 128 for ch in text)
    if cjk == 0:
        return "en"
    # any non-trivial amount of CJK -> treat as Chinese (handles code-switch)
    return "zh" if cjk * 2 >= letters or cjk >= 2 else "en"


def strip_control_tags(text: str) -> str:
    """Remove inline motion tags like ``<emo>cheerful1</emo>`` from spoken text."""
    return re.sub(r"<emo>.*?</emo>", "", text).strip()


# order matters: match the name-carrying "<emo>yes1" before the bare "<emo>" tag
_SPOKEN_JUNK = re.compile(r"<emo>[A-Za-z0-9_]*|</?(?:emo|think)>|[*`#]+")


def clean_spoken(text: str) -> str:
    """Scrub a clause before TTS: drop stray control tags and markdown the model may leak.

    Small models don't always honor the "no markdown / one clean <emo> tag" instruction, so we
    strip any residual ``<emo…>`` / ``<think>`` fragments and ``* ` #`` markdown from what we speak.
    """
    return _SPOKEN_JUNK.sub("", text).strip()


class ClauseAccumulator:
    """Feed token deltas; get back speakable clauses as they complete.

    A clause is emitted when a boundary char is seen and enough characters have
    accumulated, or when a soft cap is exceeded (so a long comma-free sentence still
    starts speaking). ``flush()`` returns whatever remains at end-of-stream.
    """

    def __init__(self, min_chars: int = 6, soft_cap: int = 60) -> None:
        self.min_chars = min_chars
        self.soft_cap = soft_cap
        self._buf = ""

    def push(self, delta: str) -> List[str]:
        out: List[str] = []
        self._buf += delta
        while True:
            clause = self._take()
            if clause is None:
                break
            if clause.strip():
                out.append(clause.strip())
        return out

    def _take(self) -> Optional[str]:
        # hard boundary anywhere -> cut there
        for i, ch in enumerate(self._buf):
            if ch in _HARD_BOUNDARY and i + 1 >= self.min_chars:
                cut = self._buf[: i + 1]
                self._buf = self._buf[i + 1 :]
                return cut
        # soft boundary once past min_chars
        if len(self._buf) >= self.min_chars:
            for i, ch in enumerate(self._buf):
                if ch in _BOUNDARY and i + 1 >= self.min_chars:
                    cut = self._buf[: i + 1]
                    self._buf = self._buf[i + 1 :]
                    return cut
        # soft cap: flush at last space (en) or just cut (zh)
        if len(self._buf) >= self.soft_cap:
            sp = self._buf.rfind(" ", self.min_chars)
            idx = sp if sp > 0 else self.soft_cap
            cut = self._buf[:idx]
            self._buf = self._buf[idx:]
            return cut
        return None

    def flush(self) -> Optional[str]:
        rest, self._buf = self._buf.strip(), ""
        return rest or None
