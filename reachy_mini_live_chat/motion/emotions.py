"""Emotion library wrapper + bilingual intent→move mapping.

The omni model speaks end-to-end, so we can't ask it for an inline motion tag (that tag
would be read aloud). Instead :func:`map_intent` picks a move from bilingual keyword cues
in the model's *transcript* (zh + en), so the robot's body language still matches what it
just said, in either language.

Moves come from ``pollen-robotics/reachy-mini-emotions-library`` (authored within the safe
range). If the library can't load (offline), the controller falls back to a small
procedural gesture instead.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..config import Config

log = logging.getLogger("live_chat.emotions")

# Curated subset of the emotions library that maps cleanly to conversational moods.
ALLOWED_EMOTIONS = [
    "yes1", "no1", "curious1", "cheerful1", "laughing1", "surprised1", "sad1",
    "thoughtful1", "welcoming1", "proud1", "confused1", "attentive1", "grateful1",
    "loving1", "enthusiastic1", "calming1", "relief1", "uncertain1",
]

# Bilingual cue -> emotion move. Order matters (first match wins).
_INTENT_RULES = [
    ("yes1", ["是的", "对", "好的", "没错", "当然", "可以", "嗯", "yes", "sure", "of course", "okay", "ok", "right"]),
    ("no1", ["不", "不行", "不是", "别", "no", "nope", "don't", "can't", "cannot"]),
    ("grateful1", ["谢谢", "感谢", "多谢", "thank", "thanks", "appreciate"]),
    ("welcoming1", ["你好", "您好", "嗨", "早上好", "晚上好", "hello", "hi", "hey", "welcome", "good morning"]),
    ("laughing1", ["哈哈", "太好笑", "有意思", "haha", "lol", "funny", "hilarious"]),
    ("surprised1", ["哇", "天哪", "真的吗", "不会吧", "wow", "really", "no way", "amazing", "whoa"]),
    ("sad1", ["难过", "抱歉", "对不起", "遗憾", "sorry", "sad", "unfortunate", "afraid"]),
    ("proud1", ["搞定", "完成", "成功", "做到了", "done", "success", "finished", "nailed it"]),
    ("confused1", ["不懂", "没听清", "什么意思", "confused", "not sure what", "don't understand"]),
    ("curious1", ["为什么", "怎么", "什么", "吗", "?", "？", "why", "how", "what", "which", "who", "where"]),
    ("cheerful1", ["开心", "太好了", "棒", "great", "awesome", "nice", "cool", "good"]),
]


def map_intent(text: str, lang: str) -> Optional[str]:
    """Pick an emotion from keyword cues; returns None if nothing fits."""
    if not text:
        return None
    low = text.lower()
    for emo, cues in _INTENT_RULES:
        for c in cues:
            if c in low:
                return emo
    return None


class EmotionLibrary:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lib = None
        self._cache: dict = {}
        self._names: list[str] = list(ALLOWED_EMOTIONS)
        self._load()

    def _load(self) -> None:
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves  # lazy

            self._lib = RecordedMoves(self.cfg.emotions_dataset)
            try:
                self._names = self._lib.list_moves()
            except Exception:
                pass
            log.info("emotions: loaded %s (%d moves)", self.cfg.emotions_dataset, len(self._names))
        except Exception as e:
            log.warning("emotions: library unavailable (%s); procedural fallback", e)
            self._lib = None

    def available(self, name: str) -> bool:
        return name in self._names

    def get(self, name: str):
        """Return a RecordedMove for ``name`` or None (caller does a procedural gesture)."""
        if self._lib is None or not name:
            return None
        if name in self._cache:
            return self._cache[name]
        try:
            move = self._lib.get(name)
        except Exception as e:
            log.debug("emotions.get(%s) failed: %s", name, e)
            move = None
        self._cache[name] = move
        return move
