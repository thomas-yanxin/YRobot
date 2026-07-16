"""Emotion library wrapper + bilingual intent→move mapping.

The omni model speaks end-to-end, so we can't ask it for an inline motion tag (that tag
would be read aloud). Instead :func:`map_intent` picks an *intent* from bilingual keyword
cues in the model's transcript (zh + en), and :meth:`EmotionLibrary.resolve` turns the
intent into a concrete recorded move — choosing randomly among the clips that express it,
so the same reply doesn't always trigger the identical gesture.

Intents and the intent→clips table mirror the official ``reachy_mini_conversation_app``'s
``play_emotion`` tool (its clip names are the ground truth for
``pollen-robotics/reachy-mini-emotions-library``). If the library can't load (offline),
the controller falls back to a small procedural gesture keyed by the intent family.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

from ..config import Config

log = logging.getLogger("live_chat.emotions")

# Intent → candidate clips, straight from the official app's play_emotion tool.
INTENT_TO_MOVES: dict[str, tuple[str, ...]] = {
    "happy": ("laughing2", "laughing1"),
    "excited": ("dance3", "dance2"),
    "loving": ("loving1",),
    "grateful": ("grateful1",),
    "success": ("success1", "success2"),
    "thinking": ("thoughtful1", "thoughtful2"),
    "attentive": ("attentive1", "attentive2"),
    "confused": ("confused1",),
    "uncertain": ("uncertain1",),
    "sad": ("sad1", "sad2", "downcast1"),
    "downcast": ("downcast1", "sad1"),
    "lonely": ("lonely1",),
    "angry": ("rage1", "irritated2", "irritated1"),
    "irritated": ("irritated1", "irritated2", "displeased2"),
    "displeased": ("displeased1", "displeased2"),
    "disgusted": ("disgusted1",),
    "scared": ("scared1", "fear1", "anxiety1"),
    "anxious": ("anxiety1", "fear1", "scared1"),
    "surprised": ("surprised1", "surprised2", "amazed1"),
    "amazed": ("amazed1", "surprised1"),
    "calming": ("calming1",),
    "relief": ("relief1", "relief2"),
    "impatient": ("impatient2",),
    "embarrassed": ("shy1",),
    "bored": ("boredom2", "boredom1"),
    "tired": ("exhausted1", "sleep1"),
    "sleepy": ("sleep1", "exhausted1"),
    "yes": ("yes1", "understanding2"),
    "no": ("no1",),
    "no_sad": ("no_sad1",),
    "no_excited": ("no_excited1",),
    "welcoming": ("welcoming2",),
    "greeting": ("welcoming2",),
    "goodbye": ("loving1", "welcoming2"),
    "helpful": ("helpful1",),
    "dance": ("dance2", "dance3"),
}

# Fallback move names when the library can't be listed (offline): every clip the
# intents above reference. Kept as the default `available()` universe.
ALLOWED_EMOTIONS = sorted({m for moves in INTENT_TO_MOVES.values() for m in moves})

# Bilingual cue -> intent. Order matters (first match wins): specific feelings
# before generic yes/question cues, so "谢谢" beats the "吗" question fallback.
_INTENT_RULES = [
    ("grateful", ["谢谢", "感谢", "多谢", "thank", "thanks", "appreciate"]),
    ("goodbye", ["再见", "拜拜", "回头见", "晚安", "goodbye", "bye", "see you", "good night"]),
    ("greeting", ["你好", "您好", "嗨", "早上好", "晚上好", "hello", "hi ", "hey", "welcome", "good morning"]),
    ("happy", ["哈哈", "太好笑", "好笑", "有意思", "有趣", "haha", "lol", "funny", "hilarious"]),
    ("loving", ["爱你", "喜欢你", "抱抱", "love you", "adore"]),
    ("surprised", ["哇", "天哪", "真的吗", "不会吧", "居然", "wow", "really?", "no way", "amazing", "whoa"]),
    ("scared", ["好可怕", "吓", "恐怖", "scary", "terrifying", "frightening"]),
    ("sad", ["难过", "抱歉", "对不起", "遗憾", "可惜", "伤心", "sorry", "sad", "unfortunate", "afraid"]),
    ("calming", ["别担心", "放心", "没关系", "不用怕", "冷静", "don't worry", "no worries", "it's okay", "calm"]),
    ("success", ["搞定", "完成", "成功", "做到了", "太棒了", "done!", "success", "finished", "nailed it"]),
    ("confused", ["不懂", "没听清", "没听懂", "什么意思", "confused", "not sure what", "don't understand"]),
    ("thinking", ["让我想想", "我想一下", "考虑一下", "嗯……", "let me think", "thinking about", "hmm"]),
    ("tired", ["好累", "困了", "想睡", "tired", "sleepy", "exhausted"]),
    ("no", ["不行", "不是", "不可以", "别这样", "no,", "nope", "don't", "can't", "cannot"]),
    ("yes", ["是的", "对", "好的", "没错", "当然", "可以", "嗯", "yes", "sure", "of course", "okay", "ok", "right"]),
    ("excited", ["太棒", "太好了", "棒极了", "awesome", "fantastic", "incredible", "excited"]),
    ("uncertain", ["也许", "可能", "不一定", "说不好", "maybe", "perhaps", "not sure"]),
    ("attentive", ["我在听", "继续说", "然后呢", "i'm listening", "go on", "tell me more"]),
    # Questions the model asks back → a pondering tilt (the library has no dedicated
    # "curious" clip; the official app expresses curiosity with thoughtful*).
    ("thinking", ["为什么", "怎么", "什么", "吗", "?", "？", "why", "how", "what", "which", "who", "where"]),
]


def map_intent(text: str, lang: str) -> Optional[str]:
    """Pick an emotional intent from keyword cues; returns None if nothing fits."""
    if not text:
        return None
    low = text.lower()
    for intent, cues in _INTENT_RULES:
        for c in cues:
            if c in low:
                return intent
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

    def resolve(self, intent_or_clip: str) -> str:
        """Turn an intent (or a direct clip name) into a concrete clip name.

        Picks randomly among the intent's *available* clips so repeated intents vary.
        Unknown names pass through unchanged (the controller's procedural fallback
        understands intent families by base name).
        """
        if not intent_or_clip:
            return intent_or_clip
        if intent_or_clip in self._names:
            return intent_or_clip
        cands = INTENT_TO_MOVES.get(intent_or_clip, ())
        avail = [c for c in cands if c in self._names]
        if avail:
            return random.choice(avail)
        if cands:
            return random.choice(list(cands))
        return intent_or_clip

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
