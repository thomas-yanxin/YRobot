from reachy_mini_live_chat.llm.prompts import ALLOWED_EMOTIONS
from reachy_mini_live_chat.motion.emotions import map_intent


def test_affirmation_zh():
    assert map_intent("好的，没问题", "zh") == "yes1"


def test_affirmation_en():
    assert map_intent("Sure, that works", "en") == "yes1"


def test_negation():
    assert map_intent("不行", "zh") == "no1"
    assert map_intent("No, don't do that", "en") == "no1"


def test_thanks_before_question():
    # gratitude (earlier rule) should win over a trailing question cue
    assert map_intent("谢谢你，为什么呢？", "zh") == "grateful1"


def test_question_zh():
    assert map_intent("这是为什么", "zh") == "curious1"


def test_none_when_neutral():
    assert map_intent("我在桌子旁边坐着", "zh") is None


def test_all_mapped_emotions_are_allowed():
    for _, _cues in []:
        pass
    from reachy_mini_live_chat.motion.emotions import _INTENT_RULES

    for emo, _ in _INTENT_RULES:
        assert emo in ALLOWED_EMOTIONS
