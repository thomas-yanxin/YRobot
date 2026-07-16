from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.motion.emotions import (
    ALLOWED_EMOTIONS,
    INTENT_TO_MOVES,
    EmotionLibrary,
    map_intent,
)


def test_affirmation_zh():
    assert map_intent("好的，没问题", "zh") == "yes"


def test_affirmation_en():
    assert map_intent("Sure, that works", "en") == "yes"


def test_negation():
    assert map_intent("不行", "zh") == "no"
    assert map_intent("No, don't do that", "en") == "no"


def test_thanks_before_question():
    # gratitude (earlier rule) should win over a trailing question cue
    assert map_intent("谢谢你，为什么呢？", "zh") == "grateful"


def test_question_maps_to_thinking():
    assert map_intent("这是为什么", "zh") == "thinking"


def test_none_when_neutral():
    assert map_intent("我在桌子旁边坐着", "zh") is None


def test_all_rule_intents_have_moves():
    from reachy_mini_live_chat.motion.emotions import _INTENT_RULES

    for intent, _ in _INTENT_RULES:
        assert intent in INTENT_TO_MOVES, intent


def test_allowed_covers_every_mapped_clip():
    for clips in INTENT_TO_MOVES.values():
        for clip in clips:
            assert clip in ALLOWED_EMOTIONS


def test_resolve_intent_picks_a_mapped_clip():
    lib = EmotionLibrary(Config())
    for _ in range(10):
        clip = lib.resolve("sad")
        assert clip in INTENT_TO_MOVES["sad"]


def test_resolve_varies_between_calls():
    lib = EmotionLibrary(Config())
    picks = {lib.resolve("surprised") for _ in range(50)}
    assert len(picks) > 1  # random among surprised1/surprised2/amazed1


def test_resolve_passes_through_direct_clip_and_unknown():
    lib = EmotionLibrary(Config())
    assert lib.resolve("laughing2") == "laughing2"
    assert lib.resolve("not_a_thing") == "not_a_thing"
