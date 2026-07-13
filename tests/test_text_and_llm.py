from reachy_mini_live_chat.text_utils import (
    ClauseAccumulator,
    detect_lang,
    strip_control_tags,
)
from reachy_mini_live_chat.llm.router import _scan_emotion
from reachy_mini_live_chat.llm.client import strip_think


def _joined(chunks):
    return "".join(strip_think(iter(chunks)))


def test_strip_think_passthrough():
    assert _joined(["hello ", "world"]) == "hello world"


def test_strip_think_single_chunk():
    assert _joined(["<think>secret</think>好的"]) == "好的"


def test_strip_think_span_across_chunks():
    assert _joined(["ans1 <thi", "nk>reason", "ing</thi", "nk> ans2"]) == "ans1  ans2"


def test_strip_think_keeps_surrounding_text():
    assert _joined(["A<think>x</think>", "B"]) == "AB"


def test_strip_think_no_close_drops_tail():
    # unterminated think -> everything after the open tag is dropped
    assert _joined(["keep<think>never ends"]) == "keep"


def test_detect_lang():
    assert detect_lang("你好世界") == "zh"
    assert detect_lang("hello world") == "en"
    assert detect_lang("我用 Python 写代码") == "zh"  # code-switch -> zh


def test_strip_control_tags():
    assert strip_control_tags("<emo>yes1</emo>好的") == "好的"


def test_clause_accumulator_flushes_on_punctuation():
    acc = ClauseAccumulator(min_chars=2)
    out = acc.push("你好，")
    assert out == ["你好，"]
    out2 = acc.push("今天天气不错。")
    assert out2 == ["今天天气不错。"]


def test_clause_accumulator_soft_cap():
    acc = ClauseAccumulator(min_chars=2, soft_cap=10)
    # long, comma-free -> should still emit at the soft cap
    out = acc.push("abcdefghijklmnop")
    assert out and len("".join(out)) >= 10


def test_clause_flush_tail():
    acc = ClauseAccumulator()
    acc.push("residual text")
    assert acc.flush() == "residual text"


def test_scan_emotion_valid():
    name, remainder, decided = _scan_emotion("<emo>yes1</emo>好的")
    assert decided and name == "yes1" and remainder == "好的"


def test_scan_emotion_no_tag():
    name, remainder, decided = _scan_emotion("好的没问题")
    assert decided and name is None and remainder == "好的没问题"


def test_scan_emotion_invalid_name_ignored():
    name, remainder, decided = _scan_emotion("<emo>bogus</emo>hi")
    assert decided and name is None and remainder == "hi"


def test_scan_emotion_incomplete_waits():
    name, remainder, decided = _scan_emotion("<emo>ye")
    assert decided is False


def test_scan_emotion_unterminated_tag():
    # model dropped </emo> and bolted on markdown — accept the name, keep the rest as text
    name, remainder, decided = _scan_emotion("<emo>yes1**Reachy")
    assert decided and name == "yes1" and remainder == "**Reachy"


def test_scan_emotion_name_still_growing_waits():
    # ends right at the name -> could still grow (welcoming1), so wait
    _, _, decided = _scan_emotion("<emo>welcoming1")
    assert decided is False


def test_clean_spoken_strips_markdown_and_tags():
    from reachy_mini_live_chat.text_utils import clean_spoken

    assert clean_spoken("**你好** `code`") == "你好 code"
    assert clean_spoken("<emo>yes1 好的") == "好的"
    assert clean_spoken("答案</emo> 是") == "答案 是"
