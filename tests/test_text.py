from reachy_mini_live_chat.text_utils import (
    clean_spoken,
    detect_lang,
    strip_control_tags,
)


def test_detect_lang():
    assert detect_lang("你好世界") == "zh"
    assert detect_lang("hello world") == "en"
    assert detect_lang("我用 Python 写代码") == "zh"  # code-switch -> zh


def test_strip_control_tags():
    assert strip_control_tags("<emo>yes1</emo>好的") == "好的"


def test_clean_spoken_strips_markdown_and_tags():
    assert clean_spoken("**你好** `code`") == "你好 code"
    assert clean_spoken("<emo>yes1 好的") == "好的"
    assert clean_spoken("答案</emo> 是") == "答案 是"
