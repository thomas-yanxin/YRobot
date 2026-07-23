"""Protocol encoding and session-rotation policy."""

import base64
import json

import numpy as np

from yrobot.config import Config
from yrobot.omni import ThinkFilter, encode_append, should_rotate


def test_encode_append_audio_only():
    pcm = np.array([0.0, 0.5, -0.5], np.float32)
    msg = json.loads(encode_append(pcm, None, False))
    assert msg["type"] == "input.append"
    decoded = np.frombuffer(base64.b64decode(msg["input"]["audio"]), np.float32)
    np.testing.assert_array_equal(decoded, pcm)
    assert msg["input"]["force_listen"] is False
    assert "video_frames" not in msg["input"]


def test_encode_append_with_frame_and_force():
    msg = json.loads(encode_append(np.zeros(4, np.float32), b"\xff\xd8jpeg", True))
    assert msg["input"]["force_listen"] is True
    assert base64.b64decode(msg["input"]["video_frames"][0]) == b"\xff\xd8jpeg"
    assert msg["input"]["max_slice_nums"] == 1


def test_rotation_policy():
    cfg = Config()  # kv_soft=6500 kv_hard=7800, audio budget 570 s
    assert should_rotate(cfg, kv=100, age_s=10, quiet=True) is None
    assert should_rotate(cfg, kv=100, age_s=10, quiet=False) is None
    assert should_rotate(cfg, kv=7900, age_s=10, quiet=False)  # hard kv
    assert should_rotate(cfg, kv=6600, age_s=10, quiet=False) is None  # soft needs quiet
    assert should_rotate(cfg, kv=6600, age_s=10, quiet=True)
    assert should_rotate(cfg, kv=100, age_s=571, quiet=False)  # hard age
    assert should_rotate(cfg, kv=100, age_s=545, quiet=True)  # near-cap + quiet


def test_url_and_budget():
    cfg = Config(url="wss://h:8006/v1/realtime", mode="audio")
    assert cfg.full_url.endswith("?mode=audio")
    assert cfg.session_budget_s == 570.0
    # a stale ?mode=video in the URL is overridden by cfg.mode
    stale = Config(url="wss://h:8006/v1/realtime?mode=video")
    assert stale.full_url == "wss://h:8006/v1/realtime?mode=audio"
    assert stale.session_budget_s == 570.0
    video = Config(url="wss://h:8006/v1/realtime", mode="video")
    assert video.full_url.endswith("?mode=video")
    assert video.session_budget_s == 280.0


def test_system_prompt_composition():
    cfg = Config()
    assert cfg.system_prompt == "You are a helpful assistant."
    lines = cfg.full_system_prompt.split("\n")
    assert lines[0] == "You are a helpful assistant."
    assert len(lines) == 2 and lines[1] and lines[1] != lines[0]
    bare = Config(instruction="")
    assert bare.full_system_prompt == "You are a helpful assistant."


def test_think_filter_strips_blocks_and_split_tags():
    f = ThinkFilter()
    assert f.feed("你好<think>秘密推理</think>世界") == "你好世界"
    f = ThinkFilter()
    assert f.feed("我在学<thi") == "我在学"
    assert f.feed("nk>abc</think>好的") == "好的"
    f = ThinkFilter()  # stray closing tag (opening lost in an earlier session)
    assert f.feed("有个人</think>坐着看手机") == "有个人坐着看手机"
    f = ThinkFilter()
    assert f.feed("平常文字") == "平常文字"
    assert f.feed("不受影响") == "不受影响"
