"""Unit tests for environment-driven settings."""

from yrobot.config import TRAINED_SYSTEM_LINE, Settings


def test_defaults_target_official_gateway():
    s = Settings()
    assert s.url == "wss://minicpmo45.modelbest.cn/v1/realtime?mode=audio"
    assert s.chunk_ms == 500
    assert s.system_prompt.startswith(TRAINED_SYSTEM_LINE + "\n")


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("YROBOT_REALTIME_URL", "10.0.16.184:8006")
    monkeypatch.setenv("YROBOT_TLS_VERIFY", "0")
    monkeypatch.setenv("YROBOT_CHUNK_MS", "500")
    monkeypatch.setenv("YROBOT_PERSONA", "只说中文。")
    monkeypatch.setenv("YROBOT_SEND_VIDEO", "false")
    s = Settings.from_env()
    assert s.url == "wss://10.0.16.184:8006/v1/realtime?mode=audio"
    assert s.tls_verify is False
    assert s.send_video is False
    assert s.system_prompt == f"{TRAINED_SYSTEM_LINE}\n只说中文。"


def test_empty_persona_keeps_trained_line_only(monkeypatch):
    monkeypatch.setenv("YROBOT_PERSONA", "  ")
    assert Settings.from_env().system_prompt == TRAINED_SYSTEM_LINE
