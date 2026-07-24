"""Unit tests for environment-driven settings."""

import pytest

from yrobot.config import TRAINED_SYSTEM_LINE, Settings


def test_defaults_target_official_gateway():
    s = Settings()
    assert s.url == "wss://minicpmo45.modelbest.cn/v1/realtime?mode=audio"
    assert s.chunk_ms == 1000
    assert s.send_video is False
    assert s.realtime_mode == "audio"
    assert s.system_prompt.startswith(TRAINED_SYSTEM_LINE + "\n")


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("YROBOT_REALTIME_URL", "10.0.16.184:8006")
    monkeypatch.setenv("YROBOT_TLS_VERIFY", "0")
    monkeypatch.setenv("YROBOT_CHUNK_MS", "1000")
    monkeypatch.setenv("YROBOT_PERSONA", "只说中文。")
    monkeypatch.setenv("YROBOT_SEND_VIDEO", "false")
    monkeypatch.setenv("YROBOT_BARGE_ECHO_SIMILARITY", "0.8")
    monkeypatch.setenv("YROBOT_BARGE_UNEXPLAINED_DB", "-44")
    s = Settings.from_env()
    assert s.url == "wss://10.0.16.184:8006/v1/realtime?mode=audio"
    assert s.tls_verify is False
    assert s.send_video is False
    assert s.barge_echo_similarity == 0.8
    assert s.barge_unexplained_db == -44.0
    assert s.system_prompt == f"{TRAINED_SYSTEM_LINE}\n只说中文。"


def test_empty_persona_keeps_trained_line_only(monkeypatch):
    monkeypatch.setenv("YROBOT_PERSONA", "  ")
    assert Settings.from_env().system_prompt == TRAINED_SYSTEM_LINE


def test_send_video_selects_video_mode_for_legacy_bare_url(monkeypatch):
    monkeypatch.setenv("YROBOT_REALTIME_URL", "10.0.16.184:8006")
    monkeypatch.setenv("YROBOT_SEND_VIDEO", "true")
    s = Settings.from_env()
    assert s.realtime_mode == "video"
    assert s.session_budget_s == 280.0


def test_explicit_audio_mode_rejects_video_frames(monkeypatch):
    monkeypatch.setenv(
        "YROBOT_REALTIME_URL",
        "wss://10.0.16.184:8006/v1/realtime?mode=audio",
    )
    monkeypatch.setenv("YROBOT_SEND_VIDEO", "true")
    with pytest.raises(ValueError, match="mode=video"):
        Settings.from_env()


@pytest.mark.parametrize("chunk_ms", [20, 500, 2000])
def test_chunk_size_must_match_model_inference_unit(chunk_ms):
    with pytest.raises(ValueError, match="must be 1000"):
        Settings(chunk_ms=chunk_ms)


@pytest.mark.parametrize("similarity", [-0.1, 1.1])
def test_echo_similarity_must_be_normalized(similarity):
    with pytest.raises(ValueError, match="between 0 and 1"):
        Settings(barge_echo_similarity=similarity)


@pytest.mark.parametrize("unexplained_db", [-121.0, 0.1])
def test_unexplained_energy_threshold_must_be_decibels(unexplained_db):
    with pytest.raises(ValueError, match="between -120 and 0"):
        Settings(barge_unexplained_db=unexplained_db)
