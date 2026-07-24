from __future__ import annotations

from dataclasses import replace

import pytest

from yrobot.config import OFFICIAL_REALTIME_URL, Settings


def test_official_realtime_defaults_are_fixed_to_video_duplex() -> None:
    settings = Settings()
    settings.validate()

    assert settings.realtime_url == OFFICIAL_REALTIME_URL
    assert settings.input_sample_rate == 16_000
    assert settings.output_sample_rate == 24_000
    assert settings.input_unit_ms == 1_000
    assert settings.camera_width == 640
    assert settings.camera_fps == 1.0
    assert settings.vision_send_interval_seconds == 1.0
    assert settings.playback_preroll_ms == 0
    assert settings.barge_attack_ms == 80
    assert settings.echo_correlation == 0.72
    assert settings.doa_hz == 10.0
    assert settings.motion_hz == 50.0
    assert settings.session_seconds < 300


def test_environment_only_exposes_operational_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yrobot.config.load_dotenv", lambda: None)
    monkeypatch.setenv(
        "YROBOT_REALTIME_URL",
        "wss://brain.local:8006/v1/realtime?mode=video",
    )
    monkeypatch.setenv("YROBOT_TLS_VERIFY", "false")
    monkeypatch.setenv("YROBOT_VAD_MODE", "3")
    monkeypatch.setenv("YROBOT_BARGE_ATTACK_MS", "60")
    monkeypatch.setenv("YROBOT_VISION_SEND_INTERVAL_SECONDS", "4")
    monkeypatch.setenv("YROBOT_DOA_HZ", "15")

    settings = Settings.from_env()

    assert settings.realtime_url == ("wss://brain.local:8006/v1/realtime?mode=video")
    assert settings.tls_verify is False
    assert settings.vad_mode == 3
    assert settings.barge_attack_ms == 60
    assert settings.vision_send_interval_seconds == 4
    assert settings.doa_hz == 15


@pytest.mark.parametrize(
    "settings",
    [
        replace(Settings(), realtime_url="https://example/v1/realtime?mode=video"),
        replace(Settings(), realtime_url="wss://example/v1/realtime?mode=audio"),
        replace(Settings(), realtime_url="wss://example/v1/realtime?mode=video&x=1"),
        replace(Settings(), input_unit_ms=500),
        replace(Settings(), input_sample_rate=24_000),
        replace(Settings(), output_sample_rate=16_000),
        replace(Settings(), session_seconds=300),
        replace(Settings(), reconnect_max_seconds=float("nan")),
        replace(Settings(), vad_min_rms=float("nan")),
        replace(Settings(), vad_noise_ratio=float("nan")),
        replace(Settings(), vad_noise_ratio=1),
        replace(Settings(), camera_width=800),
        replace(Settings(), camera_fps=2),
        replace(Settings(), vision_send_interval_seconds=0.5),
        replace(Settings(), vision_send_interval_seconds=11),
        replace(Settings(), motion_hz=40),
        replace(Settings(), barge_attack_ms=90),
        replace(Settings(), echo_correlation=1),
        replace(Settings(), doa_hold_seconds=float("nan")),
    ],
)
def test_invalid_protocol_or_control_tuning_is_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError):
        settings.validate()


def test_invalid_boolean_has_actionable_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yrobot.config.load_dotenv", lambda: None)
    monkeypatch.setenv("YROBOT_TLS_VERIFY", "sometimes")

    with pytest.raises(ValueError, match="YROBOT_TLS_VERIFY"):
        Settings.from_env()
