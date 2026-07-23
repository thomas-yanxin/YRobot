import ssl

import pytest

from yrobot.config import DEFAULT_SYSTEM_PROMPT, Config, normalize_realtime_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "10.0.16.184:8006",
            "wss://10.0.16.184:8006/v1/realtime?mode=video",
        ),
        (
            "10.0.16.184:8006/v1/realtime?mode=video",
            "wss://10.0.16.184:8006/v1/realtime?mode=video",
        ),
        (
            "http://brain.local:8006",
            "ws://brain.local:8006/v1/realtime?mode=video",
        ),
        (
            "https://brain.local/v1/realtime?mode=audio",
            "wss://brain.local/v1/realtime?mode=video",
        ),
        (
            "ws://brain.local:8006/v1/realtime/",
            "ws://brain.local:8006/v1/realtime?mode=video",
        ),
        (
            "wss://brain.local",
            "wss://brain.local/v1/realtime?mode=video",
        ),
    ],
)
def test_normalize_realtime_url(value: str, expected: str) -> None:
    assert normalize_realtime_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "ws://brain.local:28099/backend",
        "https://brain.local/some/backend",
        "ftp://brain.local",
        "ws://brain.local/not-realtime",
        "ws://brain.local/v1/realtime?token=secret",
        "ws://brain.local/v1/realtime#fragment",
        "ws://brain.local:not-a-port",
        "",
    ],
)
def test_normalize_realtime_url_rejects_legacy_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_realtime_url(value)


def test_config_loads_yrobot_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YROBOT_REALTIME_URL", "https://brain.local:8006")
    monkeypatch.setenv("YROBOT_TLS_VERIFY", "0")
    monkeypatch.setenv("YROBOT_SEND_VIDEO", "false")
    monkeypatch.setenv("YROBOT_ENABLE_TTS", "false")
    monkeypatch.setenv("YROBOT_LENGTH_PENALTY", "1.25")
    monkeypatch.setenv("YROBOT_FORCE_LISTEN_COUNT", "1")
    monkeypatch.setenv("YROBOT_RECONNECT_INITIAL", "0.25")
    monkeypatch.setenv("YROBOT_RECONNECT_MAX", "4")
    monkeypatch.setenv("YROBOT_SESSION_ROLLOVER", "280")

    config = Config.load()

    assert config.realtime_url == "wss://brain.local:8006/v1/realtime?mode=video"
    assert config.tls_verify is False
    assert config.send_video is False
    assert config.enable_tts is False
    assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert config.length_penalty == 1.25
    assert config.force_listen_count == 1
    assert config.reconnect_initial_delay == 0.25
    assert config.reconnect_max_delay == 4.0
    assert config.session_rollover == 280.0


def test_config_defaults_use_minicpm_realtime_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        "YROBOT_REALTIME_URL",
        "YROBOT_TLS_VERIFY",
        "YROBOT_SEND_VIDEO",
        "YROBOT_ENABLE_TTS",
        "YROBOT_SYSTEM_PROMPT",
        "YROBOT_LENGTH_PENALTY",
        "YROBOT_FORCE_LISTEN_COUNT",
        "YROBOT_RECONNECT_INITIAL",
        "YROBOT_RECONNECT_MAX",
        "YROBOT_SESSION_ROLLOVER",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)

    # Legacy variables are deliberately ignored by the clean 2.0 boundary.
    monkeypatch.setenv("OMNI_WS_URL", "ws://legacy.invalid/backend")
    monkeypatch.setenv("OMNI_FORCE_LISTEN_COUNT", "9")

    config = Config.load()

    assert config.realtime_url == ("wss://10.0.16.184:8006/v1/realtime?mode=video")
    assert config.tls_verify is False
    assert config.send_video is True
    assert config.enable_tts is True
    assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert config.length_penalty == 1.1
    assert config.force_listen_count == 1
    assert config.session_rollover == 285.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("YROBOT_TLS_VERIFY", "maybe"),
        ("YROBOT_LENGTH_PENALTY", "nan"),
        ("YROBOT_LENGTH_PENALTY", "5.1"),
        ("YROBOT_FORCE_LISTEN_COUNT", "-1"),
        ("YROBOT_FORCE_LISTEN_COUNT", "not-an-integer"),
        ("YROBOT_SESSION_ROLLOVER", "299"),
        ("YROBOT_SYSTEM_PROMPT", "unsupported persona"),
    ],
)
def test_config_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Config.load()


def test_config_rejects_inverted_reconnect_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YROBOT_RECONNECT_INITIAL", "4")
    monkeypatch.setenv("YROBOT_RECONNECT_MAX", "1")
    with pytest.raises(ValueError, match="YROBOT_RECONNECT_MAX"):
        Config.load()


def test_ssl_context_follows_normalized_transport() -> None:
    plain = Config(
        realtime_url="ws://brain.local/v1/realtime?mode=video",
        tls_verify=True,
        send_video=True,
        system_prompt="test",
    )
    assert plain.ssl_context() is None

    secure = Config(
        realtime_url="wss://brain.local/v1/realtime?mode=video",
        tls_verify=False,
        send_video=True,
        system_prompt="test",
    )
    context = secure.ssl_context()
    assert context is not None
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False
