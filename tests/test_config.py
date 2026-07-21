import ssl

import pytest

from yrobot.config import Config, normalize_backend_url


def test_backend_url_adds_path() -> None:
    assert normalize_backend_url("wss://robot-brain:28099") == ("wss://robot-brain:28099/backend")


@pytest.mark.parametrize(
    "url",
    [
        "https://robot-brain:28099/backend",
        "wss://robot-brain:8006/v1/realtime?mode=video",
        "not-a-url",
    ],
)
def test_backend_url_rejects_non_backend_urls(url: str) -> None:
    with pytest.raises(ValueError):
        normalize_backend_url(url)


def test_config_loads_small_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNI_WS_URL", "ws://127.0.0.1:28099")
    monkeypatch.setenv("OMNI_TLS_VERIFY", "1")
    monkeypatch.setenv("OMNI_SEND_VIDEO", "0")
    monkeypatch.setenv("OMNI_SYSTEM_PROMPT", "short prompt")

    config = Config.load()

    assert config.omni_url == "ws://127.0.0.1:28099/backend"
    assert config.tls_verify is True
    assert config.send_video is False
    assert config.system_prompt == "short prompt"
    assert config.ssl_context() is None


def test_unverified_tls_context() -> None:
    config = Config(
        omni_url="wss://127.0.0.1:28099/backend",
        tls_verify=False,
        send_video=True,
        system_prompt="test",
    )
    context = config.ssl_context()
    assert context is not None
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False
