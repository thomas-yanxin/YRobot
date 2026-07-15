"""Config URL resolution: gateway vs raw backend vs verbatim path."""
from reachy_mini_live_chat.config import Config


def _cfg(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_gateway_url_default():
    c = _cfg(omni_ws_url="wss://host:8006", omni_endpoint="gateway", omni_gateway_mode="video")
    assert c.omni_backend_url == "wss://host:8006/v1/realtime?mode=video"


def test_gateway_url_audio_mode():
    c = _cfg(omni_ws_url="wss://host:8006", omni_endpoint="gateway", omni_gateway_mode="audio")
    assert c.omni_backend_url == "wss://host:8006/v1/realtime?mode=audio"


def test_raw_backend_url():
    c = _cfg(omni_ws_url="wss://host:28099", omni_endpoint="backend")
    assert c.omni_backend_url == "wss://host:28099/backend"


def test_explicit_path_used_verbatim():
    c = _cfg(omni_ws_url="wss://host:8006/v1/realtime?mode=video", omni_endpoint="backend")
    # a path is present → endpoint is ignored, URL used as-is
    assert c.omni_backend_url == "wss://host:8006/v1/realtime?mode=video"


def test_trailing_slash_base():
    c = _cfg(omni_ws_url="wss://host:8006/", omni_endpoint="gateway", omni_gateway_mode="video")
    assert c.omni_backend_url == "wss://host:8006/v1/realtime?mode=video"
