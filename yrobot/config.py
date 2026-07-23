"""Environment-driven configuration for the YRobot realtime client."""

from __future__ import annotations

import math
import os
import ssl
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
AUDIO_UNIT_SAMPLES = INPUT_SAMPLE_RATE

DEFAULT_REALTIME_URL = "wss://10.0.16.184:8006/v1/realtime?mode=video"
# The immutable comni deployment puts a natural-language system prompt into
# the reference-audio suffix and breaks the duplex template. Empty preserves
# the server's valid built-in prompt.
DEFAULT_SYSTEM_PROMPT = ""


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value!r}")
    return parsed


def _int_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value!r}")
    return parsed


def normalize_realtime_url(value: str) -> str:
    """Normalize a host or HTTP/WebSocket URL to the video Realtime Gateway."""

    raw = value.strip()
    if not raw:
        raise ValueError("YROBOT_REALTIME_URL must not be empty")

    # The official Gateway is HTTPS/WSS by default. urlsplit treats
    # ``host:port`` as a custom scheme, so bare endpoints need an explicit
    # secure transport before parsing. Plain deployments remain available by
    # spelling http:// or ws:// explicitly.
    if "://" not in raw:
        raw = f"wss://{raw.lstrip('/')}"

    parsed = urlsplit(raw)
    scheme_map = {
        "http": "ws",
        "https": "wss",
        "ws": "ws",
        "wss": "wss",
    }
    scheme = scheme_map.get(parsed.scheme.lower())
    if scheme is None or not parsed.netloc or parsed.hostname is None:
        raise ValueError("YROBOT_REALTIME_URL must be a host or an http(s)/ws(s) URL")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("YROBOT_REALTIME_URL contains an invalid port") from exc

    path = parsed.path.rstrip("/")
    if path == "/backend" or path.endswith("/backend"):
        raise ValueError(
            "The legacy /backend endpoint is not supported; use /v1/realtime?mode=video"
        )
    if path not in {"", "/v1/realtime"}:
        raise ValueError("YROBOT_REALTIME_URL path must be /v1/realtime")
    if parsed.fragment:
        raise ValueError("YROBOT_REALTIME_URL must not contain a fragment")

    query = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key != "mode" for key, _ in query):
        raise ValueError("YROBOT_REALTIME_URL only supports the mode query parameter")

    return urlunsplit(
        (
            scheme,
            parsed.netloc,
            "/v1/realtime",
            urlencode({"mode": "video"}),
            "",
        )
    )


@dataclass(frozen=True, slots=True)
class Config:
    realtime_url: str
    tls_verify: bool
    send_video: bool
    system_prompt: str
    enable_tts: bool = True
    length_penalty: float = 1.1
    force_listen_count: int = 1
    reconnect_initial_delay: float = 0.5
    reconnect_max_delay: float = 8.0
    reconnect_reset_after: float = 30.0
    handshake_timeout: float = 120.0
    close_ack_timeout: float = 5.0
    session_rollover: float = 285.0
    max_message_size: int = 16 * 1024 * 1024

    @classmethod
    def load(cls) -> Config:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        if os.getenv("YROBOT_SYSTEM_PROMPT", "").strip():
            raise ValueError(
                "YROBOT_SYSTEM_PROMPT is unsupported by the fixed comni deployment; leave it empty"
            )
        reconnect_initial = _float_env(
            "YROBOT_RECONNECT_INITIAL",
            0.5,
            minimum=0.05,
            maximum=60.0,
        )
        reconnect_max = _float_env(
            "YROBOT_RECONNECT_MAX",
            8.0,
            minimum=0.05,
            maximum=120.0,
        )
        if reconnect_max < reconnect_initial:
            raise ValueError(
                "YROBOT_RECONNECT_MAX must be greater than or equal to YROBOT_RECONNECT_INITIAL"
            )

        return cls(
            realtime_url=normalize_realtime_url(
                os.getenv("YROBOT_REALTIME_URL", DEFAULT_REALTIME_URL)
            ),
            # The configured on-premise Gateway currently uses the self-signed
            # certificate generated by MiniCPM-o-Demo. Reissue a trusted
            # certificate with a matching IP/DNS SAN, then set this to 1.
            tls_verify=_bool_env("YROBOT_TLS_VERIFY", False),
            send_video=_bool_env("YROBOT_SEND_VIDEO", True),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            enable_tts=_bool_env("YROBOT_ENABLE_TTS", True),
            length_penalty=_float_env(
                "YROBOT_LENGTH_PENALTY",
                1.1,
                minimum=0.1,
                maximum=5.0,
            ),
            force_listen_count=_int_env(
                "YROBOT_FORCE_LISTEN_COUNT",
                1,
                minimum=0,
                maximum=10,
            ),
            reconnect_initial_delay=reconnect_initial,
            reconnect_max_delay=reconnect_max,
            session_rollover=_float_env(
                "YROBOT_SESSION_ROLLOVER",
                285.0,
                minimum=30.0,
                maximum=290.0,
            ),
        )

    def ssl_context(self) -> ssl.SSLContext | None:
        if self.realtime_url.startswith("ws://"):
            return None
        if self.tls_verify:
            return ssl.create_default_context()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
