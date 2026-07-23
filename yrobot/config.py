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
DEFAULT_AUDIO_CHUNK_MS = 500

DEFAULT_REALTIME_URL = "wss://10.0.16.184:8006/v1/realtime?mode=video"
# MiniCPM-o's duplex examples and the deployed model use this exact prompt.
# Keeping the model on its trained template is materially more reliable than
# injecting a long persona or instruction block into session.init.
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


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
    audio_chunk_ms: int = DEFAULT_AUDIO_CHUNK_MS
    frame_active_interval: float = 1.0
    frame_idle_interval: float = 5.0
    playback_lead_seconds: float = 0.120
    kv_soft_limit: int = 6_500
    kv_hard_limit: int = 7_800
    reconnect_initial_delay: float = 1.0
    reconnect_max_delay: float = 8.0
    reconnect_reset_after: float = 30.0
    handshake_timeout: float = 120.0
    close_ack_timeout: float = 5.0
    session_rollover: float = 280.0
    max_message_size: int = 16 * 1024 * 1024

    @property
    def audio_unit_samples(self) -> int:
        """Configured 16 kHz uplink unit size."""

        return INPUT_SAMPLE_RATE * self.audio_chunk_ms // 1_000

    @classmethod
    def load(cls) -> Config:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        system_prompt = os.getenv("YROBOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
        if system_prompt != DEFAULT_SYSTEM_PROMPT:
            raise ValueError(
                "YROBOT_SYSTEM_PROMPT must remain exactly "
                f"{DEFAULT_SYSTEM_PROMPT!r} for the MiniCPM-o duplex template"
            )
        audio_chunk_ms = _int_env(
            "YROBOT_CHUNK_MS",
            DEFAULT_AUDIO_CHUNK_MS,
            minimum=500,
            maximum=1_000,
        )
        if audio_chunk_ms % 20:
            raise ValueError("YROBOT_CHUNK_MS must be a multiple of 20")
        frame_active_interval = _float_env(
            "YROBOT_FRAME_ACTIVE_S",
            1.0,
            minimum=0.5,
            maximum=10.0,
        )
        frame_idle_interval = _float_env(
            "YROBOT_FRAME_IDLE_S",
            5.0,
            minimum=1.0,
            maximum=30.0,
        )
        if frame_idle_interval < frame_active_interval:
            raise ValueError(
                "YROBOT_FRAME_IDLE_S must be greater than or equal to "
                "YROBOT_FRAME_ACTIVE_S"
            )
        kv_soft_limit = _int_env(
            "YROBOT_KV_SOFT",
            6_500,
            minimum=1_000,
            maximum=8_191,
        )
        kv_hard_limit = _int_env(
            "YROBOT_KV_HARD",
            7_800,
            minimum=1_001,
            maximum=8_192,
        )
        if kv_hard_limit <= kv_soft_limit:
            raise ValueError("YROBOT_KV_HARD must be greater than YROBOT_KV_SOFT")
        reconnect_initial = _float_env(
            "YROBOT_RECONNECT_INITIAL",
            1.0,
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
            system_prompt=system_prompt,
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
            audio_chunk_ms=audio_chunk_ms,
            frame_active_interval=frame_active_interval,
            frame_idle_interval=frame_idle_interval,
            playback_lead_seconds=_float_env(
                "YROBOT_PLAYBACK_LEAD_S",
                0.120,
                minimum=0.020,
                maximum=0.150,
            ),
            kv_soft_limit=kv_soft_limit,
            kv_hard_limit=kv_hard_limit,
            reconnect_initial_delay=reconnect_initial,
            reconnect_max_delay=reconnect_max,
            session_rollover=_float_env(
                "YROBOT_SESSION_ROLLOVER",
                280.0,
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
