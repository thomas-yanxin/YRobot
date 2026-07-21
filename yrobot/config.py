"""Small, environment-driven configuration for the CM4 client."""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
CHUNK_SAMPLES = INPUT_SAMPLE_RATE

DEFAULT_SYSTEM_PROMPT = (
    "你是 Reachy Mini，一个友好、有生命力的桌面机器人。你正在和面前的人进行实时面对面交谈，"
    "能听见声音并看见摄像头画面。请使用对方的语言，回答自然、简短、口语化，通常一到两句话。"
    "不要使用 Markdown、列表或会被语音读出的动作标签。"
)


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


def normalize_backend_url(value: str) -> str:
    """Normalize a raw llama-omni-server URL to its `/backend` WebSocket."""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("OMNI_WS_URL must be a ws:// or wss:// URL")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/backend"
    if path != "/backend":
        raise ValueError("YRobot connects directly to llama-omni-server; URL path must be /backend")
    if parsed.query or parsed.fragment:
        raise ValueError("OMNI_WS_URL must not contain a query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True, slots=True)
class Config:
    omni_url: str
    tls_verify: bool
    send_video: bool
    system_prompt: str
    reconnect_delay: float = 1.5
    session_timeout: float = 120.0
    max_message_size: int = 64 * 1024 * 1024

    @classmethod
    def load(cls) -> Config:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        prompt = os.getenv("OMNI_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT
        return cls(
            omni_url=normalize_backend_url(
                os.getenv("OMNI_WS_URL", "wss://10.0.16.187:28099/backend")
            ),
            tls_verify=_bool_env("OMNI_TLS_VERIFY", False),
            send_video=_bool_env("OMNI_SEND_VIDEO", True),
            system_prompt=prompt,
        )

    def ssl_context(self) -> ssl.SSLContext | None:
        if self.omni_url.startswith("ws://"):
            return None
        if self.tls_verify:
            return ssl.create_default_context()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
