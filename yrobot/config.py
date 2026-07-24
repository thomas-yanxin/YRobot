"""Runtime configuration.

All tunables live in one frozen dataclass built from ``YROBOT_*`` environment
variables (a ``.env`` file is honoured), so no other module reads the
environment. Defaults encode gateway behaviour verified against the official
MiniCPM-o 4.5 realtime API (https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit, urlunsplit

# The duplex template was trained with this exact first line; a free-form
# persona in its place drifts the model out of its full-duplex distribution
# and leaks <think> blocks into speech. Keep the persona to one short line.
TRAINED_SYSTEM_LINE = "You are a helpful assistant."
DEFAULT_PERSONA = "你是 Reachy，一个友好的桌面机器人，用对方的语言简短口语化地回复。"


def normalize_url(raw: str, mode: str | None = None) -> str:
    """Normalize a bare host, host:port or ws(s) URL to the realtime endpoint.

    An explicit mode in ``raw`` is preserved unless ``mode`` is supplied.
    Audio is the latency-first default; camera frames are valid only in video
    mode according to the public gateway contract.
    """
    if "://" not in raw:
        raw = f"wss://{raw}"
    parts = urlsplit(raw)
    path = parts.path if parts.path not in ("", "/") else "/v1/realtime"
    query = {k: v[0] for k, v in parse_qs(parts.query).items()}
    selected_mode = mode or query.get("mode", "audio")
    if selected_mode not in {"audio", "video"}:
        raise ValueError("YRobot realtime mode must be 'audio' or 'video'")
    query["mode"] = selected_mode
    qs = "&".join(f"{k}={v}" for k, v in sorted(query.items()))
    return urlunsplit((parts.scheme, parts.netloc, path, qs, ""))


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _num(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    url: str = normalize_url("minicpmo45.modelbest.cn")
    tls_verify: bool = True
    system_prompt: str = f"{TRAINED_SYSTEM_LINE}\n{DEFAULT_PERSONA}"
    length_penalty: float = 1.1

    # MiniCPM-o 4.5 advances its duplex timeline in fixed one-second audio
    # units. Sub-second input.append calls can return a synthetic listen
    # without applying force_listen because no inference logits were produced.
    chunk_ms: int = 1000

    # Audio mode is the latency-first default. Enabling video switches a bare
    # URL to mode=video; an explicitly contradictory URL is rejected.
    send_video: bool = False
    frame_period_active_s: float = 1.0
    frame_period_idle_s: float = 5.0

    # Session rotation: rotate before the 600 s server cap or before the
    # ~8192-token kv budget degrades replies, whichever comes first — and
    # only at a quiet listen boundary.
    session_budget_s: float = 550.0
    kv_budget_tokens: float = 7200.0
    reconnect_delay_s: float = 2.5

    vad_aggressiveness: int = 2
    head_tracking_weight: float = 0.4

    def __post_init__(self) -> None:
        if self.chunk_ms != 1000:
            raise ValueError("YROBOT_CHUNK_MS must be 1000 for MiniCPM-o 4.5 duplex")
        if self.send_video and self.realtime_mode != "video":
            raise ValueError("YROBOT_SEND_VIDEO requires realtime mode=video")

    @property
    def realtime_mode(self) -> str:
        """Public gateway mode selected in ``url``."""
        return parse_qs(urlsplit(self.url).query).get("mode", ["audio"])[0]

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``YROBOT_*`` environment variables."""
        persona = os.environ.get("YROBOT_PERSONA", DEFAULT_PERSONA).strip()
        raw_url = os.environ.get("YROBOT_REALTIME_URL", "minicpmo45.modelbest.cn")
        send_video = _flag("YROBOT_SEND_VIDEO", False)
        requested_mode = os.environ.get("YROBOT_REALTIME_MODE")
        if requested_mode is not None:
            requested_mode = requested_mode.strip().lower()
        else:
            raw_query = parse_qs(urlsplit(raw_url if "://" in raw_url else f"wss://{raw_url}").query)
            # Keep an explicit URL mode authoritative. For legacy deployments
            # that only set SEND_VIDEO, select the protocol mode that can
            # actually carry frames.
            requested_mode = None if "mode" in raw_query else ("video" if send_video else "audio")
        url = normalize_url(raw_url, requested_mode)
        actual_mode = parse_qs(urlsplit(url).query).get("mode", ["audio"])[0]
        if send_video and actual_mode != "video":
            raise ValueError("YROBOT_SEND_VIDEO requires realtime mode=video")
        default_session_budget = 280.0 if actual_mode == "video" else 550.0
        return cls(
            url=url,
            tls_verify=_flag("YROBOT_TLS_VERIFY", True),
            system_prompt=f"{TRAINED_SYSTEM_LINE}\n{persona}" if persona else TRAINED_SYSTEM_LINE,
            length_penalty=_num("YROBOT_LENGTH_PENALTY", 1.1),
            chunk_ms=int(_num("YROBOT_CHUNK_MS", 1000)),
            send_video=send_video,
            session_budget_s=_num("YROBOT_SESSION_BUDGET_S", default_session_budget),
            kv_budget_tokens=_num("YROBOT_KV_BUDGET", 7200.0),
            reconnect_delay_s=_num("YROBOT_RECONNECT_DELAY_S", 2.5),
            vad_aggressiveness=int(_num("YROBOT_VAD_AGGRESSIVENESS", 2)),
            head_tracking_weight=_num("YROBOT_HEAD_TRACKING_WEIGHT", 0.4),
        )
