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


def normalize_url(raw: str) -> str:
    """Normalize a bare host, host:port or ws(s) URL to the realtime endpoint.

    ``mode=audio`` is forced: it is the only full-duplex mode with a 600 s
    session cap (video allows 300 s) and the gateway still accepts
    ``video_frames`` in audio mode.
    """
    if "://" not in raw:
        raw = f"wss://{raw}"
    parts = urlsplit(raw)
    path = parts.path if parts.path not in ("", "/") else "/v1/realtime"
    query = {k: v[0] for k, v in parse_qs(parts.query).items()}
    query["mode"] = "audio"
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

    # Uplink cadence. 500 ms halves perceived reply latency versus the 1 s
    # browser-demo cadence; 250 ms makes the model answer mid-utterance.
    chunk_ms: int = 500

    # Camera frames ride along in input.append. Vision costs ~64 kv tokens
    # per frame, so cadence adapts to conversation activity and frames are
    # never sent while only the robot is speaking.
    send_video: bool = True
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

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``YROBOT_*`` environment variables."""
        persona = os.environ.get("YROBOT_PERSONA", DEFAULT_PERSONA).strip()
        return cls(
            url=normalize_url(os.environ.get("YROBOT_REALTIME_URL", "minicpmo45.modelbest.cn")),
            tls_verify=_flag("YROBOT_TLS_VERIFY", True),
            system_prompt=f"{TRAINED_SYSTEM_LINE}\n{persona}" if persona else TRAINED_SYSTEM_LINE,
            length_penalty=_num("YROBOT_LENGTH_PENALTY", 1.1),
            chunk_ms=int(_num("YROBOT_CHUNK_MS", 500)),
            send_video=_flag("YROBOT_SEND_VIDEO", True),
            session_budget_s=_num("YROBOT_SESSION_BUDGET_S", 550.0),
            kv_budget_tokens=_num("YROBOT_KV_BUDGET", 7200.0),
            reconnect_delay_s=_num("YROBOT_RECONNECT_DELAY_S", 2.5),
            vad_aggressiveness=int(_num("YROBOT_VAD_AGGRESSIVENESS", 2)),
            head_tracking_weight=_num("YROBOT_HEAD_TRACKING_WEIGHT", 0.4),
        )
