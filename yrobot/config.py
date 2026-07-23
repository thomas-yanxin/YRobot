"""Environment-driven configuration (YROBOT_* variables, see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing vars."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.split("#", 1)[0].strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _s(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _f(name: str, default: float) -> float:
    return float(_s(name, str(default)))


def _i(name: str, default: int) -> int:
    return int(_s(name, str(default)))


def _b(name: str, default: bool) -> bool:
    return _s(name, str(int(default))).lower() in ("1", "true", "yes", "on")


DEFAULT_SYSTEM_PROMPT = (
    "你是 Reachy Mini，一个放在桌上的可爱小机器人，通过摄像头看着眼前的世界。"
    "像朋友一样口语化聊天：回答简短自然，通常一两句话，除非用户明确要求展开。"
    "用户说中文就用中文回答，说英文就用英文回答。"
)


@dataclass(frozen=True)
class Config:
    """All runtime knobs. Defaults reflect measurements against the live gateway."""

    # --- gateway ---
    url: str = "wss://10.0.16.184:8006/v1/realtime"
    mode: str = "audio"  # "audio" = 600 s sessions; still accepts video frames
    tls_verify: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    length_penalty: float = 1.1
    force_listen_count: int = 1  # listen slices forced at session start
    temperature: float = 0.0  # 0 = server default
    top_p: float = 0.0  # 0 = server default

    # --- uplink ---
    chunk_ms: int = 500  # < 500 breaks server turn-taking; 1000 adds latency
    send_video: bool = True
    frame_active_s: float = 1.0  # frame cadence while conversation is active
    frame_idle_s: float = 5.0  # sparse frames while idle (vision ~64 kv-tok each)

    # --- downlink ---
    model_out_sr: int = 24000

    # --- session budget (kv cache overflows at 8192; vision burns ~64 tok/frame) ---
    kv_soft: int = 6500  # rotate at next quiet moment
    kv_hard: int = 7800  # rotate immediately
    session_max_s: float = 0.0  # 0 = auto from mode (audio 570 / video 280)
    reconnect_initial_s: float = 2.0  # server rejects immediate reconnects
    reconnect_max_s: float = 8.0

    # --- voice gate (on the AEC'd mic stream) ---
    gate_ratio: float = 4.0  # speech threshold = noise floor × ratio
    gate_min_rms: float = 0.010
    gate_barge_mult: float = 1.5  # stricter while the robot is speaking
    onset_ms: float = 100.0
    barge_onset_ms: float = 120.0
    release_ms: float = 400.0
    floor_tau_s: float = 3.0

    # --- uplink AGC (quiet speakers are otherwise ignored by the model) ---
    agc_target_rms: float = 0.12
    agc_max_gain: float = 6.0

    # --- turn gate / barge-in ---
    quiet_s: float = 0.7  # user silence required before a listen unlatches
    hold_max_s: float = 12.0  # discard-latch safety cap
    reforce_s: float = 1.0  # min interval between re-forced listens

    # --- motion ---
    motion: bool = True
    motion_hz: float = 20.0
    track_weight_idle: float = 0.4
    track_weight_listen: float = 0.6
    track_weight_speak: float = 0.25
    doa: bool = True
    doa_min_turn_rad: float = 0.26  # ignore DOA errors below ~15°

    log_level: str = "INFO"

    @property
    def full_url(self) -> str:
        if "mode=" in self.url:
            return self.url
        sep = "&" if "?" in self.url else "?"
        return f"{self.url}{sep}mode={self.mode}"

    @property
    def effective_mode(self) -> str:
        if "mode=" in self.url:
            return self.url.rsplit("mode=", 1)[1].split("&", 1)[0]
        return self.mode

    @property
    def session_budget_s(self) -> float:
        """Stay under the gateway's hard cap (video 300 s / audio 600 s)."""
        if self.session_max_s > 0:
            return self.session_max_s
        return 280.0 if self.effective_mode == "video" else 570.0

    def session_config(self) -> dict:
        cfg: dict = {
            "length_penalty": self.length_penalty,
            "force_listen_count": self.force_listen_count,
        }
        if self.temperature > 0:
            cfg["temperature"] = self.temperature
        if self.top_p > 0:
            cfg["top_p"] = self.top_p
        return cfg

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        d = cls()
        return cls(
            url=_s("YROBOT_REALTIME_URL", d.url),
            mode=_s("YROBOT_MODE", d.mode),
            tls_verify=_b("YROBOT_TLS_VERIFY", d.tls_verify),
            system_prompt=_s("YROBOT_SYSTEM_PROMPT", d.system_prompt),
            length_penalty=_f("YROBOT_LENGTH_PENALTY", d.length_penalty),
            force_listen_count=_i("YROBOT_FORCE_LISTEN_COUNT", d.force_listen_count),
            temperature=_f("YROBOT_TEMPERATURE", d.temperature),
            top_p=_f("YROBOT_TOP_P", d.top_p),
            chunk_ms=_i("YROBOT_CHUNK_MS", d.chunk_ms),
            send_video=_b("YROBOT_SEND_VIDEO", d.send_video),
            frame_active_s=_f("YROBOT_FRAME_ACTIVE_S", d.frame_active_s),
            frame_idle_s=_f("YROBOT_FRAME_IDLE_S", d.frame_idle_s),
            model_out_sr=_i("YROBOT_MODEL_OUT_SR", d.model_out_sr),
            kv_soft=_i("YROBOT_KV_SOFT", d.kv_soft),
            kv_hard=_i("YROBOT_KV_HARD", d.kv_hard),
            session_max_s=_f("YROBOT_SESSION_MAX_S", d.session_max_s),
            reconnect_initial_s=_f("YROBOT_RECONNECT_INITIAL", d.reconnect_initial_s),
            reconnect_max_s=_f("YROBOT_RECONNECT_MAX", d.reconnect_max_s),
            gate_ratio=_f("YROBOT_GATE_RATIO", d.gate_ratio),
            gate_min_rms=_f("YROBOT_GATE_MIN_RMS", d.gate_min_rms),
            gate_barge_mult=_f("YROBOT_GATE_BARGE_MULT", d.gate_barge_mult),
            onset_ms=_f("YROBOT_ONSET_MS", d.onset_ms),
            barge_onset_ms=_f("YROBOT_BARGE_ONSET_MS", d.barge_onset_ms),
            release_ms=_f("YROBOT_RELEASE_MS", d.release_ms),
            floor_tau_s=_f("YROBOT_FLOOR_TAU_S", d.floor_tau_s),
            agc_target_rms=_f("YROBOT_AGC_TARGET_RMS", d.agc_target_rms),
            agc_max_gain=_f("YROBOT_AGC_MAX_GAIN", d.agc_max_gain),
            quiet_s=_f("YROBOT_QUIET_S", d.quiet_s),
            hold_max_s=_f("YROBOT_HOLD_MAX_S", d.hold_max_s),
            reforce_s=_f("YROBOT_REFORCE_S", d.reforce_s),
            motion=_b("YROBOT_MOTION", d.motion),
            motion_hz=_f("YROBOT_MOTION_HZ", d.motion_hz),
            track_weight_idle=_f("YROBOT_TRACK_IDLE", d.track_weight_idle),
            track_weight_listen=_f("YROBOT_TRACK_LISTEN", d.track_weight_listen),
            track_weight_speak=_f("YROBOT_TRACK_SPEAK", d.track_weight_speak),
            doa=_b("YROBOT_DOA", d.doa),
            doa_min_turn_rad=_f("YROBOT_DOA_MIN_TURN_RAD", d.doa_min_turn_rad),
            log_level=_s("LOG_LEVEL", d.log_level),
        )
