"""Validated configuration for the Wireless CM4 realtime runtime."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .protocol import validate_video_url

OFFICIAL_REALTIME_URL = "wss://minicpmo45.modelbest.cn/v1/realtime?mode=video"

DEFAULT_SYSTEM_PROMPT = """\
你是 Reachy Mini，一个自然、敏捷、友善的双语机器人。
持续结合用户语音和当前画面理解现场；回答简洁自然，不复述问题。
你可以边听边说。用户插话时立刻停止当前内容，先听清新问题再回答。
不要描述内部协议、推理过程或不存在的机器人能力。"""


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


@dataclass(frozen=True, slots=True)
class Settings:
    realtime_url: str = OFFICIAL_REALTIME_URL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tls_verify: bool = True
    length_penalty: float = 1.1
    session_seconds: float = 285.0
    reconnect_max_seconds: float = 8.0

    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    input_unit_ms: int = 1_000
    local_frame_ms: int = 20
    mic_channel: int = 0
    playback_preroll_ms: int = 0
    playback_buffers: int = 2

    vad_mode: int = 2
    vad_min_rms: float = 0.006
    vad_noise_ratio: float = 2.2
    barge_attack_ms: int = 80
    barge_debounce_ms: int = 350
    near_end_hold_ms: int = 300
    echo_correlation: float = 0.72

    camera_width: int = 640
    camera_jpeg_quality: int = 72
    camera_fps: float = 1.0
    vision_send_interval_seconds: float = 2.0
    doa_hz: float = 10.0
    doa_hold_seconds: float = 3.0
    motion_hz: float = 50.0

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        settings = cls(
            realtime_url=os.getenv(
                "YROBOT_REALTIME_URL",
                os.getenv("MINICPM_REALTIME_URL", OFFICIAL_REALTIME_URL),
            ),
            system_prompt=os.getenv("YROBOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            tls_verify=_bool("YROBOT_TLS_VERIFY", True),
            length_penalty=_float("YROBOT_LENGTH_PENALTY", 1.1),
            session_seconds=_float("YROBOT_SESSION_SECONDS", 285.0),
            reconnect_max_seconds=_float("YROBOT_RECONNECT_MAX_SECONDS", 8.0),
            playback_preroll_ms=_int("YROBOT_PLAYBACK_PREROLL_MS", 0),
            playback_buffers=_int("YROBOT_PLAYBACK_BUFFERS", 2),
            mic_channel=_int("YROBOT_MIC_CHANNEL", 0),
            vad_mode=_int("YROBOT_VAD_MODE", 2),
            vad_min_rms=_float("YROBOT_VAD_MIN_RMS", 0.006),
            vad_noise_ratio=_float("YROBOT_VAD_NOISE_RATIO", 2.2),
            barge_attack_ms=_int("YROBOT_BARGE_ATTACK_MS", 80),
            barge_debounce_ms=_int("YROBOT_BARGE_DEBOUNCE_MS", 350),
            near_end_hold_ms=_int("YROBOT_NEAR_END_HOLD_MS", 300),
            echo_correlation=_float("YROBOT_ECHO_CORRELATION", 0.72),
            camera_width=_int("YROBOT_CAMERA_WIDTH", 640),
            camera_jpeg_quality=_int("YROBOT_CAMERA_JPEG_QUALITY", 72),
            camera_fps=_float("YROBOT_CAMERA_FPS", 1.0),
            vision_send_interval_seconds=_float(
                "YROBOT_VISION_SEND_INTERVAL_SECONDS",
                2.0,
            ),
            doa_hz=_float("YROBOT_DOA_HZ", 10.0),
            doa_hold_seconds=_float("YROBOT_DOA_HOLD_SECONDS", 3.0),
            motion_hz=_float("YROBOT_MOTION_HZ", 50.0),
            log_level=os.getenv("YROBOT_LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        validate_video_url(self.realtime_url)
        if not self.system_prompt.strip():
            raise ValueError("YROBOT_SYSTEM_PROMPT must not be empty")
        if not math.isfinite(self.length_penalty) or self.length_penalty <= 0:
            raise ValueError("YROBOT_LENGTH_PENALTY must be a positive finite number")
        if self.input_sample_rate != 16_000 or self.output_sample_rate != 24_000:
            raise ValueError("MiniCPM-o realtime rates are fixed at 16 kHz in / 24 kHz out")
        if self.input_unit_ms != 1_000:
            raise ValueError("MiniCPM-o full-duplex units must remain at 1000 ms")
        if not 30 <= self.session_seconds < 300:
            raise ValueError("video session duration must be in [30, 300) seconds")
        if not math.isfinite(self.reconnect_max_seconds) or self.reconnect_max_seconds <= 0:
            raise ValueError("YROBOT_RECONNECT_MAX_SECONDS must be positive")
        if self.local_frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD frame size must be 10, 20, or 30 ms")
        if self.mic_channel not in {-1, 0, 1}:
            raise ValueError("YROBOT_MIC_CHANNEL must be -1 (mean), 0, or 1")
        if not 0 <= self.playback_preroll_ms <= 250:
            raise ValueError("YROBOT_PLAYBACK_PREROLL_MS must be 0..250")
        if not 1 <= self.playback_buffers <= 8:
            raise ValueError("YROBOT_PLAYBACK_BUFFERS must be 1..8")
        if self.vad_mode not in range(4):
            raise ValueError("YROBOT_VAD_MODE must be 0..3")
        if (
            not math.isfinite(self.vad_min_rms)
            or not math.isfinite(self.vad_noise_ratio)
            or self.vad_min_rms <= 0
            or self.vad_noise_ratio <= 1
        ):
            raise ValueError("VAD RMS must be positive and noise ratio must be >1")
        if self.barge_attack_ms < self.local_frame_ms:
            raise ValueError("YROBOT_BARGE_ATTACK_MS must cover at least one VAD frame")
        if self.barge_attack_ms % self.local_frame_ms:
            raise ValueError("YROBOT_BARGE_ATTACK_MS must align to VAD frames")
        if self.barge_debounce_ms < 0 or self.near_end_hold_ms < 0:
            raise ValueError("barge debounce and near-end hold must be non-negative")
        if not 0.0 < self.echo_correlation < 1.0:
            raise ValueError("YROBOT_ECHO_CORRELATION must be between 0 and 1")
        if self.camera_width != 640:
            raise ValueError("MiniCPM-o video frames must remain 640 px wide")
        if not 1 <= self.camera_jpeg_quality <= 95:
            raise ValueError("YROBOT_CAMERA_JPEG_QUALITY must be 1..95")
        if not 0 < self.camera_fps <= 1:
            raise ValueError("MiniCPM-o video cadence must not exceed 1 fps")
        if not 1 <= self.vision_send_interval_seconds <= 10:
            raise ValueError("YROBOT_VISION_SEND_INTERVAL_SECONDS must be between 1 and 10")
        if not 10 <= self.doa_hz <= 20:
            raise ValueError("YROBOT_DOA_HZ must be between 10 and 20")
        if not math.isfinite(self.doa_hold_seconds) or self.doa_hold_seconds < 0:
            raise ValueError("YROBOT_DOA_HOLD_SECONDS must be non-negative")
        if self.motion_hz != 50:
            raise ValueError("Reachy Mini Wireless motion loop must remain at 50 Hz")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("YROBOT_LOG_LEVEL is invalid")
