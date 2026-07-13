"""Environment-driven configuration.

All knobs live in one dataclass so every module can be handed a single ``Config``.
Values come from environment variables (loaded from ``.env`` if present); the
defaults match ``.env.example`` and are safe for a hardware-free ``--sim`` run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _flag(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # Language
    lang: str = field(default_factory=lambda: _env("LANG", "auto"))  # auto|zh|en

    # Conversational VLM — MiniCPM-V-4.6 served by llama.cpp (OpenAI-compatible).
    # Handles BOTH text and vision turns: SigLIP2-400M + Qwen3.5-0.8B, so text-only
    # turns are fast and images are understood locally (no cloud, no per-token cost).
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:8080/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openbmb/MiniCPM-V-4.6-gguf"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "not-needed"))

    # Vision turns default to the SAME local MiniCPM-V server. Override to a cloud VLM
    # (e.g. ModelScope Qwen3.7-Plus) only if you'd rather offload image turns.
    vision_base_url: str = field(default_factory=lambda: _env("VISION_BASE_URL", "http://localhost:8080/v1"))
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "openbmb/MiniCPM-V-4.6-gguf"))
    vision_api_key: str = field(default_factory=lambda: _env("VISION_API_KEY", "not-needed"))

    # ASR
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "iic/SenseVoiceSmall"))
    asr_device: str = field(default_factory=lambda: _env("ASR_DEVICE", "cpu"))

    # TTS
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "mlx-community/Kokoro-82M-4bit"))
    tts_voice_zh: str = field(default_factory=lambda: _env("TTS_VOICE_ZH", "zf_xiaobei"))
    tts_voice_en: str = field(default_factory=lambda: _env("TTS_VOICE_EN", "af_heart"))
    tts_speed: float = field(default_factory=lambda: _float("TTS_SPEED", 1.1))

    # VAD / turn
    vad_threshold: float = field(default_factory=lambda: _float("VAD_THRESHOLD", 0.5))
    vad_silence_ms: int = field(default_factory=lambda: _int("VAD_SILENCE_MS", 320))
    vad_min_speech_ms: int = field(default_factory=lambda: _int("VAD_MIN_SPEECH_MS", 200))
    use_semantic_turn: bool = field(default_factory=lambda: _flag("USE_SEMANTIC_TURN", False))

    # Full duplex
    enable_aec: bool = field(default_factory=lambda: _flag("ENABLE_AEC", False))
    barge_in_energy: float = field(default_factory=lambda: _float("BARGE_IN_ENERGY", 0.02))

    # Vision gating
    enable_vision: bool = field(default_factory=lambda: _flag("ENABLE_VISION", True))
    vision_max_edge: int = field(default_factory=lambda: _int("VISION_MAX_EDGE", 768))
    vision_jpeg_quality: int = field(default_factory=lambda: _int("VISION_JPEG_QUALITY", 75))
    vision_phash_threshold: int = field(default_factory=lambda: _int("VISION_PHASH_THRESHOLD", 8))

    # Motion
    enable_motion: bool = field(default_factory=lambda: _flag("ENABLE_MOTION", True))
    enable_doa: bool = field(default_factory=lambda: _flag("ENABLE_DOA", True))
    emotions_dataset: str = field(default_factory=lambda: _env("EMOTIONS_DATASET", "pollen-robotics/reachy-mini-emotions-library"))
    control_hz: int = field(default_factory=lambda: _int("CONTROL_HZ", 100))

    # Web UI
    web_ui: bool = field(default_factory=lambda: _flag("WEB_UI", True))
    web_port: int = field(default_factory=lambda: _int("WEB_PORT", 8042))

    # Runtime flags (set from CLI, not env)
    sim: bool = False
    stub: bool = False

    # Audio format (Reachy media is 16 kHz float32); confirmed from device at runtime.
    sample_rate: int = 16000

    @classmethod
    def load(cls, dotenv: bool = True) -> "Config":
        if dotenv:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except Exception:
                pass
        return cls()

    def as_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        # Never expose secrets over the web UI.
        for k in ("llm_api_key", "vision_api_key"):
            if d.get(k):
                d[k] = "***"
        return d
