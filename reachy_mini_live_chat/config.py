"""Environment-driven configuration.

All knobs live in one dataclass so every module can be handed a single ``Config``.
Values come from environment variables (loaded from ``.env`` if present); the
defaults match ``.env.example``.

The conversational brain is a **remote end-to-end omni model** (MiniCPM-o 4.5 served by
``llama.cpp-omni``'s ``llama-omni-server``) reached over a full-duplex WebSocket — there
are no local ASR/LLM/TTS models. The robot side only does audio/video I/O, the WebSocket,
and 100 Hz motion (DOA + expressive gestures), so it runs comfortably on the Wireless CM4.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Optional


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


def _opt_float(key: str):
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _opt_int(key: str):
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


# Concise *spoken*-dialogue persona. No markdown, no emotion tags: the omni model
# speaks its text end-to-end (server-side TTS), so any tag would be read aloud.
# Body language is inferred separately from the transcript (motion/emotions.py).
DEFAULT_SYSTEM_PROMPT = (
    "你是一个叫 Reachy Mini 的小型桌面机器人，正在和面前的人进行实时的、面对面的语音对话。"
    "你能通过摄像头看到对方，也能听到对方说话。"
    "请遵守：1) 用对方说话的语言回答（中文或英文），语气自然亲切；"
    "2) 这是口语对话，回答要简短、口语化，一般一到两句话，不要用列表、markdown 或表情符号；"
    "3) 你的回答会被实时合成成语音，所以要直奔主题；"
    "4) 只有当对方问到画面内容时才描述你看到的东西。\n"
    "You are Reachy Mini, a small desk robot in a live face-to-face voice chat. Reply in the "
    "user's language, keep it short and spoken, no markdown, get to the point."
)


@dataclass
class Config:
    # Language hint (auto|zh|en) — used for the greeting + motion cue language.
    lang: str = field(default_factory=lambda: _env("LANG", "auto"))

    # -- Omni conversational model (remote, over WebSocket) ------------------
    # Base WS URL (scheme://host:port). The path is derived from `omni_endpoint`:
    #   gateway  → /v1/realtime?mode=<omni_gateway_mode>   (MiniCPM-o-Demo gateway.py, e.g. :8006)
    #   backend  → /backend                                (raw llama-omni-server, e.g. :28099)
    # If OMNI_WS_URL already contains a path, it's used verbatim (escape hatch).
    # Self-signed TLS certs are common → verification off by default.
    omni_ws_url: str = field(default_factory=lambda: _env("OMNI_WS_URL", "wss://10.0.16.187:8006"))
    omni_endpoint: str = field(default_factory=lambda: _env("OMNI_ENDPOINT", "gateway"))  # gateway|backend
    # Gateway routing mode (only for endpoint=gateway): video = audiovisual full-duplex,
    # audio = audio-only full-duplex, chat = turn-based.
    omni_gateway_mode: str = field(default_factory=lambda: _env("OMNI_GATEWAY_MODE", "video"))
    # session.init protocol mode sent to the backend (full_duplex|turn_based).
    omni_mode: str = field(default_factory=lambda: _env("OMNI_MODE", "full_duplex"))
    omni_use_tts: bool = field(default_factory=lambda: _flag("OMNI_USE_TTS", True))
    omni_tls_insecure: bool = field(default_factory=lambda: _flag("OMNI_TLS_INSECURE", True))
    # How long to wait for session.created before streaming audio anyway. The gateway
    # queues sessions and the backend lazy-loads the model on first use (10–60 s), so
    # keep this generous.
    omni_session_ready_s: float = field(default_factory=lambda: _float("OMNI_SESSION_READY_S", 60.0))
    omni_system_prompt: str = field(default_factory=lambda: _env("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    # Optional reference .wav (path) for voice-cloning the robot's spoken voice.
    omni_voice_ref: str = field(default_factory=lambda: _env("OMNI_VOICE_REF", ""))

    # Audio timing. Input MUST be mono float32 PCM @ 16 kHz; one input.append per
    # chunk (the model's 1 Hz time-division period). Output PCM sample rate is not
    # returned by the server; MiniCPM-o token2wav is ~24 kHz — flip to 16000 if the
    # played voice sounds pitched/sped.
    omni_chunk_ms: int = field(default_factory=lambda: _int("OMNI_CHUNK_MS", 1000))
    omni_out_sr: int = field(default_factory=lambda: _int("OMNI_OUT_SR", 24000))
    # Echo/noise/gain on Reachy Mini is done in HARDWARE by the ReSpeaker XVF3800 mic board
    # (always on). omni_respeaker_config writes the tuned XVF3800 params at startup, the way
    # the official app does — this is the real echo fix. Leave it on unless you're on
    # hardware without that board.
    omni_respeaker_config: bool = field(default_factory=lambda: _flag("OMNI_RESPEAKER_CONFIG", True))

    # On barge-in, immediately send the partially-filled capture buffer (with
    # force_listen) instead of waiting for the 1 s chunk boundary — the server learns
    # about the interruption up to ~1 s sooner, so it stops generating sooner.
    omni_barge_flush: bool = field(default_factory=lambda: _flag("OMNI_BARGE_FLUSH", True))

    omni_reconnect_s: float = field(default_factory=lambda: _float("OMNI_RECONNECT_S", 1.5))
    # Diagnostics: when set to a file path, every uplink chunk (exactly what the model
    # hears) is appended as raw s16le mono 16 kHz. Play: ffplay -f s16le -ar 16000 -i <path>
    omni_dump_uplink: str = field(default_factory=lambda: _env("OMNI_DUMP_UPLINK", ""))
    # Fixed software gain applied to the mic before the VAD + uplink (clipped to ±1).
    # The XVF3800 does AGC in hardware, but if speech still reaches the model too quiet
    # (uplink rms peak <~0.08 while talking → the model treats it as background and only
    # listens), raise this to 2.0–4.0. Ratio-based VAD is unaffected by a constant gain.
    omni_mic_gain: float = field(default_factory=lambda: _float("OMNI_MIC_GAIN", 1.0))
    # Software AGC on the uplink only: measure the RMS of frames the VAD marks as speech
    # and scale the audio sent to the model so speech lands near OMNI_MIC_AGC_TARGET.
    # This is the robust fix for "I talk but the robot never answers" — too-quiet speech
    # makes the omni model treat the user as background noise and emit listen forever.
    # Never attenuates (gain >= 1; the hardware AGC handles "too loud") and is applied
    # AFTER the VAD, so the adaptive noise floor never sees a changing gain.
    omni_mic_agc: bool = field(default_factory=lambda: _flag("OMNI_MIC_AGC", True))
    omni_mic_agc_target: float = field(default_factory=lambda: _float("OMNI_MIC_AGC_TARGET", 0.12))
    omni_mic_agc_max_gain: float = field(default_factory=lambda: _float("OMNI_MIC_AGC_MAX_GAIN", 8.0))
    # Playback pacing: feed the speaker in ~60 ms buffers and stay ~200 ms ahead of real
    # time. The cushion absorbs CPU/scheduling jitter on the CM4 so speech doesn't stutter;
    # pacing to the cushion keeps latency bounded if the server produces audio fast.
    omni_playback_chunk_ms: int = field(default_factory=lambda: _int("OMNI_PLAYBACK_CHUNK_MS", 60))
    omni_playback_cushion_ms: int = field(default_factory=lambda: _int("OMNI_PLAYBACK_CUSHION_MS", 200))

    # Video → omni: attach a current frame to input.append. Sending a frame every chunk
    # makes the server run its vision encoder ~1×/s; if its GPU can't sustain vision+audio
    # at real time, a backlog builds (laggy/choppy speech). Raise OMNI_VIDEO_EVERY_N (e.g.
    # 3) to attach a frame only every Nth chunk — cheaper on the server, still grounded.
    omni_send_video: bool = field(default_factory=lambda: _flag("OMNI_SEND_VIDEO", True))
    omni_video_every_n: int = field(default_factory=lambda: max(1, _int("OMNI_VIDEO_EVERY_N", 1)))
    omni_video_fps: float = field(default_factory=lambda: _float("OMNI_VIDEO_FPS", 1.0))
    omni_video_max_edge: int = field(default_factory=lambda: _int("OMNI_VIDEO_MAX_EDGE", 448))
    omni_video_jpeg_quality: int = field(default_factory=lambda: _int("OMNI_VIDEO_JPEG_QUALITY", 70))

    # Optional sampling/decoding passthrough → session.init.config (unset = model default).
    omni_temperature: Optional[float] = field(default_factory=lambda: _opt_float("OMNI_TEMPERATURE"))
    omni_top_p: Optional[float] = field(default_factory=lambda: _opt_float("OMNI_TOP_P"))
    omni_top_k: Optional[int] = field(default_factory=lambda: _opt_int("OMNI_TOP_K"))
    omni_listen_prob_scale: Optional[float] = field(default_factory=lambda: _opt_float("OMNI_LISTEN_PROB_SCALE"))
    omni_force_listen_count: Optional[int] = field(default_factory=lambda: _opt_int("OMNI_FORCE_LISTEN_COUNT"))
    omni_max_speak_tokens_per_chunk: Optional[int] = field(default_factory=lambda: _opt_int("OMNI_MAX_SPEAK_TOKENS_PER_CHUNK"))
    omni_tts_temperature: Optional[float] = field(default_factory=lambda: _opt_float("OMNI_TTS_TEMPERATURE"))

    # -- VAD (energy gate on the AEC'd mic; drives DOA / listen-mood / barge-in) ----
    # The AEC'd mic is the only signal free of the robot's own voice, which is what
    # makes barge-in work. (The XVF3800's own speech flag runs PRE-AEC — it hears the
    # robot's speaker and servos — so it's only used for the DOA angle, never voice.)
    vad_threshold: float = field(default_factory=lambda: _float("VAD_THRESHOLD", 0.5))
    vad_silence_ms: int = field(default_factory=lambda: _int("VAD_SILENCE_MS", 320))
    vad_min_speech_ms: int = field(default_factory=lambda: _int("VAD_MIN_SPEECH_MS", 200))
    # Shorter onset gate while the robot is speaking = how fast you can cut it off.
    vad_barge_min_speech_ms: int = field(default_factory=lambda: _int("VAD_BARGE_MIN_SPEECH_MS", 100))

    # -- Motion --------------------------------------------------------------
    enable_motion: bool = field(default_factory=lambda: _flag("ENABLE_MOTION", True))
    enable_doa: bool = field(default_factory=lambda: _flag("ENABLE_DOA", True))
    # Daemon-native speech wobble (mini.enable_wobbling): the daemon taps the playback
    # pipeline and composes head offsets synced to the *actual* audio clock — better
    # than our app-level approximation. Auto-detected; falls back to the app-level
    # oscillators when the SDK doesn't have it.
    enable_daemon_wobble: bool = field(default_factory=lambda: _flag("ENABLE_DAEMON_WOBBLE", True))
    # Daemon-native visual face tracking (mini.start_head_tracking): the robot looks
    # at the person even when nobody is talking. Paused (weight 0) while the robot
    # speaks so the speech wobble owns the head, like the official app. Auto-detected.
    enable_face_tracking: bool = field(default_factory=lambda: _flag("ENABLE_FACE_TRACKING", True))
    emotions_dataset: str = field(default_factory=lambda: _env("EMOTIONS_DATASET", "pollen-robotics/reachy-mini-emotions-library"))
    # 30 Hz keeps set_target IPC light on the CM4 (each call competes with gstreamer +
    # the daemon); the EMA smoothing keeps motion fluid at this rate. Raise on a laptop.
    control_hz: int = field(default_factory=lambda: _int("CONTROL_HZ", 30))

    # -- Web UI --------------------------------------------------------------
    web_ui: bool = field(default_factory=lambda: _flag("WEB_UI", True))
    web_port: int = field(default_factory=lambda: _int("WEB_PORT", 8042))

    # Audio format (Reachy media is 16 kHz float32); confirmed from device at runtime.
    sample_rate: int = 16000

    @property
    def omni_video_active(self) -> bool:
        """Send camera frames? Never in an audio-only session (sending video into an
        audio duplex session confuses the backend → it stops responding)."""
        if not self.omni_send_video:
            return False
        return "mode=audio" not in self.omni_backend_url

    @property
    def omni_backend_url(self) -> str:
        """Resolve the WS URL to connect to (adds the right path unless one is given)."""
        url = self.omni_ws_url.rstrip("/")
        after_scheme = url.split("://", 1)[-1]
        if "/" in after_scheme:  # explicit path/query provided → use verbatim
            return self.omni_ws_url
        if self.omni_endpoint == "gateway":
            return f"{url}/v1/realtime?mode={self.omni_gateway_mode}"
        return f"{url}/backend"

    def omni_sampling_config(self) -> dict:
        """Assemble the optional session.init `config` object from set knobs only."""
        cfg = {
            "temperature": self.omni_temperature,
            "top_p": self.omni_top_p,
            "top_k": self.omni_top_k,
            "listen_prob_scale": self.omni_listen_prob_scale,
            "force_listen_count": self.omni_force_listen_count,
            "max_new_speak_tokens_per_chunk": self.omni_max_speak_tokens_per_chunk,
            "tts_temperature": self.omni_tts_temperature,
        }
        return {k: v for k, v in cfg.items() if v is not None}

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
        # surface the resolved backend URL + sampling config for the web UI
        d["omni_backend_url"] = self.omni_backend_url
        d["omni_sampling"] = self.omni_sampling_config()
        return d
