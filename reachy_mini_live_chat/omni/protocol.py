"""Wire codec for the llama.cpp-omni WebSocket ``/backend`` protocol.

Verified against ``tc-mb/llama.cpp-omni`` (``tools/server/ws_handler.cpp`` +
``protocol.cpp``). Everything here is pure (no I/O), so it unit-tests without a server.

Key facts baked in:

* **Audio** is base64 of **raw little-endian float32 PCM**. Input is **mono, 16 kHz**.
  (The server writes it straight into a 16 k/mono/IEEE-float WAV.) Output deltas are
  float32 PCM at the model's TTS rate (~24 kHz; the server does not report the rate).
* First client message is ``session.init``; the server replies ``session.created``.
* In ``full_duplex`` each ``input.append`` carries ~1 s of ``audio`` (+ optional
  ``video_frames``; only the first frame is used) and MUST NOT carry ``messages``.
* Server events: ``response.output.delta`` with ``kind`` ∈ {``text``, ``audio``,
  ``listen``}, ``response.done`` (turn boundary), ``session.closed``.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# base64 <-> float32 PCM
# ---------------------------------------------------------------------------
def pcm_f32_to_b64(pcm: np.ndarray) -> str:
    """Encode a 1-D float32 mono waveform as base64 of raw little-endian float32."""
    arr = np.ascontiguousarray(np.asarray(pcm, dtype="<f4").ravel())
    return base64.b64encode(arr.tobytes()).decode("ascii")


def b64_to_pcm_f32(b64: str) -> np.ndarray:
    """Decode base64 raw little-endian float32 → 1-D float32 array (empty on junk)."""
    if not b64:
        return np.zeros(0, dtype=np.float32)
    # tolerate a leading data-URL prefix, mirroring the server's strip_data_url_prefix
    if "," in b64 and "base64" in b64.split(",", 1)[0]:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    usable = len(raw) - (len(raw) % 4)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw[:usable], dtype="<f4").astype(np.float32)


# ---------------------------------------------------------------------------
# client -> server message builders
# ---------------------------------------------------------------------------
def build_session_init(
    *,
    mode: str = "full_duplex",
    use_tts: bool = True,
    system_prompt: str = "",
    ref_audio_b64: str = "",
    config: Optional[dict] = None,
) -> dict:
    """Build the ``session.init`` message. ``config`` is the optional sampling knobs."""
    payload: dict = {"mode": mode, "use_tts": bool(use_tts)}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if ref_audio_b64:
        payload["voice"] = {"ref_audio": ref_audio_b64, "tts_ref_audio": ref_audio_b64}
    if config:
        payload["config"] = dict(config)
    return {"type": "session.init", "payload": payload}


def build_input_append(
    audio_pcm_f32_16k: np.ndarray,
    *,
    frame_b64: Optional[str] = None,
    force_listen: bool = False,
) -> dict:
    """Build a full-duplex ``input.append``: ~1 s mono float32 16 kHz audio + one frame.

    ``frame_b64`` is a base64 JPEG (no data-URL prefix needed). Only one frame is used
    by the server per append, so we send a single-element list.
    """
    inp: dict = {"audio": pcm_f32_to_b64(audio_pcm_f32_16k)}
    if frame_b64:
        inp["video_frames"] = [frame_b64]
    if force_listen:
        inp["force_listen"] = True
    return {"type": "input.append", "input": inp}


# ---------------------------------------------------------------------------
# server -> client event parsing
# ---------------------------------------------------------------------------
# Normalised event categories (independent of the raw wire ``type`` string).
EV_CREATED = "created"
EV_TEXT = "text"
EV_AUDIO = "audio"
EV_LISTEN = "listen"
EV_DONE = "done"
EV_CLOSED = "closed"
EV_OTHER = "other"


@dataclass
class OmniEvent:
    """A parsed, transport-agnostic server event."""

    category: str                       # one of EV_* above
    session_id: Optional[str] = None
    response_id: Optional[str] = None
    text: str = ""
    audio: Optional[np.ndarray] = None  # float32 mono, at the server's TTS rate
    reason: Optional[str] = None        # response.done reason / session.closed reason
    mode: Optional[str] = None          # session.created mode
    raw: dict = field(default_factory=dict)


def parse_event(msg: dict) -> OmniEvent:
    """Normalise one decoded JSON event into an :class:`OmniEvent`."""
    if not isinstance(msg, dict):
        return OmniEvent(category=EV_OTHER, raw={})

    mtype = msg.get("type")
    session_id = msg.get("session_id")
    response_id = msg.get("response_id")

    if mtype == "session.created":
        return OmniEvent(EV_CREATED, session_id=session_id, mode=msg.get("mode"), raw=msg)

    if mtype == "response.output.delta":
        kind = msg.get("kind")
        if kind == "text":
            return OmniEvent(EV_TEXT, session_id, response_id, text=msg.get("text", "") or "", raw=msg)
        if kind == "audio":
            return OmniEvent(EV_AUDIO, session_id, response_id, audio=b64_to_pcm_f32(msg.get("audio", "")), raw=msg)
        if kind == "listen":
            return OmniEvent(EV_LISTEN, session_id, response_id, raw=msg)
        return OmniEvent(EV_OTHER, session_id, response_id, raw=msg)

    if mtype == "response.done":
        audio_b64 = msg.get("audio")
        audio = b64_to_pcm_f32(audio_b64) if isinstance(audio_b64, str) and audio_b64 else None
        return OmniEvent(
            EV_DONE, session_id, response_id,
            text=msg.get("text", "") or "", audio=audio, reason=msg.get("reason"), raw=msg,
        )

    if mtype == "session.closed":
        return OmniEvent(EV_CLOSED, session_id=session_id, reason=msg.get("reason"), raw=msg)

    return OmniEvent(EV_OTHER, session_id, response_id, raw=msg)


def split_frames(pcm: np.ndarray, frame_len: int) -> List[np.ndarray]:
    """Split a waveform into fixed-length frames (drops a short trailing remainder)."""
    if frame_len <= 0 or len(pcm) < frame_len:
        return []
    n = len(pcm) // frame_len
    return [pcm[i * frame_len:(i + 1) * frame_len] for i in range(n)]
