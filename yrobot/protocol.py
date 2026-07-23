"""MiniCPM-o 4.5 video Realtime wire protocol.

The module deliberately contains no transport or device code.  It only turns
validated values into public-protocol messages, parses server messages, and
checks their lifecycle ordering.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal, TypeAlias, TypedDict
from urllib.parse import parse_qs, urlsplit

INPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_SAMPLE_RATE_HZ = 24_000
CHANNELS = 1
F32_BYTES = 4


class ProtocolError(ValueError):
    """A MiniCPM-o wire message violates the public protocol."""


class ProtocolStateError(ProtocolError):
    """A valid wire message occurred at an invalid lifecycle point."""


class SessionInit(TypedDict):
    type: Literal["session.init"]
    payload: dict[str, object]


class InputAppendPayload(TypedDict):
    audio: str
    video_frames: list[str]
    force_listen: bool
    max_slice_nums: int


class InputAppend(TypedDict):
    type: Literal["input.append"]
    input: InputAppendPayload


class SessionClose(TypedDict):
    type: Literal["session.close"]
    reason: str


ClientEvent: TypeAlias = SessionInit | InputAppend | SessionClose


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Decoded 24 kHz mono float32 little-endian output audio."""

    pcm_f32le: bytes
    sample_rate_hz: int = field(init=False, default=OUTPUT_SAMPLE_RATE_HZ)
    channels: int = field(init=False, default=CHANNELS)

    @property
    def sample_count(self) -> int:
        return len(self.pcm_f32le) // F32_BYTES

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class QueueStatus:
    event_type: Literal["session.queued", "session.queue_update"]
    position: int
    estimated_wait_s: float | None = None
    ticket_id: str | None = None
    queue_length: int | None = None


@dataclass(frozen=True, slots=True)
class QueueDone:
    event_type: Literal["session.queue_done"] = field(init=False, default="session.queue_done")


@dataclass(frozen=True, slots=True)
class SessionCreated:
    session_id: str
    mode: Literal["full_duplex"]
    metrics: dict[str, object]
    server_send_ts: float | None = None
    event_type: Literal["session.created"] = field(init=False, default="session.created")


@dataclass(frozen=True, slots=True)
class ResponseDelta:
    kind: Literal["listen", "text", "audio"]
    metrics: dict[str, object]
    session_id: str | None = None
    response_id: str | None = None
    input_id: str | None = None
    text: str | None = None
    audio: AudioChunk | None = None
    server_send_ts: float | None = None
    event_type: Literal["response.output.delta"] = field(
        init=False, default="response.output.delta"
    )


@dataclass(frozen=True, slots=True)
class SessionClosed:
    reason: str
    session_id: str | None = None
    diagnostic: dict[str, object] | None = None
    server_send_ts: float | None = None
    event_type: Literal["session.closed"] = field(init=False, default="session.closed")


@dataclass(frozen=True, slots=True)
class ServerError:
    error: dict[str, object]
    server_send_ts: float | None = None
    event_type: Literal["error"] = field(init=False, default="error")


ServerEvent: TypeAlias = (
    QueueStatus | QueueDone | SessionCreated | ResponseDelta | SessionClosed | ServerError
)


def validate_video_url(url: str) -> str:
    """Return a valid video Realtime WebSocket URL or raise ``ProtocolError``."""

    if not isinstance(url, str) or not url:
        raise ProtocolError("Realtime URL must be a non-empty string")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ProtocolError("malformed Realtime URL") from exc
    if parsed.scheme not in {"ws", "wss"}:
        raise ProtocolError("Realtime URL scheme must be ws or wss")
    if not parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise ProtocolError("Realtime URL must contain a host and no user info")
    if parsed.path != "/v1/realtime":
        raise ProtocolError("Realtime URL path must be /v1/realtime")
    if parsed.fragment:
        raise ProtocolError("Realtime URL must not contain a fragment")
    try:
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise ProtocolError("malformed Realtime URL query") from exc
    if query != {"mode": ["video"]}:
        raise ProtocolError("Realtime URL query must be exactly mode=video")
    return url


def encode_input_audio(pcm_f32le: bytes | bytearray | memoryview) -> str:
    """Base64-encode one non-empty 16 kHz mono F32LE input chunk."""

    raw = _pcm_bytes(pcm_f32le, label="input audio")
    return base64.b64encode(raw).decode("ascii")


def encode_jpeg(jpeg: bytes | bytearray | memoryview) -> str:
    """Base64-encode one non-empty JPEG frame."""

    if not isinstance(jpeg, bytes | bytearray | memoryview):
        raise ProtocolError("JPEG frame must be bytes-like")
    raw = bytes(jpeg)
    if not raw:
        raise ProtocolError("JPEG frame must not be empty")
    return base64.b64encode(raw).decode("ascii")


def decode_output_audio(audio_base64: str) -> AudioChunk:
    """Strictly decode one 24 kHz mono F32LE output audio delta."""

    raw = _decode_base64(audio_base64, label="output audio")
    if not raw or len(raw) % F32_BYTES:
        raise ProtocolError("output audio must contain complete F32LE samples")
    return AudioChunk(raw)


def session_init(
    system_prompt: str,
    *,
    length_penalty: float | None = None,
    ref_audio_base64: str | None = None,
    tts_ref_audio_base64: str | None = None,
) -> SessionInit:
    """Build the public ``session.init`` message for video full-duplex mode."""

    if not isinstance(system_prompt, str):
        raise ProtocolError("system_prompt must be a string")
    payload: dict[str, object] = {"system_prompt": system_prompt}
    if length_penalty is not None:
        if (
            isinstance(length_penalty, bool)
            or not isinstance(length_penalty, int | float)
            or not math.isfinite(length_penalty)
        ):
            raise ProtocolError("length_penalty must be a finite number")
        payload["config"] = {"length_penalty": float(length_penalty)}

    voice: dict[str, str] = {}
    if ref_audio_base64 is not None:
        _validate_pcm_base64(ref_audio_base64, label="ref_audio_base64")
        voice["ref_audio_base64"] = ref_audio_base64
    if tts_ref_audio_base64 is not None:
        _validate_pcm_base64(tts_ref_audio_base64, label="tts_ref_audio_base64")
        voice["tts_ref_audio_base64"] = tts_ref_audio_base64
    if voice:
        payload["voice"] = voice
    return {"type": "session.init", "payload": payload}


def input_append(
    pcm_f32le: bytes | bytearray | memoryview,
    *,
    video_frames: Sequence[bytes | bytearray | memoryview] = (),
    force_listen: bool = False,
    max_slice_nums: int = 1,
) -> InputAppend:
    """Build one canonical video-mode ``input.append`` message."""

    if not isinstance(force_listen, bool):
        raise ProtocolError("force_listen must be a bool")
    if (
        isinstance(max_slice_nums, bool)
        or not isinstance(max_slice_nums, int)
        or max_slice_nums < 1
    ):
        raise ProtocolError("max_slice_nums must be a positive integer")
    frames = [encode_jpeg(frame) for frame in video_frames]
    return {
        "type": "input.append",
        "input": {
            "audio": encode_input_audio(pcm_f32le),
            "video_frames": frames,
            "force_listen": force_listen,
            "max_slice_nums": max_slice_nums,
        },
    }


def session_close(reason: str = "user_stop") -> SessionClose:
    """Build the public ``session.close`` message."""

    if not isinstance(reason, str) or not reason:
        raise ProtocolError("close reason must be a non-empty string")
    return {"type": "session.close", "reason": reason}


def serialize_client_event(event: ClientEvent) -> str:
    """Serialize a client event as a compact JSON text WebSocket frame."""

    return json.dumps(
        event,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def parse_server_event(raw: str) -> ServerEvent:
    """Parse one strict MiniCPM-o video-mode server JSON text frame."""

    if not isinstance(raw, str):
        raise ProtocolError("server frame must be JSON text")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError("invalid server JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("server event must be a JSON object")
    event_type = _required_str(value, "type")

    if event_type in {"session.queued", "session.queue_update"}:
        return _parse_queue_status(value, event_type)
    if event_type == "session.queue_done":
        _check_keys(value, required={"type"})
        return QueueDone()
    if event_type == "session.created":
        return _parse_session_created(value)
    if event_type == "response.output.delta":
        return _parse_response_delta(value)
    if event_type == "session.closed":
        return _parse_session_closed(value)
    if event_type == "error":
        _check_keys(value, required={"type", "error"}, optional={"server_send_ts"})
        error = value["error"]
        if not isinstance(error, dict):
            raise ProtocolError("error must be a JSON object")
        return ServerError(
            error=error,
            server_send_ts=_optional_float(value, "server_send_ts"),
        )
    raise ProtocolError(f"unsupported server event type: {event_type}")


class Phase(StrEnum):
    CONNECTED = "connected"
    QUEUE_DONE = "queue_done"
    INIT = "init"
    CREATED = "created"
    STREAMING = "streaming"
    CLOSE = "close"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ProtocolState:
    """Immutable MiniCPM-o video-session lifecycle state."""

    phase: Phase = Phase.CONNECTED
    session_id: str | None = None


def transition_client(state: ProtocolState, event: ClientEvent) -> ProtocolState:
    """Advance lifecycle state after sending a validated client event."""

    event_type = event.get("type")
    if event_type == "session.init":
        if state.phase is not Phase.QUEUE_DONE:
            raise ProtocolStateError("session.init requires session.queue_done")
        return replace(state, phase=Phase.INIT)
    if event_type == "input.append":
        if state.phase not in {Phase.CREATED, Phase.STREAMING}:
            raise ProtocolStateError("input.append requires session.created")
        return replace(state, phase=Phase.STREAMING)
    if event_type == "session.close":
        if state.phase not in {Phase.CREATED, Phase.STREAMING}:
            raise ProtocolStateError("session.close requires session.created")
        return replace(state, phase=Phase.CLOSE)
    raise ProtocolError(f"unsupported client event type: {event_type}")


def transition_server(state: ProtocolState, event: ServerEvent) -> ProtocolState:
    """Advance lifecycle state after receiving a parsed server event."""

    if isinstance(event, QueueStatus):
        if state.phase is not Phase.CONNECTED:
            raise ProtocolStateError("queue status received after queue completion")
        return state
    if isinstance(event, QueueDone):
        if state.phase is not Phase.CONNECTED:
            raise ProtocolStateError("session.queue_done is out of order")
        return replace(state, phase=Phase.QUEUE_DONE)
    if isinstance(event, SessionCreated):
        if state.phase is not Phase.INIT:
            raise ProtocolStateError("session.created requires session.init")
        return replace(
            state,
            phase=Phase.CREATED,
            session_id=event.session_id,
        )
    if isinstance(event, ResponseDelta):
        if state.phase is not Phase.STREAMING:
            raise ProtocolStateError("response delta requires streaming input")
        _check_session_id(state, event.session_id)
        return state
    if isinstance(event, SessionClosed):
        if state.phase not in {
            Phase.INIT,
            Phase.CREATED,
            Phase.STREAMING,
            Phase.CLOSE,
        }:
            raise ProtocolStateError("session.closed is out of order")
        _check_session_id(state, event.session_id)
        return replace(state, phase=Phase.CLOSED)
    if isinstance(event, ServerError):
        if state.phase is Phase.CLOSED:
            raise ProtocolStateError("error received after session.closed")
        return state
    raise ProtocolError("unsupported server event")


def _pcm_bytes(
    pcm_f32le: bytes | bytearray | memoryview,
    *,
    label: str,
) -> bytes:
    if not isinstance(pcm_f32le, bytes | bytearray | memoryview):
        raise ProtocolError(f"{label} must be bytes-like")
    try:
        raw = bytes(pcm_f32le)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be bytes-like") from exc
    if not raw or len(raw) % F32_BYTES:
        raise ProtocolError(f"{label} must contain complete F32LE samples")
    return raw


def _decode_base64(value: str, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"{label} is not valid base64") from exc


def _validate_pcm_base64(value: str, *, label: str) -> None:
    raw = _decode_base64(value, label=label)
    if not raw or len(raw) % F32_BYTES:
        raise ProtocolError(f"{label} must contain complete float32 PCM samples")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ProtocolError(f"invalid JSON number: {value}")


def _check_keys(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    if missing:
        raise ProtocolError(f"missing server fields: {', '.join(sorted(missing))}")
    unknown = value.keys() - required - optional
    if unknown:
        raise ProtocolError(f"unknown server fields: {', '.join(sorted(unknown))}")


def _required_str(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ProtocolError(f"{key} must be a non-empty string")
    return item


def _optional_str(value: dict[str, object], key: str) -> str | None:
    if key not in value:
        return None
    return _required_str(value, key)


def _optional_int(value: dict[str, object], key: str) -> int | None:
    if key not in value:
        return None
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ProtocolError(f"{key} must be a non-negative integer")
    return item


def _optional_float(value: dict[str, object], key: str) -> float | None:
    if key not in value:
        return None
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
        raise ProtocolError(f"{key} must be a finite number")
    return float(item)


def _metrics(value: dict[str, object]) -> dict[str, object]:
    item = value.get("metrics", {})
    if not isinstance(item, dict):
        raise ProtocolError("metrics must be a JSON object")
    return item


def _parse_queue_status(
    value: dict[str, object],
    event_type: Literal["session.queued", "session.queue_update"],
) -> QueueStatus:
    _check_keys(
        value,
        required={"type", "position"},
        optional={"estimated_wait_s", "ticket_id", "queue_length"},
    )
    position = _optional_int(value, "position")
    assert position is not None
    wait = value.get("estimated_wait_s")
    if wait is not None and (
        isinstance(wait, bool)
        or not isinstance(wait, int | float)
        or not math.isfinite(wait)
        or wait < 0
    ):
        raise ProtocolError("estimated_wait_s must be a non-negative finite number")
    return QueueStatus(
        event_type=event_type,
        position=position,
        estimated_wait_s=float(wait) if wait is not None else None,
        ticket_id=_optional_str(value, "ticket_id"),
        queue_length=_optional_int(value, "queue_length"),
    )


def _parse_session_created(value: dict[str, object]) -> SessionCreated:
    _check_keys(
        value,
        required={"type", "session_id", "mode"},
        optional={"metrics", "server_send_ts"},
    )
    mode = _required_str(value, "mode")
    if mode != "full_duplex":
        raise ProtocolError("video session mode must be full_duplex")
    return SessionCreated(
        session_id=_required_str(value, "session_id"),
        mode="full_duplex",
        metrics=_metrics(value),
        server_send_ts=_optional_float(value, "server_send_ts"),
    )


def _parse_response_delta(value: dict[str, object]) -> ResponseDelta:
    common = {
        "type",
        "kind",
        "session_id",
        "response_id",
        "input_id",
        "metrics",
        "server_send_ts",
    }
    kind = _required_str(value, "kind")
    if kind == "listen":
        _check_keys(value, required={"type", "kind"}, optional=common - {"type", "kind"})
        return ResponseDelta(
            kind="listen",
            metrics=_metrics(value),
            session_id=_optional_str(value, "session_id"),
            response_id=_optional_str(value, "response_id"),
            input_id=_optional_str(value, "input_id"),
            server_send_ts=_optional_float(value, "server_send_ts"),
        )
    if kind == "text":
        _check_keys(
            value,
            required={"type", "kind", "text"},
            optional=common - {"type", "kind"},
        )
        text = value["text"]
        if not isinstance(text, str):
            raise ProtocolError("text delta must contain a string")
        return ResponseDelta(
            kind="text",
            text=text,
            metrics=_metrics(value),
            session_id=_optional_str(value, "session_id"),
            response_id=_optional_str(value, "response_id"),
            input_id=_optional_str(value, "input_id"),
            server_send_ts=_optional_float(value, "server_send_ts"),
        )
    if kind == "audio":
        _check_keys(
            value,
            required={"type", "kind", "audio"},
            optional=common - {"type", "kind"},
        )
        audio = value["audio"]
        if not isinstance(audio, str):
            raise ProtocolError("audio delta must contain a base64 string")
        return ResponseDelta(
            kind="audio",
            audio=decode_output_audio(audio),
            metrics=_metrics(value),
            session_id=_optional_str(value, "session_id"),
            response_id=_optional_str(value, "response_id"),
            input_id=_optional_str(value, "input_id"),
            server_send_ts=_optional_float(value, "server_send_ts"),
        )
    raise ProtocolError(f"unsupported response delta kind: {kind}")


def _parse_session_closed(value: dict[str, object]) -> SessionClosed:
    _check_keys(
        value,
        required={"type", "reason"},
        optional={"session_id", "diagnostic", "server_send_ts"},
    )
    diagnostic = value.get("diagnostic")
    if diagnostic is not None and not isinstance(diagnostic, dict):
        raise ProtocolError("diagnostic must be a JSON object")
    return SessionClosed(
        reason=_required_str(value, "reason"),
        session_id=_optional_str(value, "session_id"),
        diagnostic=diagnostic,
        server_send_ts=_optional_float(value, "server_send_ts"),
    )


def _check_session_id(state: ProtocolState, received: str | None) -> None:
    if state.session_id is not None and received is not None and received != state.session_id:
        raise ProtocolStateError("server event session_id does not match the session")
