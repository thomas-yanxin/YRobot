"""Low-latency Reachy Mini media boundary for realtime conversations.

The engine deliberately has three owners: the capture thread is the only
microphone reader, the camera thread owns potentially slow frame capture, and
the playback thread is the only code that pushes or flushes speaker audio.
Network callbacks only copy data into bounded queues.
"""

from __future__ import annotations

import io
import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import correlate, firwin, lfilter

try:  # Reachy's app image supplies this; tests inject a deterministic fake.
    import webrtcvad
except ImportError:  # pragma: no cover - exercised only on an incomplete robot image.
    webrtcvad = None

log = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16_000
MODEL_SAMPLE_RATE = 24_000
VAD_FRAME_SECONDS = 0.020
VAD_FRAME_SAMPLES = round(INPUT_SAMPLE_RATE * VAD_FRAME_SECONDS)
UPLINK_UNIT_SAMPLES = INPUT_SAMPLE_RATE
PLAYBACK_FRAME_SAMPLES = VAD_FRAME_SAMPLES
ECHO_REFERENCE_FRAMES = 24
ECHO_CORRELATION_THRESHOLD = 0.88
DEVICE_PLAYOUT_TAIL_SECONDS = 0.180
MAX_UPLINK_CAPTURE_AGE_SECONDS = 1.25

AUDIO_STARTUP_CONFIG = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)

StateCallback = Callable[[str], None]
ErrorCallback = Callable[[BaseException], None]


class PlayerClearError(RuntimeError):
    """The speaker queue could not be proven empty after invalidation."""


def _mono_channel_zero(samples: np.ndarray) -> np.ndarray:
    """Copy Reachy's post-AEC channel zero as contiguous float32 mono."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return np.ascontiguousarray(audio)
    if audio.ndim == 2 and audio.shape[1] >= 1:
        return np.ascontiguousarray(audio[:, 0])
    raise ValueError(f"unexpected Reachy microphone shape: {audio.shape}")


def _pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return np.asarray(np.rint(clipped * 32767.0), dtype="<i2").tobytes()


class _FarEndEchoGuard:
    """Conservatively reject mic frames that match recently played PCM."""

    def __init__(self) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=ECHO_REFERENCE_FRAMES)
        self._lock = threading.Lock()

    def remember(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append(np.asarray(frame, dtype=np.float32).copy())

    def matches(self, mic_frame: np.ndarray) -> bool:
        with self._lock:
            if not self._frames:
                return False
            reference = np.concatenate(tuple(self._frames)).astype(np.float64, copy=False)
        frame = np.asarray(mic_frame, dtype=np.float64)
        if reference.size < frame.size:
            return False

        centered = frame - np.mean(frame)
        frame_energy = float(np.dot(centered, centered))
        if frame_energy <= 1e-9:
            return False

        # Check every sample offset: physical speaker/microphone delay is not
        # quantized to a convenient capture-frame boundary.
        dots = correlate(reference, centered, mode="valid", method="fft")
        prefix = np.concatenate(([0.0], np.cumsum(reference)))
        square_prefix = np.concatenate(([0.0], np.cumsum(np.square(reference))))
        window_sum = prefix[frame.size :] - prefix[: -frame.size]
        window_square_sum = square_prefix[frame.size :] - square_prefix[: -frame.size]
        window_energy = window_square_sum - np.square(window_sum) / frame.size
        valid = window_energy > 1e-9
        if not np.any(valid):
            return False
        correlation = np.max(np.abs(dots[valid]) / np.sqrt(frame_energy * window_energy[valid]))
        return bool(correlation >= ECHO_CORRELATION_THRESHOLD)


class StreamingResampler24To16:
    """Stateful anti-aliased 24 kHz to 16 kHz rational resampler.

    Upsampling, FIR state, and decimation phase all survive delta boundaries,
    so splitting a model response into arbitrary WebSocket messages does not
    introduce a click or change the resulting samples.
    """

    _UP = 2
    _DOWN = 3

    def __init__(self, taps: int = 49) -> None:
        if taps < 3 or taps % 2 == 0:
            raise ValueError("taps must be an odd integer of at least 3")
        # The coefficients operate after zero insertion. Multiplying by the
        # interpolation factor preserves unity gain in the pass band.
        self._taps = (
            firwin(taps, 1.0 / max(self._UP, self._DOWN), window=("kaiser", 5.0)) * self._UP
        ).astype(np.float64)
        self._zi = np.zeros(taps - 1, dtype=np.float64)
        self._input_offset_mod_down = 0

    def reset(self) -> None:
        self._zi.fill(0.0)
        self._input_offset_mod_down = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        source = np.asarray(samples, dtype=np.float32)
        if source.ndim != 1:
            raise ValueError("model playback audio must be mono")
        if source.size == 0:
            return np.empty(0, dtype=np.float32)

        upsampled = np.zeros(source.size * self._UP, dtype=np.float64)
        upsampled[:: self._UP] = source
        filtered, self._zi = lfilter(self._taps, (1.0,), upsampled, zi=self._zi)

        first = (-self._input_offset_mod_down) % self._DOWN
        output = filtered[first :: self._DOWN]
        self._input_offset_mod_down = (self._input_offset_mod_down + upsampled.size) % self._DOWN
        return np.asarray(output, dtype=np.float32)


@dataclass(slots=True)
class _PlaybackItem:
    epoch: int
    response_id: str
    samples: np.ndarray
    received_at: float
    delta_first: bool


@dataclass(slots=True)
class _UplinkItem:
    generation: int
    samples: np.ndarray
    captured_at: float


@dataclass(slots=True)
class _ClearRequest:
    reason: str
    requested_at: float
    external_metrics: Mapping[str, Any] | None = None


class AudioEngine:
    """Threaded media boundary implementing the realtime client's port.

    All public handlers are synchronous and bounded. Device I/O is deferred to
    workers, while :meth:`next_audio_unit` is intentionally blocking so an
    asyncio caller can use ``asyncio.to_thread`` without polling.
    """

    def __init__(
        self,
        mini: object,
        *,
        state_callback: StateCallback | None = None,
        error_callback: ErrorCallback | None = None,
        capture_video: bool = True,
        configure_audio: bool = True,
        vad: object | None = None,
        vad_mode: int = 2,
        vad_onset_frames: int = 2,
        vad_release_frames: int = 10,
        vad_rms_threshold: float = 0.008,
        uplink_queue_size: int = 2,
        uplink_unit_samples: int = UPLINK_UNIT_SAMPLES,
        camera_fps: float = 1.0,
        camera_idle_fps: float = 0.2,
        camera_active_hold_seconds: float = 3.0,
        playback_lead_seconds: float = 0.040,
        playback_queue_seconds: float = 12.0,
        join_timeout: float = 3.0,
    ) -> None:
        if vad_onset_frames not in {2, 3}:
            raise ValueError("vad_onset_frames must be 2 or 3")
        if vad_release_frames < 1:
            raise ValueError("vad_release_frames must be positive")
        if vad_rms_threshold < 0.0:
            raise ValueError("vad_rms_threshold must not be negative")
        if uplink_queue_size < 1:
            raise ValueError("uplink_queue_size must be positive")
        if (
            not INPUT_SAMPLE_RATE // 2 <= uplink_unit_samples <= INPUT_SAMPLE_RATE
            or uplink_unit_samples % VAD_FRAME_SAMPLES
        ):
            raise ValueError(
                "uplink_unit_samples must represent 500-1000 ms in 20 ms increments"
            )
        if camera_fps <= 0.0:
            raise ValueError("camera_fps must be positive")
        if camera_idle_fps <= 0.0 or camera_idle_fps > camera_fps:
            raise ValueError("camera_idle_fps must be positive and no greater than camera_fps")
        if camera_active_hold_seconds < 0.0:
            raise ValueError("camera_active_hold_seconds must not be negative")
        if not 0.020 <= playback_lead_seconds <= 0.150:
            raise ValueError("playback_lead_seconds must be between 0.020 and 0.150")
        if playback_queue_seconds < playback_lead_seconds:
            raise ValueError("playback_queue_seconds must be at least the playback lead")
        if join_timeout <= 0.0:
            raise ValueError("join_timeout must be positive")

        if vad is None:
            if webrtcvad is None:
                raise RuntimeError(
                    "webrtcvad is required on the robot; inject vad= only in deterministic tests"
                )
            vad = webrtcvad.Vad(vad_mode)

        self.mini = mini
        self._state_callback = state_callback
        self._error_callback = error_callback
        self._capture_video = capture_video
        self._configure_audio = configure_audio
        self._audio_config_attempted = False
        self._vad = vad
        self._vad_onset_frames = vad_onset_frames
        self._vad_release_frames = vad_release_frames
        self._vad_rms_threshold = vad_rms_threshold
        self._uplink_unit_samples = uplink_unit_samples
        self._camera_active_period = 1.0 / camera_fps
        self._camera_idle_period = 1.0 / camera_idle_fps
        self._camera_active_hold_seconds = camera_active_hold_seconds
        self._playback_lead_seconds = playback_lead_seconds
        self._lead_frames = max(1, math.ceil(playback_lead_seconds / VAD_FRAME_SECONDS))
        self._playback_capacity = max(
            self._lead_frames,
            math.ceil(playback_queue_seconds / VAD_FRAME_SECONDS),
        )
        self._join_timeout = join_timeout

        self._uplink: queue.Queue[_UplinkItem] = queue.Queue(maxsize=uplink_queue_size)
        self._uplink_generation = 0
        self._capture_reset_generation = 0
        self._camera_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._camera_wake = threading.Event()

        self._lock = threading.RLock()
        self._playback_ready = threading.Condition(self._lock)
        self._playback: deque[_PlaybackItem] = deque()
        self._playback_inflight = False
        self._clear_requests: deque[_ClearRequest] = deque()
        self._resampler = StreamingResampler24To16()
        self._echo_guard = _FarEndEchoGuard()
        self._resampled_tail = np.empty(0, dtype=np.float32)
        self._epoch = 0
        self._active_response_id: str | None = None
        self._turn_response_ids: set[str] = set()
        self._invalid_response_ids: set[str] = set()
        self._session_ready = False
        self._server_listening = True
        self._model_speaking = False
        self._user_speaking = False
        self._audible_until = 0.0
        self._drop_until_listen = False
        self._drop_overflowed_turn = False
        self._listen_seen_after_interrupt = False
        self._force_listen = False
        self._interrupt_onset_at: float | None = None
        self._force_dispatch_recorded = False
        self._last_user_activity_at = 0.0
        self._state = "idle"

        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._started = False
        self._accepting_media = False
        self._recording = False
        self._playing = False
        self._capture_thread: threading.Thread | None = None
        self._camera_thread: threading.Thread | None = None
        self._playback_thread: threading.Thread | None = None

        self._metrics: dict[str, int | float] = {
            "uplink_units": 0,
            "uplink_dropped_units": 0,
            "uplink_stale_units": 0,
            "force_control_units": 0,
            "playback_frames_enqueued": 0,
            "playback_frames_pushed": 0,
            "playback_dropped_frames": 0,
            "playback_overflowed_turns": 0,
            "stale_playback_frames": 0,
            "interruptions": 0,
            "clear_player_requests": 0,
            "clear_player_successes": 0,
            "clear_player_failures": 0,
            "capture_errors": 0,
            "camera_errors": 0,
            "camera_frames_captured": 0,
            "camera_frames_consumed": 0,
            "playback_errors": 0,
            "audio_config_successes": 0,
            "audio_config_failures": 0,
            "audio_config_unavailable": 0,
            "slow_server_events": 0,
            "self_echo_frames_suppressed": 0,
        }
        self._last_error: BaseException | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def last_error(self) -> BaseException | None:
        with self._lock:
            return self._last_error

    @property
    def metrics(self) -> dict[str, int | float]:
        """Return a snapshot, including current bounded-queue depths."""
        with self._lock:
            snapshot = dict(self._metrics)
            snapshot["playback_queue_depth"] = len(self._playback)
            snapshot["playback_queue_capacity"] = self._playback_capacity
            snapshot["epoch"] = self._epoch
        snapshot["uplink_queue_depth"] = self._uplink.qsize()
        snapshot["uplink_queue_capacity"] = self._uplink.maxsize
        return snapshot

    @property
    def worker_threads(self) -> tuple[threading.Thread, ...]:
        return tuple(
            thread
            for thread in (
                self._capture_thread,
                self._camera_thread,
                self._playback_thread,
            )
            if thread is not None
        )

    def start(self, *, session_ready: bool = True) -> None:
        """Start Reachy media once and launch independent bounded workers."""
        with self._lifecycle_lock:
            if self._started:
                return
            self._stop_event.clear()
            try:
                self.mini.media.start_recording()
                self._recording = True
                self.mini.media.start_playing()
                self._playing = True
                self._bound_device_playback_queue()
                self._apply_audio_startup_config()
            except Exception:
                self._stop_media()
                raise

            suffix = f"-{id(self):x}"
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"yrobot-audio-capture{suffix}",
            )
            self._playback_thread = threading.Thread(
                target=self._playback_loop,
                name=f"yrobot-audio-playback{suffix}",
            )
            if self._capture_video:
                self._camera_thread = threading.Thread(
                    target=self._camera_loop,
                    name=f"yrobot-camera{suffix}",
                )
            else:
                self._camera_thread = None

            self._started = True
            with self._lock:
                self._accepting_media = True
            # Establish clean generations before capture/camera can publish
            # their first sample. The playback worker consumes the queued
            # device flush immediately after it starts.
            self.invalidate_session("start")
            self._playback_thread.start()
            self._capture_thread.start()
            if self._camera_thread is not None:
                self._camera_thread.start()
        if session_ready:
            self.handle_session_ready()

    def stop(self) -> None:
        """Invalidate pending media, flush the player, and join every worker."""
        with self._lifecycle_lock:
            if not self._started:
                return
            with self._lock:
                self._accepting_media = False
            self.invalidate_session("stop")
            self._stop_event.set()
            self._camera_wake.set()
            with self._playback_ready:
                self._playback_ready.notify_all()

            threads = self.worker_threads
            for thread in threads:
                thread.join(timeout=self._join_timeout)
            # Stopping media can release a backend call that did not honor the
            # stop event (notably a blocked local capture read). Give it one
            # final bounded join before reporting a leaked worker.
            self._stop_media()
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=self._join_timeout)
            alive = [thread.name for thread in threads if thread.is_alive()]
            if alive:
                error = RuntimeError(f"media worker(s) did not stop: {', '.join(alive)}")
                self._record_error(error, "playback_errors")

            self._drain_uplink()
            self._started = False

    def next_audio_unit(self, timeout: float) -> tuple[np.ndarray, bool] | None:
        """Return one configured channel-zero mic unit and its force latch."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item = self._uplink.get(timeout=remaining)
            except queue.Empty:
                return None
            with self._lock:
                now = time.monotonic()
                if item.generation != self._uplink_generation:
                    self._metrics["uplink_dropped_units"] += 1
                    if now >= deadline:
                        return None
                    continue
                capture_age = now - item.captured_at
                if capture_age > MAX_UPLINK_CAPTURE_AGE_SECONDS:
                    self._metrics["uplink_dropped_units"] += 1
                    self._metrics["uplink_stale_units"] += 1
                    if now >= deadline:
                        return None
                    continue
                self._metrics["last_uplink_capture_age_ms"] = capture_age * 1000.0
                self._metrics["max_uplink_capture_age_ms"] = max(
                    capture_age * 1000.0,
                    float(self._metrics.get("max_uplink_capture_age_ms", 0.0)),
                )
                force_listen = self._force_listen
                if (
                    force_listen
                    and not self._force_dispatch_recorded
                    and self._interrupt_onset_at is not None
                ):
                    self._metrics["last_force_dispatch_latency_ms"] = (
                        time.monotonic() - self._interrupt_onset_at
                    ) * 1000.0
                    self._force_dispatch_recorded = True
            return item.samples, force_listen

    def latest_frame_jpeg(self) -> bytes | None:
        """Consume the latest immutable JPEG without touching the camera.

        A frame is attached to at most one audio unit, avoiding duplicate
        vision tokens when the audio cadence is faster than the camera cadence.
        """

        with self._camera_lock:
            jpeg = self._latest_jpeg
            self._latest_jpeg = None
        if jpeg is not None:
            with self._lock:
                self._metrics["camera_frames_consumed"] += 1
        return jpeg

    def ready_for_rollover(self) -> bool:
        """Prefer replacing a 300-second video session between turns."""

        with self._lock:
            return bool(
                self._session_ready
                and self._server_listening
                and not self._user_speaking
                and not self._model_speaking
                and not self._force_listen
                and not self._drop_until_listen
                and not self._playout_pending_locked()
            )

    def handle_session_ready(self) -> None:
        """Open a clean microphone epoch once the Realtime backend is ready."""

        with self._playback_ready:
            if not self._accepting_media:
                return
            # Drain before publishing readiness. Draining after unlocking can
            # race with the first new-generation capture unit and delete it.
            self._drain_uplink()
            self._session_ready = True
            self._server_listening = True
            self._uplink_generation += 1
            self._capture_reset_generation += 1
            changed = self._transition_state_locked("listening")
            self._playback_ready.notify_all()
        self._camera_wake.set()
        if changed:
            self._emit_state("listening")

    def handle_audio_delta(
        self,
        samples24: np.ndarray,
        response_id: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Resample and enqueue one model delta without doing device I/O."""
        source = np.asarray(samples24, dtype=np.float32)
        if source.ndim != 1:
            raise ValueError("model playback audio must be mono")
        if source.size == 0:
            return

        self._observe_server_metrics(metrics)
        notify: str | None = None
        with self._playback_ready:
            if not self._accepting_media or not self._session_ready:
                self._metrics["stale_playback_frames"] += max(
                    1, math.ceil(source.size / (MODEL_SAMPLE_RATE * VAD_FRAME_SECONDS))
                )
                return
            if (
                response_id in self._invalid_response_ids
                or self._drop_until_listen
                or self._drop_overflowed_turn
            ):
                self._metrics["stale_playback_frames"] += max(
                    1, math.ceil(source.size / (MODEL_SAMPLE_RATE * VAD_FRAME_SECONDS))
                )
                return

            if self._user_speaking:
                notify = self._interrupt_locked(
                    reason="model_audio_while_user_speaking",
                    response_id=response_id,
                    metrics=metrics,
                )
            else:
                self._active_response_id = response_id
                if response_id:
                    self._turn_response_ids.add(response_id)
                self._server_listening = False
                self._model_speaking = True
                if self._transition_state_locked("speaking"):
                    notify = "speaking"

                converted = self._resampler.process(source)
                if self._resampled_tail.size:
                    converted = np.concatenate((self._resampled_tail, converted))
                complete = (converted.size // PLAYBACK_FRAME_SAMPLES) * PLAYBACK_FRAME_SAMPLES
                now = time.monotonic()
                delta_first = True
                frame_offsets = range(0, complete, PLAYBACK_FRAME_SAMPLES)
                for frame_index, offset in enumerate(frame_offsets):
                    frame = converted[offset : offset + PLAYBACK_FRAME_SAMPLES].copy()
                    admitted = self._append_playback_locked(
                        _PlaybackItem(
                            self._epoch,
                            response_id,
                            frame,
                            now,
                            delta_first,
                        )
                    )
                    if not admitted:
                        remaining = len(frame_offsets) - frame_index - 1
                        self._metrics["playback_dropped_frames"] += remaining
                        self._drop_overflowed_turn = True
                        self._metrics["playback_overflowed_turns"] += 1
                        self._resampled_tail = np.empty(0, dtype=np.float32)
                        break
                    delta_first = False
                if not self._drop_overflowed_turn:
                    self._resampled_tail = converted[complete:].copy()
                self._playback_ready.notify_all()

        if notify is not None:
            self._emit_state(notify)

    def handle_listen(
        self,
        response_id: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Close a natural turn or confirm an interruption flush."""
        del response_id
        self._observe_server_metrics(metrics)
        with self._playback_ready:
            was_interrupted = self._drop_until_listen or self._force_listen
            self._server_listening = True
            self._listen_seen_after_interrupt = was_interrupted
            self._invalid_response_ids.update(self._turn_response_ids)
            self._turn_response_ids.clear()
            if was_interrupted:
                self._invalidate_playback_locked(
                    "interrupt_listen_ack", metrics=metrics, reset_force=False
                )
            else:
                if not self._drop_overflowed_turn:
                    self._enqueue_resampled_tail_locked()
                else:
                    self._resampled_tail = np.empty(0, dtype=np.float32)
                self._resampler.reset()
                self._drop_overflowed_turn = False
            self._active_response_id = None
            if was_interrupted and self._user_speaking:
                self._drop_until_listen = True
                self._force_listen = True
            else:
                self._drop_until_listen = False
                self._force_listen = False
                self._listen_seen_after_interrupt = False
            playout_pending = self._playout_pending_locked()
            self._model_speaking = not was_interrupted and playout_pending
            if self._interrupt_onset_at is not None:
                self._metrics["last_interrupt_to_listen_ms"] = (
                    time.monotonic() - self._interrupt_onset_at
                ) * 1000.0
                if not self._force_listen:
                    self._interrupt_onset_at = None
                    self._force_dispatch_recorded = False
            target_state = "speaking" if self._model_speaking else "listening"
            changed = self._transition_state_locked(target_state)
            self._playback_ready.notify_all()
        if changed:
            self._emit_state(target_state)

    def handle_text(self, text: str, response_id: str) -> None:
        """Text is intentionally not coupled to media timing."""
        del text, response_id

    def invalidate_session(self, reason: str) -> None:
        """Flush both directions for disconnect, reconnect, start, or stop."""
        with self._playback_ready:
            self._invalidate_playback_locked(reason, reset_force=True)
            self._invalid_response_ids.clear()
            self._turn_response_ids.clear()
            self._active_response_id = None
            self._session_ready = False
            self._server_listening = True
            self._model_speaking = False
            self._user_speaking = False
            self._drop_until_listen = False
            self._drop_overflowed_turn = False
            self._listen_seen_after_interrupt = False
            self._force_listen = False
            self._interrupt_onset_at = None
            self._force_dispatch_recorded = False
            self._uplink_generation += 1
            self._capture_reset_generation += 1
            changed = self._transition_state_locked("idle")
            self._playback_ready.notify_all()
        self._drain_uplink()
        with self._camera_lock:
            self._latest_jpeg = None
        self._camera_wake.set()
        if changed:
            self._emit_state("idle")

    def _append_playback_locked(self, item: _PlaybackItem) -> bool:
        if len(self._playback) >= self._playback_capacity:
            self._metrics["playback_dropped_frames"] += 1
            return False
        self._playback.append(item)
        self._metrics["playback_frames_enqueued"] += 1
        return True

    def _enqueue_resampled_tail_locked(self) -> None:
        if self._resampled_tail.size == 0:
            return
        padded = np.zeros(PLAYBACK_FRAME_SAMPLES, dtype=np.float32)
        padded[: self._resampled_tail.size] = self._resampled_tail
        self._append_playback_locked(
            _PlaybackItem(
                self._epoch,
                self._active_response_id or "natural_tail",
                padded,
                time.monotonic(),
                False,
            )
        )
        self._resampled_tail = np.empty(0, dtype=np.float32)
        self._drop_overflowed_turn = False

    def _interrupt_locked(
        self,
        *,
        reason: str,
        response_id: str | None,
        metrics: Mapping[str, Any] | None,
    ) -> str | None:
        if response_id:
            self._invalid_response_ids.add(response_id)
        self._invalid_response_ids.update(self._turn_response_ids)
        self._turn_response_ids.clear()
        self._invalidate_playback_locked(reason, metrics=metrics, reset_force=False)
        self._active_response_id = None
        self._server_listening = False
        self._model_speaking = False
        self._drop_until_listen = True
        self._listen_seen_after_interrupt = False
        self._force_listen = True
        self._interrupt_onset_at = time.monotonic()
        self._force_dispatch_recorded = False
        self._uplink_generation += 1
        self._drain_uplink()
        self._metrics["interruptions"] += 1
        self._playback_ready.notify_all()
        if self._transition_state_locked("interrupted"):
            return "interrupted"
        return None

    def _invalidate_playback_locked(
        self,
        reason: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        reset_force: bool,
    ) -> None:
        self._epoch += 1
        dropped = len(self._playback)
        self._playback.clear()
        self._metrics["stale_playback_frames"] += dropped
        self._resampler.reset()
        self._resampled_tail = np.empty(0, dtype=np.float32)
        self._drop_overflowed_turn = False
        self._audible_until = 0.0
        self._clear_requests.append(_ClearRequest(reason, time.monotonic(), metrics))
        self._metrics["clear_player_requests"] += 1
        if reset_force:
            self._force_listen = False

    def _capture_loop(self) -> None:
        uplink_pending = np.empty(0, dtype=np.float32)
        capture_pending = np.empty(0, dtype=np.float32)
        onset_count = 0
        release_count = 0
        with self._lock:
            capture_generation = self._capture_reset_generation

        while not self._stop_event.is_set():
            try:
                sample = self.mini.media.get_audio_sample()
                if sample is None:
                    self._stop_event.wait(0.005)
                    continue
                mono = _mono_channel_zero(sample)
                if mono.size == 0:
                    continue
                with self._lock:
                    current_generation = self._capture_reset_generation
                    session_ready = self._session_ready
                if not session_ready:
                    uplink_pending = np.empty(0, dtype=np.float32)
                    capture_pending = np.empty(0, dtype=np.float32)
                    onset_count = 0
                    release_count = 0
                    capture_generation = current_generation
                    continue
                if current_generation != capture_generation:
                    uplink_pending = np.empty(0, dtype=np.float32)
                    capture_pending = np.empty(0, dtype=np.float32)
                    onset_count = 0
                    release_count = 0
                    capture_generation = current_generation
                capture_pending = np.concatenate((capture_pending, mono))

                while capture_pending.size >= VAD_FRAME_SAMPLES:
                    frame = capture_pending[:VAD_FRAME_SAMPLES]
                    capture_pending = capture_pending[VAD_FRAME_SAMPLES:]
                    rms = math.sqrt(float(np.mean(np.square(frame, dtype=np.float64))))
                    with self._lock:
                        far_end_active = self._model_speaking or self._playout_pending_locked()
                    echo = bool(
                        far_end_active
                        and rms >= self._vad_rms_threshold
                        and self._echo_guard.matches(frame)
                    )
                    if echo:
                        with self._lock:
                            self._metrics["self_echo_frames_suppressed"] += 1

                    voiced = False
                    if not echo and rms >= self._vad_rms_threshold:
                        voiced = bool(self._vad.is_speech(_pcm16_bytes(frame), INPUT_SAMPLE_RATE))

                    interrupted = False
                    if voiced:
                        onset_count += 1
                        release_count = 0
                        if not self._user_speaking and onset_count >= self._vad_onset_frames:
                            interrupted = self._on_user_speech_onset()
                    else:
                        onset_count = 0
                        if self._user_speaking:
                            release_count += 1
                            if release_count >= self._vad_release_frames:
                                release_count = 0
                                self._on_user_speech_release()

                    # Correlated far-end residue must not reach the model,
                    # otherwise the server can interrupt itself even when the
                    # local VAD correctly rejects the frame.
                    uplink_frame = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32) if echo else frame
                    uplink_pending = np.concatenate((uplink_pending, uplink_frame))

                    if interrupted:
                        control_unit = np.zeros(self._uplink_unit_samples, dtype=np.float32)
                        control_unit[: uplink_pending.size] = uplink_pending
                        uplink_pending = np.empty(0, dtype=np.float32)
                        self._put_uplink(control_unit)
                        with self._lock:
                            self._metrics["force_control_units"] += 1
                    elif uplink_pending.size >= self._uplink_unit_samples:
                        unit = uplink_pending[: self._uplink_unit_samples].copy()
                        uplink_pending = uplink_pending[self._uplink_unit_samples :]
                        self._put_uplink(unit)
            except Exception as exc:
                self._record_error(exc, "capture_errors")
                self._stop_event.wait(0.050)

    def _on_user_speech_onset(self) -> bool:
        notify: str | None = None
        interrupted = False
        with self._playback_ready:
            if self._user_speaking:
                return False
            self._user_speaking = True
            self._last_user_activity_at = time.monotonic()
            if self._model_speaking or self._playout_pending_locked():
                interrupted = True
                notify = self._interrupt_locked(
                    reason="user_barge_in",
                    response_id=self._active_response_id,
                    metrics=None,
                )
        if notify is not None:
            self._emit_state(notify)
        self._camera_wake.set()
        return interrupted

    def _on_user_speech_release(self) -> None:
        with self._playback_ready:
            self._user_speaking = False
            self._last_user_activity_at = time.monotonic()
            if self._listen_seen_after_interrupt:
                self._drop_until_listen = False
                self._force_listen = False
                self._listen_seen_after_interrupt = False
                self._interrupt_onset_at = None
                self._force_dispatch_recorded = False
        self._camera_wake.set()

    def _put_uplink(self, unit: np.ndarray) -> None:
        with self._lock:
            generation = self._uplink_generation
        item = _UplinkItem(generation, unit, time.monotonic())
        try:
            self._uplink.put_nowait(item)
        except queue.Full:
            try:
                self._uplink.get_nowait()
            except queue.Empty:  # pragma: no cover - another consumer won the race.
                pass
            self._uplink.put_nowait(item)
            with self._lock:
                self._metrics["uplink_dropped_units"] += 1
        with self._lock:
            self._metrics["uplink_units"] += 1

    def _drain_uplink(self) -> None:
        while True:
            try:
                self._uplink.get_nowait()
            except queue.Empty:
                return

    def _camera_loop(self) -> None:
        next_capture = time.monotonic()
        while not self._stop_event.is_set():
            delay = next_capture - time.monotonic()
            if delay > 0.0:
                self._camera_wake.wait(delay)
                self._camera_wake.clear()
                if self._stop_event.is_set():
                    return
            try:
                jpeg = self._read_frame_jpeg()
                if jpeg:
                    with self._camera_lock:
                        self._latest_jpeg = jpeg
                    with self._lock:
                        self._metrics["camera_frames_captured"] += 1
            except Exception as exc:
                self._record_error(exc, "camera_errors", level=logging.DEBUG)
            finished = time.monotonic()
            with self._lock:
                active = bool(
                    self._user_speaking
                    or finished - self._last_user_activity_at
                    <= self._camera_active_hold_seconds
                )
            period = self._camera_active_period if active else self._camera_idle_period
            next_capture = finished + period

    def _read_frame_jpeg(self) -> bytes | None:
        get_jpeg = getattr(self.mini.media, "get_frame_jpeg", None)
        if callable(get_jpeg):
            frame = get_jpeg()
            return bytes(frame) if frame else None

        get_frame = getattr(self.mini.media, "get_frame", None)
        if not callable(get_frame):
            return None
        frame = get_frame()
        if frame is None:
            return None
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is a project dependency.
            raise RuntimeError("Pillow is required to encode camera frames") from exc
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame)).save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

    def _playback_loop(self) -> None:
        active_epoch: int | None = None
        first_item_at = 0.0
        primed = False
        next_push_at = 0.0

        while True:
            clear_requests: list[_ClearRequest] = []
            item: _PlaybackItem | None = None
            notify: str | None = None
            with self._playback_ready:
                while item is None and not clear_requests:
                    if self._clear_requests:
                        clear_requests = list(self._clear_requests)
                        self._clear_requests.clear()
                        active_epoch = None
                        primed = False
                        break
                    if self._stop_event.is_set():
                        return

                    while self._playback and self._playback[0].epoch != self._epoch:
                        self._playback.popleft()
                        self._metrics["stale_playback_frames"] += 1
                    if not self._playback:
                        now = time.monotonic()
                        if self._finish_playout_locked(now):
                            notify = "listening"
                            break
                        audible_wait = max(0.0, self._audible_until - now)
                        self._playback_ready.wait(timeout=min(0.050, audible_wait or 0.050))
                        continue

                    head = self._playback[0]
                    now = time.monotonic()
                    if head.epoch != active_epoch:
                        active_epoch = head.epoch
                        first_item_at = head.received_at
                        primed = False
                        next_push_at = now
                    if not primed:
                        lead_elapsed = now - first_item_at
                        if len(self._playback) >= self._lead_frames:
                            primed = True
                        elif lead_elapsed >= self._playback_lead_seconds:
                            primed = True
                        else:
                            self._playback_ready.wait(
                                timeout=self._playback_lead_seconds - lead_elapsed
                            )
                            continue

                    wait_for = next_push_at - now
                    if wait_for > 0.0:
                        self._playback_ready.wait(timeout=wait_for)
                        continue
                    item = self._playback.popleft()
                    self._playback_inflight = True

            if notify is not None:
                self._emit_state(notify)
                continue
            if clear_requests:
                self._clear_player(clear_requests)
                continue
            if item is None:
                continue

            pushed = False
            # Hold the generation lock across the non-blocking appsrc push.
            # An interrupt either wins first, or waits and then flushes this
            # exact frame; there is no push-after-flush stale-audio window.
            with self._playback_ready:
                if (
                    item.epoch == self._epoch
                    and not self._drop_until_listen
                    and not self._clear_requests
                ):
                    try:
                        self.mini.media.push_audio_sample(item.samples)
                        self._echo_guard.remember(item.samples)
                        self._audible_until = max(
                            self._audible_until,
                            time.monotonic() + DEVICE_PLAYOUT_TAIL_SECONDS,
                        )
                        self._metrics["playback_frames_pushed"] += 1
                        if item.delta_first:
                            latency_ms = (time.monotonic() - item.received_at) * 1000.0
                            self._metrics["last_delta_to_speaker_ms"] = latency_ms
                            self._metrics["max_delta_to_speaker_ms"] = max(
                                latency_ms,
                                float(self._metrics.get("max_delta_to_speaker_ms", 0.0)),
                            )
                        pushed = True
                    except Exception as exc:
                        self._record_error_locked(exc, "playback_errors")
                else:
                    self._metrics["stale_playback_frames"] += 1
                self._playback_inflight = False
                self._playback_ready.notify_all()
            if not pushed:
                active_epoch = None
                primed = False
            next_push_at += VAD_FRAME_SECONDS
            pushed_at = time.monotonic()
            if next_push_at < pushed_at - VAD_FRAME_SECONDS:
                next_push_at = pushed_at

    def _clear_player(self, requests: list[_ClearRequest]) -> None:
        clear_player = getattr(getattr(self.mini.media, "audio", None), "clear_player", None)
        if not callable(clear_player):
            error = PlayerClearError("Reachy media.audio.clear_player() is unavailable")
            self._record_error(error, "clear_player_failures")
            return
        try:
            clear_player()
        except Exception as exc:
            error = PlayerClearError(
                f"Reachy clear_player failed after {requests[-1].reason}: {exc}"
            )
            error.__cause__ = exc
            self._record_error(error, "clear_player_failures")
            return
        with self._lock:
            self._metrics["clear_player_successes"] += 1
            self._metrics["last_clear_latency_ms"] = (
                time.monotonic() - requests[-1].requested_at
            ) * 1000.0

    def _apply_audio_startup_config(self) -> None:
        if not self._configure_audio or self._audio_config_attempted:
            return
        self._audio_config_attempted = True
        apply_config = getattr(getattr(self.mini.media, "audio", None), "apply_audio_config", None)
        if not callable(apply_config):
            with self._lock:
                self._metrics["audio_config_unavailable"] += 1
            log.warning("Reachy XVF audio configuration API is unavailable")
            return
        try:
            applied = apply_config(
                AUDIO_STARTUP_CONFIG,
                verify=True,
                write_settle_seconds=0.1,
            )
            if applied is False:
                raise RuntimeError("Reachy XVF audio configuration verification failed")
        except Exception as exc:
            self._record_error(exc, "audio_config_failures", level=logging.WARNING)
            return
        with self._lock:
            self._metrics["audio_config_successes"] += 1

    def _bound_device_playback_queue(self) -> None:
        setter = getattr(getattr(self.mini.media, "audio", None), "set_max_output_buffers", None)
        if not callable(setter):
            return
        setter(6)

    def _playout_pending_locked(self) -> bool:
        return bool(
            self._playback
            or self._playback_inflight
            or self._resampled_tail.size
            or time.monotonic() < self._audible_until
        )

    def _finish_playout_locked(self, now: float) -> bool:
        if (
            not self._session_ready
            or not self._server_listening
            or not self._model_speaking
            or self._drop_until_listen
            or self._playback
            or self._playback_inflight
            or self._resampled_tail.size
            or now < self._audible_until
        ):
            return False
        self._model_speaking = False
        return self._transition_state_locked("listening")

    def _observe_server_metrics(self, metrics: Mapping[str, Any] | None) -> None:
        if metrics is None:
            return
        raw_kv = metrics.get("kv_cache_length")
        if (
            not isinstance(raw_kv, bool)
            and isinstance(raw_kv, int | float)
            and math.isfinite(float(raw_kv))
            and float(raw_kv) >= 0.0
        ):
            kv_value = float(raw_kv)
            with self._lock:
                self._metrics["last_server_kv_cache_length"] = kv_value
                self._metrics["max_server_kv_cache_length"] = max(
                    kv_value,
                    float(self._metrics.get("max_server_kv_cache_length", 0.0)),
                )
        raw = metrics.get("wall_clock_ms")
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            return
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            return
        with self._lock:
            self._metrics["last_server_wall_clock_ms"] = value
            self._metrics["max_server_wall_clock_ms"] = max(
                value,
                float(self._metrics.get("max_server_wall_clock_ms", 0.0)),
            )
            if value > 1_000.0:
                self._metrics["slow_server_events"] += 1
        if value > 1_000.0:
            log.warning("MiniCPM realtime event took %.1f ms server-side", value)

    def _transition_state_locked(self, state: str) -> bool:
        if state == self._state:
            return False
        self._state = state
        return True

    def _emit_state(self, state: str) -> None:
        if self._state_callback is None:
            return
        with self._lock:
            if self._state != state:
                return
        try:
            self._state_callback(state)
        except Exception as exc:
            self._record_error(exc, "playback_errors", level=logging.WARNING)

    def _record_error(
        self,
        error: BaseException,
        metric: str,
        *,
        level: int = logging.ERROR,
    ) -> None:
        with self._lock:
            self._record_error_locked(error, metric)
        log.log(level, "%s", error, exc_info=level >= logging.ERROR)
        if self._error_callback is not None:
            try:
                self._error_callback(error)
            except Exception:
                log.exception("AudioEngine error callback failed")

    def _record_error_locked(self, error: BaseException, metric: str) -> None:
        self._last_error = error
        self._metrics[metric] = int(self._metrics.get(metric, 0)) + 1

    def _stop_media(self) -> None:
        if self._playing:
            try:
                self.mini.media.stop_playing()
            except Exception as exc:
                self._record_error(exc, "playback_errors", level=logging.WARNING)
            self._playing = False
        if self._recording:
            try:
                self.mini.media.stop_recording()
            except Exception as exc:
                self._record_error(exc, "capture_errors", level=logging.WARNING)
            self._recording = False
