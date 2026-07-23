"""Low-latency capture, near-end detection, and fenced audio playback."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import webrtcvad
from scipy.signal import butter, sosfilt

FloatAudio = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AudioUnit:
    """One exact MiniCPM-o input unit: 16 kHz, one second, mono F32LE."""

    sequence: int
    samples: FloatAudio
    captured_at: float

    @property
    def f32le(self) -> bytes:
        return np.asarray(self.samples, dtype="<f4").tobytes(order="C")


@dataclass(frozen=True, slots=True)
class PlaybackPacket:
    """One server audio delta guarded by its interaction epoch."""

    epoch: int
    samples: FloatAudio
    response_id: str | None = None
    stream_generation: int = 0


@dataclass(frozen=True, slots=True)
class VoiceDecision:
    timestamp: float
    rms: float
    noise_floor: float
    energy_threshold: float
    vad_speech: bool
    current_near_end: bool
    near_end: bool
    output_active: bool
    echo_guard_active: bool
    echo_similarity: float
    echo_like: bool
    barge_in: bool


@dataclass(frozen=True, slots=True)
class PlaybackStats:
    enqueued: int
    dropped: int
    stale: int
    pushed: int
    clears: int
    errors: int


class SpeechDetector(Protocol):
    def is_speech(self, frame: FloatAudio, sample_rate: int) -> bool: ...


class CapturePort(Protocol):
    def get_audio_sample(self) -> npt.NDArray[np.generic] | None: ...


class PlaybackPort(Protocol):
    audio: object

    def push_audio_sample(self, data: FloatAudio) -> None: ...


def mono_capture(samples: npt.ArrayLike, channel: int = 0) -> FloatAudio:
    """Normalize mono, channels-last, channels-first, or higher-rank capture."""

    data = np.asarray(samples)
    if data.ndim == 0:
        raise ValueError("audio capture must have at least one dimension")
    if data.size == 0:
        return np.empty(0, dtype=np.float32)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        mono = data
    else:
        if data.ndim == 2 and data.shape[0] <= 8 and data.shape[1] > 8:
            data = data.T
        elif data.ndim > 2:
            sample_axis = int(np.argmax(data.shape))
            data = np.moveaxis(data, sample_axis, 0)
        channels = data.reshape(data.shape[0], -1)
        if channels.shape[1] == 1:
            mono = channels[:, 0]
        elif channel == -1:
            mono = channels.mean(axis=1, dtype=np.float32)
        elif 0 <= channel < channels.shape[1]:
            mono = channels[:, channel]
        else:
            raise ValueError(f"capture has {channels.shape[1]} channels; cannot select {channel}")

    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)


def _mono_float32(samples: npt.ArrayLike) -> FloatAudio:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("expected one-dimensional mono audio")
    return np.ascontiguousarray(data)


class StreamingResampler:
    """Stateful mono downsampler without per-delta filter resets."""

    def __init__(self, input_rate: int, output_rate: int) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("resampler rates must be positive")
        self.input_rate = input_rate
        self.output_rate = output_rate
        cutoff = min(0.95, 0.9 * output_rate / input_rate)
        self._sos = butter(6, cutoff, output="sos") if output_rate < input_rate else None
        self._filter_state = (
            np.zeros((self._sos.shape[0], 2), dtype=np.float64) if self._sos is not None else None
        )
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._next_numerator = 0

    def reset(self) -> None:
        if self._filter_state is not None:
            self._filter_state.fill(0.0)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._next_numerator = 0

    def process(self, samples: npt.ArrayLike) -> FloatAudio:
        data = _mono_float32(samples)
        if data.size == 0:
            return data
        if self.input_rate == self.output_rate:
            return data.copy()
        if self._sos is not None:
            filtered, self._filter_state = sosfilt(
                self._sos,
                data,
                zi=self._filter_state,
            )
            data = np.asarray(filtered, dtype=np.float32)

        self._buffer = data if self._buffer.size == 0 else np.concatenate((self._buffer, data))
        end_index = self._buffer_start + self._buffer.size - 1
        output: list[float] = []
        while True:
            low, remainder = divmod(self._next_numerator, self.output_rate)
            high = low if remainder == 0 else low + 1
            if high > end_index:
                break
            offset = low - self._buffer_start
            if remainder == 0:
                value = self._buffer[offset]
            else:
                fraction = remainder / self.output_rate
                value = (
                    self._buffer[offset] * (1.0 - fraction) + self._buffer[offset + 1] * fraction
                )
            output.append(float(value))
            self._next_numerator += self.input_rate

        next_low = self._next_numerator // self.output_rate
        discard = min(
            self._buffer.size,
            max(0, next_low - self._buffer_start),
        )
        if discard:
            self._buffer = np.ascontiguousarray(
                self._buffer[discard:],
                dtype=np.float32,
            )
            self._buffer_start += discard
        return np.asarray(output, dtype=np.float32)


class FrameSplitter:
    """Split an arbitrary stream into exact WebRTC-VAD frames."""

    def __init__(self, sample_rate: int = 16_000, frame_ms: int = 20) -> None:
        if frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD supports only 10, 20, or 30 ms frames")
        self.frame_samples = sample_rate * frame_ms // 1_000
        self._pending = np.empty(0, dtype=np.float32)

    @property
    def pending_samples(self) -> int:
        return int(self._pending.size)

    def push(self, samples: npt.ArrayLike) -> list[FloatAudio]:
        data = _mono_float32(samples)
        if data.size == 0:
            return []
        joined = data if self._pending.size == 0 else np.concatenate((self._pending, data))
        count = joined.size // self.frame_samples
        frames = [
            np.ascontiguousarray(joined[index : index + self.frame_samples], dtype=np.float32)
            for index in range(0, count * self.frame_samples, self.frame_samples)
        ]
        self._pending = np.ascontiguousarray(joined[count * self.frame_samples :], dtype=np.float32)
        return frames

    def reset(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)


class AudioUnitizer:
    """Packetize capture into monotonically numbered one-second units."""

    def __init__(self, sample_rate: int = 16_000, unit_ms: int = 1_000) -> None:
        self.unit_samples = sample_rate * unit_ms // 1_000
        if self.unit_samples <= 0:
            raise ValueError("audio unit must contain samples")
        self._pending = np.empty(0, dtype=np.float32)
        self._sequence = 0

    @property
    def pending_samples(self) -> int:
        return int(self._pending.size)

    def push(
        self,
        samples: npt.ArrayLike,
        *,
        captured_at: float | None = None,
    ) -> list[AudioUnit]:
        data = _mono_float32(samples)
        if data.size == 0:
            return []
        joined = data if self._pending.size == 0 else np.concatenate((self._pending, data))
        now = time.monotonic() if captured_at is None else captured_at
        count = joined.size // self.unit_samples
        units: list[AudioUnit] = []
        for index in range(count):
            start = index * self.unit_samples
            unit = np.ascontiguousarray(joined[start : start + self.unit_samples], dtype="<f4")
            units.append(AudioUnit(self._sequence, unit, now))
            self._sequence += 1
        self._pending = np.ascontiguousarray(joined[count * self.unit_samples :], dtype=np.float32)
        return units

    def reset(self, *, reset_sequence: bool = False) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        if reset_sequence:
            self._sequence = 0


class WebRtcSpeechDetector:
    """Float32 adapter around WebRTC VAD's signed 16-bit PCM interface."""

    def __init__(self, mode: int = 2) -> None:
        if mode not in range(4):
            raise ValueError("WebRTC VAD mode must be 0..3")
        self._vad = webrtcvad.Vad(mode)

    def is_speech(self, frame: FloatAudio, sample_rate: int) -> bool:
        pcm = np.rint(np.clip(frame, -1.0, 1.0) * 32_767.0).astype("<i2")
        return bool(self._vad.is_speech(pcm.tobytes(order="C"), sample_rate))


class EchoReference:
    """Bounded reference of audio successfully handed to the local speaker."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        history_seconds: float = 1.5,
        downsample: int = 4,
    ) -> None:
        if history_seconds <= 0 or downsample <= 0:
            raise ValueError("echo history and downsample must be positive")
        self._max_samples = max(1, round(sample_rate * history_seconds / downsample))
        self._downsample = downsample
        self._samples = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()

    def append_played(self, samples: npt.ArrayLike) -> None:
        data = _mono_float32(samples)
        usable = data.size // self._downsample * self._downsample
        if usable == 0:
            return
        reduced = data[:usable].reshape(-1, self._downsample).mean(axis=1, dtype=np.float32)
        with self._lock:
            combined = (
                reduced if self._samples.size == 0 else np.concatenate((self._samples, reduced))
            )
            self._samples = np.ascontiguousarray(combined[-self._max_samples :], dtype=np.float32)

    def similarity(self, captured_frame: npt.ArrayLike) -> float:
        frame = _mono_float32(captured_frame)
        usable = frame.size // self._downsample * self._downsample
        if usable == 0:
            return 0.0
        probe = frame[:usable].reshape(-1, self._downsample).mean(axis=1, dtype=np.float32)
        probe = probe - float(probe.mean())
        probe_norm = float(np.linalg.norm(probe))
        if probe_norm < 1e-8:
            return 0.0
        with self._lock:
            reference = self._samples.copy()
        if reference.size < probe.size:
            return 0.0
        reference = reference - float(reference.mean())
        dots = np.correlate(reference, probe, mode="valid")
        squared = np.square(reference, dtype=np.float32)
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(squared, dtype=np.float64))
        )
        window_energy = cumulative[probe.size :] - cumulative[: -probe.size]
        denominator = np.sqrt(np.maximum(window_energy, 1e-16)) * probe_norm
        correlations = np.abs(dots) / denominator
        return float(np.clip(np.max(correlations, initial=0.0), 0.0, 1.0))

    def clear(self) -> None:
        with self._lock:
            self._samples = np.empty(0, dtype=np.float32)


class NearEndDetector:
    """Combine WebRTC VAD, an adaptive noise gate, and echo correlation."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        vad_mode: int = 2,
        min_rms: float = 0.006,
        noise_ratio: float = 2.2,
        barge_attack_ms: int = 100,
        barge_debounce_ms: int = 350,
        near_end_hold_ms: int = 300,
        echo_correlation: float = 0.72,
        echo_reference: EchoReference | None = None,
        vad: SpeechDetector | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if frame_ms not in {10, 20, 30}:
            raise ValueError("frame_ms must be 10, 20, or 30")
        if min_rms <= 0 or noise_ratio <= 1:
            raise ValueError("noise gate parameters are invalid")
        if not 0 < echo_correlation < 1:
            raise ValueError("echo correlation must be between zero and one")
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1_000
        self._vad = vad or WebRtcSpeechDetector(vad_mode)
        self._echo = echo_reference
        self._clock = clock
        self._min_rms = min_rms
        self._noise_ratio = noise_ratio
        self._echo_correlation = echo_correlation
        self._attack_frames = max(1, math.ceil(barge_attack_ms / frame_ms))
        self._near_frames = max(1, math.ceil(40 / frame_ms))
        self._debounce = barge_debounce_ms / 1_000
        self._hold = near_end_hold_ms / 1_000
        self._noise_floor = max(1e-5, min_rms / noise_ratio)
        self._candidate_frames = 0
        self._last_near = -math.inf
        self._last_barge = -math.inf
        self._barge_latched = False

    def process(
        self,
        frame: npt.ArrayLike,
        *,
        output_active: bool,
        echo_guard_active: bool | None = None,
        timestamp: float | None = None,
    ) -> VoiceDecision:
        data = _mono_float32(frame)
        if data.size != self.frame_samples:
            raise ValueError(f"expected {self.frame_samples} samples, received {data.size}")
        now = self._clock() if timestamp is None else timestamp
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
        threshold = max(self._min_rms, self._noise_floor * self._noise_ratio)
        vad_speech = self._vad.is_speech(data, self.sample_rate)
        guard_active = output_active if echo_guard_active is None else echo_guard_active
        similarity = self._echo.similarity(data) if guard_active and self._echo is not None else 0.0
        echo_like = guard_active and similarity >= self._echo_correlation
        candidate = vad_speech and rms >= threshold and not echo_like

        if candidate:
            self._candidate_frames += 1
        else:
            self._candidate_frames = 0
            self._barge_latched = False

        if self._candidate_frames >= self._near_frames:
            self._last_near = now
        near_end = now - self._last_near <= self._hold

        barge_in = False
        if (
            output_active
            and self._candidate_frames >= self._attack_frames
            and not self._barge_latched
            and now - self._last_barge >= self._debounce
        ):
            barge_in = True
            self._barge_latched = True
            self._last_barge = now

        if not vad_speech and not echo_like:
            alpha = 0.08 if rms < self._noise_floor else 0.02
            self._noise_floor += alpha * (max(rms, 1e-5) - self._noise_floor)

        return VoiceDecision(
            timestamp=now,
            rms=rms,
            noise_floor=self._noise_floor,
            energy_threshold=threshold,
            vad_speech=vad_speech,
            current_near_end=candidate,
            near_end=near_end,
            output_active=output_active,
            echo_guard_active=guard_active,
            echo_similarity=similarity,
            echo_like=echo_like,
            barge_in=barge_in,
        )

    def reset(self) -> None:
        self._candidate_frames = 0
        self._last_near = -math.inf
        self._last_barge = -math.inf
        self._barge_latched = False


class PlaybackEngine:
    """One playback writer with bounded latency and an epoch race barrier."""

    def __init__(
        self,
        media: PlaybackPort,
        epoch_supplier: Callable[[], int],
        echo_reference: EchoReference,
        *,
        input_sample_rate: int = 24_000,
        output_sample_rate: int = 16_000,
        max_queue: int = 3,
        preroll_ms: int = 60,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        if input_sample_rate <= 0 or output_sample_rate <= 0:
            raise ValueError("playback sample rates must be positive")
        if max_queue <= 0 or preroll_ms < 0:
            raise ValueError("playback queue parameters are invalid")
        self._media = media
        self._epoch_supplier = epoch_supplier
        self._echo = echo_reference
        self._input_rate = input_sample_rate
        self._output_rate = output_sample_rate
        self._resampler = StreamingResampler(
            input_sample_rate,
            output_sample_rate,
        )
        self._resampler_key: tuple[int, int, str | None] | None = None
        self._stream_generation = 0
        self._max_queue = max_queue
        self._preroll_samples = input_sample_rate * preroll_ms // 1_000
        self._preroll_seconds = preroll_ms / 1_000
        self._clock = clock
        self._logger = logger or logging.getLogger(__name__)

        self._condition = threading.Condition()
        self._output_lock = threading.Lock()
        self._activity_lock = threading.Lock()
        self._queue: deque[PlaybackPacket] = deque()
        self._first_seen: dict[int, float] = {}
        self._primed_epochs: set[int] = set()
        self._fence_epoch = 0
        self._stopping = False
        self._started = False
        self._thread: threading.Thread | None = None
        self._playing_until = 0.0
        self._echo_guard_until = 0.0
        self._echo_tail_seconds = 0.18
        self._enqueued = 0
        self._dropped = 0
        self._stale = 0
        self._pushed = 0
        self._clears = 0
        self._errors = 0

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            if self._stopping:
                raise RuntimeError("playback engine cannot be restarted")
            player = getattr(self._media, "audio", None)
            if player is None or not hasattr(player, "clear_player"):
                raise RuntimeError("the local media audio backend is unavailable")
            player.set_max_output_buffers(self._max_queue)
            self._started = True
            self._thread = threading.Thread(target=self._run, name="yrobot-playback", daemon=True)
            self._thread.start()

    def enqueue(self, packet: PlaybackPacket) -> bool:
        samples = np.asarray(packet.samples, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            return False
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        normalized_samples = np.ascontiguousarray(np.clip(samples, -1.0, 1.0))
        with self._condition:
            normalized = PlaybackPacket(
                epoch=packet.epoch,
                samples=normalized_samples,
                response_id=packet.response_id,
                stream_generation=self._stream_generation,
            )
            if self._stopping or not self._is_current(normalized):
                self._stale += 1
                return False
            while len(self._queue) >= self._max_queue:
                self._queue.popleft()
                self._dropped += 1
            self._queue.append(normalized)
            self._first_seen.setdefault(packet.epoch, self._clock())
            self._enqueued += 1
            self._condition.notify()
        return True

    def mark_response_boundary(self) -> None:
        """Separate resampler state for audio received after a listen boundary."""

        with self._condition:
            self._stream_generation += 1

    def interrupt(self, epoch: int) -> bool:
        """Fence first, then serialize a hardware flush against every push."""

        with self._condition:
            self._fence_epoch = max(self._fence_epoch, epoch)
            self._dropped += len(self._queue)
            self._queue.clear()
            self._first_seen.clear()
            self._primed_epochs.clear()
            self._condition.notify_all()
        with self._output_lock:
            with self._activity_lock:
                was_playing = self._clock() < self._playing_until
            try:
                self._player().clear_player()
            except Exception:
                self._record_error()
                self._logger.exception("failed to flush Reachy Mini playback")
                cleared = False
            else:
                with self._condition:
                    self._clears += 1
                cleared = True
            with self._activity_lock:
                now = self._clock()
                self._playing_until = now
                if was_playing:
                    self._echo_guard_until = max(
                        self._echo_guard_until,
                        now + self._echo_tail_seconds,
                    )
        return cleared

    def output_active(self) -> bool:
        with self._activity_lock:
            return self._clock() < self._playing_until

    def echo_guard_active(self) -> bool:
        """Include a short room/speaker tail after predicted playback ends."""

        with self._activity_lock:
            return self._clock() < max(
                self._playing_until,
                self._echo_guard_until,
            )

    def stats(self) -> PlaybackStats:
        with self._condition:
            return PlaybackStats(
                enqueued=self._enqueued,
                dropped=self._dropped,
                stale=self._stale,
                pushed=self._pushed,
                clears=self._clears,
                errors=self._errors,
            )

    def stop(self, *, flush: bool = True, timeout: float = 2.0) -> bool:
        with self._condition:
            if self._stopping:
                thread = self._thread
            else:
                self._stopping = True
                self._dropped += len(self._queue)
                self._queue.clear()
                self._condition.notify_all()
                thread = self._thread
        if flush and self._started:
            with self._output_lock:
                try:
                    self._player().clear_player()
                except Exception:
                    self._record_error()
                    self._logger.exception("failed to flush playback during shutdown")
                else:
                    with self._condition:
                        self._clears += 1
                self._echo.clear()
                with self._activity_lock:
                    self._playing_until = self._clock()
        if thread is not None:
            thread.join(timeout)
            stopped = not thread.is_alive()
        else:
            stopped = True
        self._echo.clear()
        with self._activity_lock:
            now = self._clock()
            self._playing_until = now
            self._echo_guard_until = now
        return stopped

    def _run(self) -> None:
        while True:
            packet = self._take_packet()
            if packet is None:
                return
            if self._input_rate == self._output_rate:
                output = packet.samples
            else:
                resampler_key = (
                    packet.epoch,
                    packet.stream_generation,
                    packet.response_id,
                )
                if resampler_key != self._resampler_key:
                    self._resampler.reset()
                    self._resampler_key = resampler_key
                output = self._resampler.process(packet.samples)
            if output.size == 0:
                continue
            output = np.ascontiguousarray(output, dtype=np.float32)
            with self._output_lock:
                with self._condition:
                    valid = not self._stopping and self._is_current(packet)
                    if not valid:
                        self._stale += 1
                if not valid:
                    continue
                try:
                    self._media.push_audio_sample(output)
                except Exception:
                    self._record_error()
                    self._logger.exception("failed to push Reachy Mini audio")
                    continue
                self._echo.append_played(output)
                with self._activity_lock:
                    now = self._clock()
                    self._playing_until = max(now, self._playing_until) + (
                        output.size / self._output_rate
                    )
                    self._echo_guard_until = self._playing_until + self._echo_tail_seconds
                with self._condition:
                    self._pushed += 1

    def _take_packet(self) -> PlaybackPacket | None:
        with self._condition:
            while True:
                if self._stopping:
                    return None
                while self._queue and not self._is_current(self._queue[0]):
                    self._queue.popleft()
                    self._stale += 1
                if not self._queue:
                    self._condition.wait()
                    continue
                packet = self._queue[0]
                if self._preroll_samples and packet.epoch not in self._primed_epochs:
                    available = sum(
                        queued.samples.size
                        for queued in self._queue
                        if queued.epoch == packet.epoch
                    )
                    elapsed = self._clock() - self._first_seen[packet.epoch]
                    remaining = self._preroll_seconds - elapsed
                    if available < self._preroll_samples and remaining > 0:
                        self._condition.wait(min(remaining, 0.01))
                        continue
                    self._primed_epochs.add(packet.epoch)
                return self._queue.popleft()

    def _is_current(self, packet: PlaybackPacket) -> bool:
        return packet.epoch >= self._fence_epoch and packet.epoch == self._epoch_supplier()

    def _player(self) -> object:
        player = getattr(self._media, "audio", None)
        if player is None:
            raise RuntimeError("the local media audio backend is unavailable")
        return player

    def _record_error(self) -> None:
        with self._condition:
            self._errors += 1


class AudioCaptureWorker:
    """Own the non-blocking microphone pull loop, but never media lifecycle."""

    def __init__(
        self,
        media: CapturePort,
        *,
        channel: int,
        detector: NearEndDetector,
        output_active: Callable[[], bool],
        echo_guard_active: Callable[[], bool] | None = None,
        on_unit: Callable[[AudioUnit], None],
        on_voice: Callable[[VoiceDecision], None] | None = None,
        on_barge_in: Callable[[VoiceDecision], None] | None = None,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        unit_ms: int = 1_000,
        logger: logging.Logger | None = None,
    ) -> None:
        self._media = media
        self._channel = channel
        self._detector = detector
        self._output_active = output_active
        self._echo_guard_active = echo_guard_active or output_active
        self._on_unit = on_unit
        self._on_voice = on_voice
        self._on_barge_in = on_barge_in
        self._frames = FrameSplitter(sample_rate, frame_ms)
        self._units = AudioUnitizer(sample_rate, unit_ms)
        self._logger = logger or logging.getLogger(__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="yrobot-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                captured = self._media.get_audio_sample()
            except Exception:
                self._logger.exception("failed to read Reachy Mini microphone")
                self._stop.wait(0.02)
                continue
            if captured is None:
                self._stop.wait(0.002)
                continue
            try:
                mono = mono_capture(captured, self._channel)
            except (TypeError, ValueError):
                self._logger.exception("discarding malformed microphone samples")
                continue
            now = time.monotonic()
            # Evaluate barge-in before publishing a unit completed by the same
            # capture block, so its wire event can already carry force_listen.
            for frame in self._frames.push(mono):
                decision = self._detector.process(
                    frame,
                    output_active=self._output_active(),
                    echo_guard_active=self._echo_guard_active(),
                    timestamp=now,
                )
                if self._on_voice is not None:
                    self._call(self._on_voice, decision, "voice-state callback failed")
                if decision.barge_in and self._on_barge_in is not None:
                    self._call(self._on_barge_in, decision, "barge-in callback failed")
            for unit in self._units.push(mono, captured_at=now):
                self._call(self._on_unit, unit, "audio unit callback failed")

    def _call(
        self,
        callback: Callable[[object], None],
        value: object,
        message: str,
    ) -> None:
        try:
            callback(value)
        except Exception:
            self._logger.exception(message)
