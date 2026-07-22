"""Reachy Mini audio, camera, and restrained phase-A motion adapter."""

from __future__ import annotations

import io
import logging
import math
import queue
import random
import threading
import time
from collections.abc import Mapping
from contextlib import suppress

import numpy as np

try:
    from PIL import Image
except ImportError:  # Pillow is optional; frames then upload at native size.
    Image = None
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import (
    compose_world_offset,
    delta_angle_between_mat_rot,
    linear_pose_interpolation,
    time_trajectory,
)
from scipy.signal import firwin, lfilter

from .config import CHUNK_SAMPLES, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE

log = logging.getLogger(__name__)

ANTENNA_POSES: Mapping[str, tuple[float, float]] = {
    "idle": (-11.0, 11.0),
    "listening": (-16.0, 16.0),
    "speaking": (-20.0, 20.0),
}

# Reachy conversation-app tuning plus the XVF robust double-talk mode needed
# to preserve near-end speech while the built-in speaker is active.
AUDIO_STARTUP_CONFIG = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
    # Lowest robust double-talk mode: preserves near-end speech while the
    # speaker is active and enables XVF's extra near-end speech detector.
    ("PP_DTSENSITIVE", (10,)),
)
AUDIO_CONFIG_SETTLE_SECONDS = 0.1

# The official conversation app uses a 60 Hz single-owner motion loop. 50 Hz
# keeps the same smooth-control regime while leaving a little more CM4 margin.
MOTION_PERIOD = 0.02
MOTION_CONNECTION_GRACE = 2.5
MOTION_ERROR_LOG_INTERVAL = 5.0
DOA_PERIOD = 0.05
CAMERA_PERIOD = 1.0
# The Omni server never pads the uplink: each one-second input chunk drives
# exactly one prefill/decode step, so a stalled microphone freezes the model's
# slice clock (and its sense of elapsed time) until hardware audio returns.
# After one chunk length without samples the capture loop substitutes silent
# chunks to keep the session advancing in real time.
CAPTURE_STALL_TIMEOUT = CHUNK_SAMPLES / INPUT_SAMPLE_RATE
# MiniCPM's vision front-end rescales every frame to 448 px on its long side
# (max_slice_nums=1), so a camera-native JPEG only spends Wi-Fi bandwidth and
# send time on pixels the server immediately throws away. Frames are shrunk on
# the CM4 before upload; quality 70 matches the official web demo client.
VIDEO_MAX_DIMENSION = 448
VIDEO_JPEG_QUALITY = 70
# Reachy's GStreamer player reanchors its clock after a 200 ms input gap, and
# TTS arrives as one-second time slices whose boundaries carry real jitter.
# An utterance therefore starts behind an adaptive jitter buffer: an observed
# TTS supply gap raises it to that gap plus a margin, and every started
# utterance decays it toward the floor. The buffer applies only at an
# utterance start — a mid-utterance starvation resumes immediately, because
# waiting would only stretch the audible hole.
PLAYBACK_PREROLL_SECONDS = 0.3
PLAYBACK_PREROLL_MIN = 0.2
PLAYBACK_PREROLL_MAX = 0.5
PLAYBACK_PREROLL_MARGIN = 0.1
PLAYBACK_PREROLL_DECAY = 0.02
# Uplink AGC: hardware testing with MiniCPM-o showed the model treats quiet
# near-end speech as background and never answers, so uploaded slices are
# lifted toward a fixed speech level. Adaptation freezes while the robot is
# speaking so residual echo cannot pull the estimate down, and the gain never
# attenuates.
UPLINK_AGC_TARGET_RMS = 0.12
UPLINK_AGC_MAX_GAIN = 8.0
UPLINK_AGC_SPEECH_FLOOR_RMS = 0.006
UPLINK_AGC_SMOOTHING = 0.3
# The arm delay is dead time in which the user cannot barge in; the XVF AEC
# filters stay converged between utterances, so 0.4 s of residual-floor
# learning suffices. Confirmation is a sustained-duration test on the 20 ms
# motion tick. The XVF "speech" flag is PRE-AEC — it stays high while the
# robot's own speaker plays — so the post-AEC level test is the only gate.
INTERRUPT_ARM_DELAY = 0.4
INTERRUPT_CONFIRM_SECONDS = 0.12
INTERRUPT_COOLDOWN = 1.0
INTERRUPT_MIN_LEVEL_DB = -38.0
INTERRUPT_MIN_RISE_DB = 6.0
INTERRUPT_FLOOR_RISE_DB_PER_SAMPLE = 0.15
INTERRUPT_ACTIVITY_HOLD = 1.25
INTERRUPT_POST_LISTEN_GUARD = 0.4
INTERRUPT_ACK_TIMEOUT = 3.0
# After a forced listen the model often starts its next utterance one time
# slice later, talking straight over the still-speaking user (hardware logs
# 2026-07-21: every ack was followed ~1 s later by a resumed monologue and the
# user had to shout it down again). Playback therefore stays muted until the
# first listen boundary after the user has been quiet for the yield hold, and
# model audio that arrives while the user is still speaking re-forces listen
# directly instead of waiting for the energy detector's arm/confirm cycle.
# The cap bounds the mute in case something keeps the speech stamps fresh.
INTERRUPT_YIELD_HOLD = 0.7
INTERRUPT_REFORCE_INTERVAL = 1.0
INTERRUPT_YIELD_MAX = 12.0
MAX_HEAD_ANGULAR_STEP = math.radians(2.0)
MAX_HEAD_TRANSLATION_STEP = 0.003
MAX_HEAD_ANGULAR_SPEED = math.radians(45.0)
MAX_HEAD_TRANSLATION_SPEED = 0.04
MAX_ANTENNA_SPEED = math.radians(100.0)
HEAD_SERVO_TIME_CONSTANT = 0.22
USER_SPEECH_HOLD = 0.25
DOA_TURN_THRESHOLD = math.radians(4.0)
DOA_TURN_COOLDOWN = 0.35
LISTENING_NOD_DURATION = 0.72
STATE_TRANSITION_DURATION = 0.9
NATURAL_HEAD_PITCH_DEGREES = -4.0
DOA_GAZE_ELEVATION = math.tan(math.radians(-NATURAL_HEAD_PITCH_DEGREES))


def to_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2 and audio.shape[1] >= 1:
        # Reachy/XVF exposes two channels, but the official conversation app
        # forwards channel 0. Averaging can mix a non-target channel back into
        # the AEC-processed ASR signal.
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return audio[:, 0]
    raise ValueError(f"unexpected microphone shape: {audio.shape}")


def audio_level_db(samples: np.ndarray) -> float:
    """Return dBFS RMS for one post-AEC microphone frame."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return -120.0
    rms = math.sqrt(float(np.dot(audio, audio)) / audio.size)
    return 20.0 * math.log10(max(rms, 1e-6))


def downscale_jpeg(
    frame: bytes,
    max_dimension: int = VIDEO_MAX_DIMENSION,
    quality: int = VIDEO_JPEG_QUALITY,
) -> bytes:
    """Return the JPEG shrunk so its long side fits ``max_dimension`` pixels.

    Pillow's draft mode performs most of the reduction inside the JPEG decoder,
    keeping the per-frame cost small on the CM4. Any failure — Pillow missing,
    an unparseable frame — returns the original bytes unchanged.
    """
    if Image is None:
        return frame
    try:
        with Image.open(io.BytesIO(frame)) as image:
            width, height = image.size
            longest = max(width, height)
            if longest <= max_dimension:
                return frame
            scale = max_dimension / longest
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
            image.draft("RGB", target)
            resized = image.convert("RGB").resize(target, Image.BILINEAR)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
    except Exception as exc:
        log.debug("Frame downscale failed; sending the original frame: %s", exc)
        return frame


class UplinkAGC:
    """Lift quiet near-end speech toward the level MiniCPM answers reliably.

    The estimate tracks a smoothed speech RMS and is only trusted below the
    target, so the gain is always >= 1 and loud speech passes unchanged.
    """

    def __init__(self) -> None:
        self._speech_rms: float | None = None

    def process(self, chunk: np.ndarray, *, adapt: bool = True) -> np.ndarray:
        if chunk.size == 0:
            return chunk
        rms = math.sqrt(float(np.dot(chunk, chunk)) / chunk.size)
        if adapt and rms > UPLINK_AGC_SPEECH_FLOOR_RMS:
            self._speech_rms = (
                rms
                if self._speech_rms is None
                else (1.0 - UPLINK_AGC_SMOOTHING) * self._speech_rms
                + UPLINK_AGC_SMOOTHING * rms
            )
        if self._speech_rms is None or self._speech_rms >= UPLINK_AGC_TARGET_RMS:
            return chunk
        gain = min(UPLINK_AGC_MAX_GAIN, UPLINK_AGC_TARGET_RMS / self._speech_rms)
        return np.clip(chunk * gain, -1.0, 1.0)


class StreamingAudioResampler:
    """Causal rational resampler whose filter and phase survive TTS deltas.

    ``resample_poly`` is excellent for a complete buffer, but it pads every
    independent call with zeros.  Calling it once per streamed TTS delta makes
    those artificial edges audible.  This implementation keeps the FIR delay
    line and decimation phase across every streamed TTS delta.  RobotIO only
    resets it after playback has genuinely drained or an interruption
    invalidates the playback generation.
    """

    def __init__(self, input_rate: int = OUTPUT_SAMPLE_RATE, output_rate: int = INPUT_SAMPLE_RATE):
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("audio sample rates must be positive")
        divisor = math.gcd(input_rate, output_rate)
        self._up = output_rate // divisor
        self._down = input_rate // divisor
        self._passthrough = self._up == self._down
        self._phase = 0

        if self._passthrough:
            self._taps = np.ones(1, dtype=np.float32)
        else:
            max_rate = max(self._up, self._down)
            half_length = 10 * max_rate
            self._taps = (
                firwin(
                    2 * half_length + 1,
                    1.0 / max_rate,
                    window=("kaiser", 5.0),
                )
                * self._up
            ).astype(np.float32)
        self._denominator = np.ones(1, dtype=np.float32)
        self._filter_state = np.zeros(self._taps.size - 1, dtype=np.float32)

    def reset(self) -> None:
        self._phase = 0
        self._filter_state.fill(0.0)

    def process(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("playback audio must be mono")
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)
        if self._passthrough:
            return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=True)

        upsampled = np.zeros(audio.size * self._up, dtype=np.float32)
        upsampled[:: self._up] = audio
        filtered, self._filter_state = lfilter(
            self._taps,
            self._denominator,
            upsampled,
            zi=self._filter_state,
        )
        converted = filtered[self._phase :: self._down]
        self._phase = (self._phase - filtered.size) % self._down
        return np.clip(converted, -1.0, 1.0).astype(np.float32, copy=False)


def doa_world_direction(angle: float, head_pose: np.ndarray) -> np.ndarray:
    """Map head-relative DoA to a stable, slightly raised world gaze."""
    pose = np.asarray(head_pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("head_pose must be a 4x4 transform")

    # DoA contains azimuth but no reliable elevation. Reusing the head's full
    # rotation makes any temporary downward pitch feed into the next target and
    # causes a self-reinforcing "always looking down" drift. Preserve only yaw
    # and use a small fixed upward gaze for natural eye contact.
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    direction_head = np.array([math.sin(angle), math.cos(angle)])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    direction_world = np.array(
        [
            cos_yaw * direction_head[0] - sin_yaw * direction_head[1],
            sin_yaw * direction_head[0] + cos_yaw * direction_head[1],
            DOA_GAZE_ELEVATION,
        ]
    )
    return direction_world


def angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def step_pose(
    current: np.ndarray,
    target: np.ndarray,
    max_angular_step: float = MAX_HEAD_ANGULAR_STEP,
    max_translation_step: float = MAX_HEAD_TRANSLATION_STEP,
) -> np.ndarray:
    """Take one bounded interpolation step between two head poses."""
    current_pose = np.asarray(current, dtype=np.float64)
    target_pose = np.asarray(target, dtype=np.float64)
    if current_pose.shape != (4, 4) or target_pose.shape != (4, 4):
        raise ValueError("head poses must be 4x4 transforms")

    angular_delta = delta_angle_between_mat_rot(current_pose[:3, :3], target_pose[:3, :3])
    translation_delta = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
    fraction = 1.0
    if angular_delta > 0:
        fraction = min(fraction, max_angular_step / angular_delta)
    if translation_delta > 0:
        fraction = min(fraction, max_translation_step / translation_delta)
    return linear_pose_interpolation(current_pose, target_pose, fraction)


def smooth_pose_step(
    current: np.ndarray,
    target: np.ndarray,
    elapsed: float,
    *,
    time_constant: float = HEAD_SERVO_TIME_CONSTANT,
    max_angular_speed: float = MAX_HEAD_ANGULAR_SPEED,
    max_translation_speed: float = MAX_HEAD_TRANSLATION_SPEED,
) -> np.ndarray:
    """Ease a reactive pose target while keeping speed independent of loop rate."""
    if elapsed <= 0:
        return np.asarray(current, dtype=np.float64).copy()
    if time_constant <= 0:
        raise ValueError("time_constant must be positive")
    eased_fraction = 1.0 - math.exp(-elapsed / time_constant)
    eased_target = linear_pose_interpolation(current, target, eased_fraction)
    return step_pose(
        current,
        eased_target,
        max_angular_step=max_angular_speed * elapsed,
        max_translation_step=max_translation_speed * elapsed,
    )


def gesture_pulse(progress: float) -> float:
    """Return a minimum-jerk 0→1→0 pulse for a normalized gesture."""
    if progress < 0.0 or progress > 1.0:
        return 0.0
    leg = progress * 2.0 if progress <= 0.5 else (1.0 - progress) * 2.0
    return time_trajectory(leg)


def effective_conversation_state(model_state: str, speaking: bool) -> str:
    """Use real playback as the source of truth for the speaking posture.

    Token2Wav may emit a final audio delta after ``response.done``. That late
    delta sets the model state to speaking without a following event to clear
    it, so an expired playback deadline must settle back to listening.
    """
    if speaking:
        return "speaking"
    return "listening" if model_state == "speaking" else model_state


def lifelike_motion_overlay(
    elapsed: float,
    state: str,
    *,
    user_speaking: bool,
    transition_pulse: float = 0.0,
    nod_pulse: float = 0.0,
    glance_pulse: float = 0.0,
    glance_yaw_degrees: float = 0.0,
    glance_pitch_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a restrained additive head pose and non-repeating antenna pose.

    The head values stay deliberately below the SDK's audio-reactive wobble;
    this layer supplies continuity and conversational intent while the SDK
    remains responsible for playback-synchronised speech motion.
    """
    if state not in ANTENNA_POSES:
        raise ValueError(f"unknown conversation state: {state}")
    t = max(0.0, elapsed)

    # Slow mixed-frequency motion avoids a perfectly periodic "screensaver"
    # look. Speech uses less base motion because the SDK wobbler is additive.
    z_amplitude = {"idle": 0.0018, "listening": 0.0011, "speaking": 0.0005}[state]
    roll_amplitude = {"idle": 0.45, "listening": 0.28, "speaking": 0.15}[state]
    pitch_amplitude = {"idle": 0.28, "listening": 0.18, "speaking": 0.10}[state]
    z = z_amplitude * (
        0.72 * math.sin(2.0 * math.pi * 0.11 * t) + 0.28 * math.sin(2.0 * math.pi * 0.073 * t + 1.1)
    )
    roll = math.radians(roll_amplitude) * math.sin(2.0 * math.pi * 0.083 * t + 0.7)
    pitch = math.radians(pitch_amplitude) * math.sin(2.0 * math.pi * 0.13 * t + 1.9)
    yaw = 0.0

    if state == "speaking":
        pitch -= math.radians(1.1) * transition_pulse
        z += 0.0007 * transition_pulse
    pitch += math.radians(1.8) * nod_pulse
    yaw += math.radians(glance_yaw_degrees) * glance_pulse
    pitch += math.radians(glance_pitch_degrees) * glance_pulse

    head_offset = create_head_pose(
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        degrees=False,
    )

    antennas = np.deg2rad(np.asarray(ANTENNA_POSES[state], dtype=np.float64))
    if not user_speaking:
        # The antenna motors sit next to the microphones and speech is the only
        # window where the local double-talk detector runs, so speaking sways
        # stay smaller than the idle animation.
        sway_amplitude = {"idle": 2.4, "listening": 1.3, "speaking": 0.9}[state]
        symmetric_sway = math.radians(sway_amplitude) * (
            0.65 * math.sin(2.0 * math.pi * 0.19 * t + 0.2)
            + 0.35 * math.sin(2.0 * math.pi * 0.31 * t + 2.0)
        )
        asymmetry = math.radians(0.55) * math.sin(2.0 * math.pi * 0.127 * t + 1.4)
        antennas = antennas + np.array(
            [-symmetric_sway + asymmetry, symmetric_sway + 0.6 * asymmetry]
        )
    if state == "speaking":
        antennas = antennas + np.deg2rad([-2.0, 2.0]) * transition_pulse
    return head_offset, antennas


class RobotIO:
    """Own Reachy media and serialize every application-level motion command."""

    def __init__(
        self,
        mini: object,
        *,
        capture_video: bool = True,
        playback_preroll: float = PLAYBACK_PREROLL_SECONDS,
    ) -> None:
        if playback_preroll < 0:
            raise ValueError("playback_preroll must not be negative")
        self.mini = mini
        self._capture_video = capture_video
        self._playback_preroll = playback_preroll
        self._audio_chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        # Keep network reception independent from GStreamer.  TTS callbacks can
        # arrive in bursts; pushing them from the WebSocket receive loop makes
        # that loop stop reading while the audio backend catches up.
        self._playback_chunks: queue.Queue[tuple[int, str, np.ndarray, bool]] = queue.Queue()
        self._playback_lock = threading.Lock()
        self._camera_lock = threading.Lock()
        self._latest_frame_jpeg: bytes | None = None
        self._playback_generation = 0
        self._suppress_playback_until = 0.0
        self._discard_turn_active = False
        self._player_clear_pending = False
        self._force_listen_event = threading.Event()
        self._emit_partial_event = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._camera_thread: threading.Thread | None = None
        self._playback_thread: threading.Thread | None = None
        self._motion_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._model_state = "idle"
        self._speaking_until = 0.0
        self._interrupt_armed_at = math.inf
        self._last_near_end_activity_at = -math.inf
        self._last_user_speech_at = -math.inf
        self._force_requested_at = -math.inf
        self._interrupt_hold_started_at = -math.inf
        self._microphone_level_db = -120.0
        self._dropped_audio_chunks = 0
        self._last_audio_drop_log = 0.0
        self._recording = False
        self._playing = False
        self._wobbling = False
        self._motion_rng = random.Random()
        self._command_head = np.eye(4, dtype=np.float64)
        self._target_head = self._command_head.copy()
        self._command_antennas = np.deg2rad(ANTENNA_POSES["idle"])
        self._target_antennas = self._command_antennas.copy()

    def start(self) -> None:
        self.mini.enable_motors()
        self._initialize_motion_state()
        # Start both media directions before configuring the XVF3800 so its
        # far-end reference path is live when the official AEC tuning lands.
        self.mini.media.start_recording()
        self._recording = True
        self.mini.media.start_playing()
        self._playing = True
        self._apply_audio_startup_config()
        self.mini.enable_wobbling()
        self._wobbling = True
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="yrobot-audio", daemon=True
        )
        if self._capture_video and Image is None:
            log.warning("Pillow is unavailable; camera frames upload at native resolution")
        if self._capture_video:
            self._camera_thread = threading.Thread(
                target=self._camera_loop, name="yrobot-camera", daemon=True
            )
        self._playback_thread = threading.Thread(
            target=self._playback_loop, name="yrobot-playback", daemon=True
        )
        self._motion_thread = threading.Thread(
            target=self._motion_loop, name="yrobot-motion", daemon=True
        )
        self._capture_thread.start()
        if self._camera_thread is not None:
            self._camera_thread.start()
        self._playback_thread.start()
        self._motion_thread.start()
        log.info("Reachy media and motion started (XVF AEC + native Omni full duplex)")

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (
            self._capture_thread,
            self._camera_thread,
            self._playback_thread,
            self._motion_thread,
        ):
            if thread is not None:
                thread.join(timeout=2.0)

        if self._wobbling:
            try:
                self.mini.disable_wobbling()
            except Exception as exc:
                log.debug("Could not disable speech wobble: %s", exc)
            self._wobbling = False
        if self._playing:
            self._clear_player()
            try:
                self.mini.media.stop_playing()
            except Exception as exc:
                log.warning("Could not stop speaker playback: %s", exc)
            self._playing = False
        if self._recording:
            try:
                self.mini.media.stop_recording()
            except Exception as exc:
                log.warning("Could not stop microphone capture: %s", exc)
            self._recording = False
        self._return_to_neutral()
        log.info("Reachy media and motion stopped")

    def next_audio_chunk(self, timeout: float) -> np.ndarray | None:
        """Return one real post-XVF microphone slice, including during playback."""
        try:
            return self._audio_chunks.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush_audio_input(self) -> None:
        while True:
            try:
                self._audio_chunks.get_nowait()
            except queue.Empty:
                return

    def get_frame_jpeg(self) -> bytes | None:
        """Return the latest asynchronously encoded frame without blocking audio send."""
        with self._camera_lock:
            return self._latest_frame_jpeg

    def play_omni_audio(self, samples: np.ndarray, response_id: str = "current") -> bool:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("playback audio must be mono")
        if audio.size == 0:
            return False
        # decode_pcm owns its array, but copying here keeps this port safe for
        # callers that reuse a mutable input buffer after returning.
        with self._playback_lock:
            now = time.monotonic()
            if self._discard_turn_active:
                self._maybe_reforce_listen(now)
                return False
            if now < self._suppress_playback_until:
                return False
            # The receive loop marks the model as speaking only after this call
            # and only a listen boundary marks it listening again, so a delta
            # arriving while the model was not speaking starts a new utterance
            # and earns the start-of-utterance preroll.
            with self._state_lock:
                starts_response = self._model_state != "speaking"
            self._playback_chunks.put(
                (self._playback_generation, response_id, audio.copy(), starts_response)
            )
        return True

    def force_listen_active(self) -> bool:
        """Tell MiniCPM to stop speaking after high-confidence double talk."""
        if not self._force_listen_event.is_set():
            return False
        if time.monotonic() - self._force_requested_at > INTERRUPT_ACK_TIMEOUT:
            # Stop replacing real microphone slices with silent control slices
            # so the conversation can continue. The discarded turn stays muted
            # until its own listen/done boundary regardless.
            log.warning("Forced listen was not acknowledged; resuming normal input")
            self._force_listen_event.clear()
            return False
        return True

    def reset_interruption(self) -> None:
        """Drop interruption state that must not leak into a new Omni session."""
        self._force_listen_event.clear()
        self._emit_partial_event.clear()
        with self._playback_lock:
            self._suppress_playback_until = 0.0
            self._discard_turn_active = False

    def note_tts_supply_gap(self, gap_seconds: float) -> None:
        """Grow the playback preroll to cover an observed TTS delivery stall."""
        target = min(PLAYBACK_PREROLL_MAX, gap_seconds + PLAYBACK_PREROLL_MARGIN)
        with self._playback_lock:
            if target > self._playback_preroll:
                self._playback_preroll = target
                log.info("Playback preroll raised to %.0f ms", target * 1_000)

    def handle_omni_listen(self, response_id: str) -> None:
        """Finish a forced interruption once MiniCPM confirms listen mode."""
        now = time.monotonic()
        with self._state_lock:
            forced = self._force_listen_event.is_set()
            recent_near_end = (
                now - self._last_near_end_activity_at <= INTERRUPT_ACTIVITY_HOLD
            )
            user_speaking = (
                now - max(self._last_near_end_activity_at, self._last_user_speech_at)
                <= INTERRUPT_YIELD_HOLD
            )
            active = now < self._speaking_until or not self._playback_chunks.empty()
        self._force_listen_event.clear()
        with self._playback_lock:
            if self._discard_turn_active:
                if (
                    user_speaking
                    and now - self._interrupt_hold_started_at <= INTERRUPT_YIELD_MAX
                ):
                    # The ack arrived while the user is still mid-utterance.
                    # Lifting the mute here lets the model's next utterance
                    # talk straight over them; stay muted until the first
                    # listen after the user has gone quiet.
                    log.info(
                        "Listen ack while user still speaking; holding playback (response=%s)",
                        response_id or "unknown",
                    )
                    with self._state_lock:
                        self._model_state = "listening"
                    return
                self._discard_turn_active = False
                self._suppress_playback_until = now + INTERRUPT_POST_LISTEN_GUARD

        if active and (forced or recent_near_end):
            self._flush_playback_for_listen(now)
            log.info(
                "MiniCPM listen confirmed user interruption (response=%s)",
                response_id or "unknown",
            )
        else:
            with self._state_lock:
                self._model_state = "listening"

        if forced or recent_near_end:
            with self._playback_lock:
                self._suppress_playback_until = now + INTERRUPT_POST_LISTEN_GUARD

    def _maybe_reforce_listen(self, now: float) -> None:
        """Re-chop model speech that talks over a user who is still speaking.

        Runs on the discard path with the playback lock held. The energy
        detector cannot help here: its arm delay plus confirmation window lets
        a full second of talk-over play before it fires again.
        """
        if self._force_listen_event.is_set():
            return
        if now - self._force_requested_at < INTERRUPT_REFORCE_INTERVAL:
            return
        if now - self._interrupt_hold_started_at > INTERRUPT_YIELD_MAX:
            return
        with self._state_lock:
            user_speaking = (
                now - max(self._last_near_end_activity_at, self._last_user_speech_at)
                <= INTERRUPT_YIELD_HOLD
            )
        if not user_speaking:
            return
        self._force_requested_at = now
        self._force_listen_event.set()
        self._emit_partial_event.set()
        log.info("Model talked over the still-speaking user; re-forcing listen")

    def _request_user_interrupt(self, level_db: float, threshold_db: float) -> bool:
        """Flush current speech and hold force_listen until model acknowledgement."""
        now = time.monotonic()
        with self._state_lock:
            active = now < self._speaking_until or not self._playback_chunks.empty()
            self._last_near_end_activity_at = now
        if not active or self._force_listen_event.is_set():
            return False

        self._force_requested_at = now
        self._interrupt_hold_started_at = now
        self._force_listen_event.set()
        # Ship the user's words already sitting in the capture buffer right
        # away so the model hears WHO interrupted it in the same forced slice
        # instead of a second of silence (which made it resume its own story).
        self._emit_partial_event.set()
        self._flush_playback_for_listen(now, suppress_until_listen=True)
        log.info(
            "Post-AEC double talk detected at %.1f dBFS (threshold %.1f); forcing listen",
            level_db,
            threshold_db,
        )
        return True

    def _flush_playback_for_listen(
        self,
        now: float,
        *,
        suppress_until_listen: bool = False,
    ) -> None:
        """Invalidate queued/appsrc audio without racing the playback worker."""
        with self._playback_lock:
            self._playback_generation += 1
            if suppress_until_listen:
                # The backend streams a whole turn in bursts that can run 10+
                # seconds ahead of playback, so a time window always leaks:
                # discard everything until the model closes this turn with a
                # listen or response.done boundary.
                self._discard_turn_active = True
            else:
                self._suppress_playback_until = now + INTERRUPT_POST_LISTEN_GUARD
            while True:
                try:
                    self._playback_chunks.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._playback_chunks.task_done()
            with self._state_lock:
                self._speaking_until = now
                self._interrupt_armed_at = math.inf
                self._model_state = "listening"
            # clear_player cycles the SHARED record+playback pipeline through
            # PAUSED→flush→PLAYING; hardware testing showed calling it from any
            # thread but the sole pusher can wedge the pipeline for good, so the
            # playback worker executes it.
            self._player_clear_pending = True

    def set_conversation_state(self, state: str) -> None:
        if state not in ANTENNA_POSES:
            raise ValueError(f"unknown conversation state: {state}")
        with self._state_lock:
            self._model_state = state

    def _capture_loop(self) -> None:
        agc = UplinkAGC()
        pending = np.empty(CHUNK_SAMPLES, dtype=np.float32)
        pending_size = 0
        last_capture_at = time.monotonic()
        stalled = False
        while not self._stop_event.is_set():
            try:
                sample = self.mini.media.get_audio_sample()
                if sample is None:
                    pending_size, last_capture_at, stalled = self._pad_capture_stall(
                        agc, pending, pending_size, last_capture_at, stalled
                    )
                    self._stop_event.wait(0.01)
                    continue
                if stalled:
                    stalled = False
                    log.info("Microphone capture resumed; silence padding stopped")
                last_capture_at = time.monotonic()
                mono = to_mono(sample)
                # The double-talk detector reads the raw post-AEC level; only
                # the uploaded slice gets the AGC lift. Its estimate adapts
                # only on frames DoA recently attributed to the user, so
                # ambient noise cannot drag the speech level down and end up
                # amplified eightfold.
                now = time.monotonic()
                with self._state_lock:
                    self._microphone_level_db = audio_level_db(mono)
                    speaking = now < self._speaking_until
                    user_recent = now - self._last_user_speech_at <= 1.5
                offset = 0
                while offset < mono.size:
                    copied = min(CHUNK_SAMPLES - pending_size, mono.size - offset)
                    pending[pending_size : pending_size + copied] = mono[offset : offset + copied]
                    pending_size += copied
                    offset += copied
                    if pending_size == CHUNK_SAMPLES:
                        self._put_latest(
                            agc.process(pending.copy(), adapt=user_recent and not speaking)
                        )
                        pending_size = 0
                if self._emit_partial_event.is_set() and pending_size > 0:
                    # A confirmed interruption must not wait for the slice to
                    # fill: send the user's words to the model immediately.
                    self._emit_partial_event.clear()
                    self._put_latest(agc.process(pending[:pending_size].copy(), adapt=False))
                    pending_size = 0
            except Exception as exc:
                log.warning("Microphone read failed: %s", exc)
                pending_size, last_capture_at, stalled = self._pad_capture_stall(
                    agc, pending, pending_size, last_capture_at, stalled
                )
                self._stop_event.wait(1.0)

    def _pad_capture_stall(
        self,
        agc: UplinkAGC,
        pending: np.ndarray,
        pending_size: int,
        last_capture_at: float,
        stalled: bool,
    ) -> tuple[int, float, bool]:
        """Substitute one silent chunk per chunk length while the mic is dead.

        The server drives one prefill/decode step per uplink chunk and never
        pads missing audio itself, so without this the whole conversation
        freezes (and the model's sense of time drifts) whenever the microphone
        stops delivering.
        """
        now = time.monotonic()
        if now - last_capture_at < CAPTURE_STALL_TIMEOUT:
            return pending_size, last_capture_at, stalled
        if not stalled:
            log.warning(
                "Microphone stalled for %.1f s; sending silence to keep the "
                "Omni slice clock alive",
                now - last_capture_at,
            )
        if pending_size > 0:
            # Finish the interrupted slice with silence so speech already
            # captured keeps its place on the timeline.
            pending[pending_size:] = 0.0
            self._put_latest(agc.process(pending.copy(), adapt=False))
        else:
            self._put_latest(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
        return 0, now, True

    def _camera_loop(self) -> None:
        """Encode stills off the audio sender and expose only the latest frame."""
        next_capture = time.monotonic()
        while not self._stop_event.is_set():
            try:
                frame = self.mini.media.get_frame_jpeg()
                if frame:
                    frame = downscale_jpeg(frame)
                    with self._camera_lock:
                        self._latest_frame_jpeg = frame
            except Exception as exc:
                log.debug("Camera frame unavailable: %s", exc)
            next_capture += CAMERA_PERIOD
            now = time.monotonic()
            if next_capture < now:
                next_capture = now
            self._stop_event.wait(next_capture - now)

    def _playback_loop(self) -> None:
        """Continuously resample TTS and feed a short jitter buffer to GStreamer."""
        resampler = StreamingAudioResampler()
        resampler_generation: int | None = None
        deferred_item: tuple[int, str, np.ndarray, bool] | None = None

        while not self._stop_event.is_set():
            with self._playback_lock:
                clear_pending = self._player_clear_pending
                self._player_clear_pending = False
            if clear_pending:
                # Only this thread may cycle the shared pipeline; see
                # _flush_playback_for_listen.
                self._clear_player()

            if deferred_item is None:
                try:
                    first_item = self._playback_chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
            else:
                first_item = deferred_item
                deferred_item = None

            items = [first_item]
            generation, response_id, samples, starts_response = first_item
            with self._state_lock:
                playback_was_idle = time.monotonic() >= self._speaking_until
                last_user_speech_at = self._last_user_speech_at
            # A drained mid-response resume pushes immediately: the hole has
            # already been heard, and buffering now would only stretch it.
            should_preroll = playback_was_idle and starts_response
            buffered_samples = samples.size
            preroll_samples = round(self._playback_preroll * OUTPUT_SAMPLE_RATE)
            if should_preroll and buffered_samples < preroll_samples:
                deadline = time.monotonic() + self._playback_preroll
                while not self._stop_event.is_set() and buffered_samples < preroll_samples:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self._playback_chunks.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item[0] != generation:
                        deferred_item = item
                        break
                    items.append(item)
                    buffered_samples += item[2].size

            try:
                if generation != resampler_generation or should_preroll:
                    resampler.reset()
                    resampler_generation = generation

                audio_parts = [item_samples for _, _, item_samples, _ in items]
                source = audio_parts[0] if len(audio_parts) == 1 else np.concatenate(audio_parts)
                resample_started = time.perf_counter()
                playback = resampler.process(source)
                resample_ms = (time.perf_counter() - resample_started) * 1_000
                if playback.size == 0:
                    continue
                with self._playback_lock:
                    now = time.monotonic()
                    playback_is_current = not (
                        generation != self._playback_generation
                        or now < self._suppress_playback_until
                    )
                    if playback_is_current:
                        with self._state_lock:
                            if now >= self._speaking_until:
                                self._interrupt_armed_at = now + INTERRUPT_ARM_DELAY
                            start = max(now, self._speaking_until)
                            self._speaking_until = start + playback.size / INPUT_SAMPLE_RATE
                            self._model_state = "speaking"
                        push_started = time.perf_counter()
                        self.mini.media.push_audio_sample(playback)
                        push_ms = (time.perf_counter() - push_started) * 1_000
                        if should_preroll:
                            # Every started utterance earns a slightly faster
                            # first word until a supply gap says otherwise.
                            self._playback_preroll = max(
                                PLAYBACK_PREROLL_MIN,
                                self._playback_preroll - PLAYBACK_PREROLL_DECAY,
                            )
                if not playback_is_current:
                    resampler.reset()
                    resampler_generation = None
                    continue
                if playback_was_idle:
                    # The single end-to-end number every latency tuning knob
                    # (chunk cadence, preroll, model) must ultimately improve.
                    turn_gap = time.monotonic() - last_user_speech_at
                    if starts_response and 0.0 < turn_gap < 10.0:
                        log.info(
                            "Turn latency for %s: %.0f ms from last heard user speech",
                            response_id,
                            turn_gap * 1_000,
                        )
                    log.debug(
                        "Playback %s started with %.0f ms buffered; resample=%.1f ms, push=%.1f ms",
                        response_id,
                        source.size / OUTPUT_SAMPLE_RATE * 1_000,
                        resample_ms,
                        push_ms,
                    )
                elif resample_ms + push_ms > 100.0:
                    # Only a stage cost that threatens the real-time budget of
                    # a one-second slice is worth a warning; the CM4 routinely
                    # spends a few tens of ms here under load.
                    log.warning(
                        "Slow playback stage for %s: resample=%.1f ms, push=%.1f ms",
                        response_id,
                        resample_ms,
                        push_ms,
                    )
            except Exception as exc:
                log.warning("Speaker playback failed: %s", exc)
            finally:
                for _ in items:
                    self._playback_chunks.task_done()

    def _put_latest(self, chunk: np.ndarray) -> None:
        try:
            self._audio_chunks.put_nowait(chunk)
        except queue.Full:
            with suppress(queue.Empty):
                self._audio_chunks.get_nowait()
            self._audio_chunks.put_nowait(chunk)
            self._dropped_audio_chunks += 1
            now = time.monotonic()
            if now - self._last_audio_drop_log >= 5.0:
                log.warning(
                    "Dropped %d stale microphone slice(s) due to upload backpressure",
                    self._dropped_audio_chunks,
                )
                self._last_audio_drop_log = now

    def _motion_loop(self) -> None:
        last_effective_state = ""
        last_doa: float | None = None
        last_turn = 0.0
        last_command_error = -math.inf
        connection_error_started_at: float | None = None
        interrupt_candidate_since: float | None = None
        interrupt_cooldown_until = 0.0
        far_end_floor_db: float | None = None
        started_at = time.monotonic()
        state_entered_at = started_at
        last_tick = started_at
        next_tick = started_at
        next_doa_read = started_at
        latest_doa: tuple[float, bool] | None = None
        last_user_speech_at = -math.inf
        user_was_speaking = False
        listening_antennas = self._command_antennas.copy()
        nod_started_at: float | None = None
        next_nod_at = math.inf
        glance_started_at: float | None = None
        glance_duration = 1.6
        glance_yaw = 0.0
        glance_pitch = 0.0
        next_glance_at = started_at + self._motion_rng.uniform(12.0, 20.0)

        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed = min(max(now - last_tick, 0.001), 0.1)
            last_tick = now
            with self._state_lock:
                speaking = now < self._speaking_until
                state = self._model_state
                interrupt_armed_at = self._interrupt_armed_at
                microphone_level_db = self._microphone_level_db
            effective_state = effective_conversation_state(state, speaking)

            if effective_state != last_effective_state:
                last_effective_state = effective_state
                state_entered_at = now
                if effective_state == "speaking":
                    far_end_floor_db = microphone_level_db
                    interrupt_candidate_since = None
                    nod_started_at = None
                    glance_started_at = None
                    next_glance_at = now + self._motion_rng.uniform(12.0, 20.0)

            # DoA is the gaze source while listening. Its speech bit is PRE-AEC
            # and stays high whenever the robot's own speaker plays, so it takes
            # no part in double-talk detection; the adaptive post-AEC level
            # threshold is what prevents Reachy from interrupting itself.
            doa_sampled = now >= next_doa_read
            if doa_sampled:
                latest_doa = self._read_doa()
                next_doa_read = max(next_doa_read + DOA_PERIOD, now + DOA_PERIOD)
            speech_detected = latest_doa is not None and latest_doa[1]
            if speaking:
                if now < interrupt_armed_at:
                    # Learn the current far-end residual during the startup
                    # guard instead of treating the first loud speaker frame
                    # as near-end speech.
                    far_end_floor_db = microphone_level_db
                elif far_end_floor_db is None:
                    far_end_floor_db = microphone_level_db
                elif microphone_level_db <= far_end_floor_db:
                    far_end_floor_db = 0.7 * far_end_floor_db + 0.3 * microphone_level_db
                else:
                    far_end_floor_db = min(
                        microphone_level_db,
                        far_end_floor_db + INTERRUPT_FLOOR_RISE_DB_PER_SAMPLE,
                    )
                interrupt_threshold_db = max(
                    INTERRUPT_MIN_LEVEL_DB,
                    far_end_floor_db + INTERRUPT_MIN_RISE_DB,
                )
                near_end_candidate = (
                    now >= interrupt_armed_at
                    and now >= interrupt_cooldown_until
                    and microphone_level_db >= interrupt_threshold_db
                )
                if near_end_candidate:
                    with self._state_lock:
                        self._last_near_end_activity_at = now
                    if interrupt_candidate_since is None:
                        interrupt_candidate_since = now
                    elif now - interrupt_candidate_since >= INTERRUPT_CONFIRM_SECONDS:
                        if self._request_user_interrupt(
                            microphone_level_db,
                            interrupt_threshold_db,
                        ):
                            interrupt_cooldown_until = now + INTERRUPT_COOLDOWN
                        interrupt_candidate_since = None
                else:
                    interrupt_candidate_since = None
            else:
                far_end_floor_db = None
                interrupt_candidate_since = None
                if speech_detected:
                    last_user_speech_at = now
                    with self._state_lock:
                        self._last_user_speech_at = now
                if doa_sampled and latest_doa is not None:
                    angle = latest_doa[0]
                    moved_enough = (
                        last_doa is None or angular_distance(angle, last_doa) >= DOA_TURN_THRESHOLD
                    )
                    if speech_detected and moved_enough and now - last_turn >= DOA_TURN_COOLDOWN:
                        target = self._head_target_towards(angle)
                        if target is not None:
                            self._target_head = target
                            last_doa = angle
                            last_turn = now

            user_speaking = not speaking and now - last_user_speech_at <= USER_SPEECH_HOLD
            if user_speaking and not user_was_speaking:
                listening_antennas = self._command_antennas.copy()
                next_nod_at = now + self._motion_rng.uniform(2.2, 4.0)
                next_glance_at = now + self._motion_rng.uniform(12.0, 20.0)
            elif not user_speaking and user_was_speaking:
                next_nod_at = math.inf
            user_was_speaking = user_speaking

            if user_speaking and nod_started_at is None and now >= next_nod_at:
                nod_started_at = now
                next_nod_at = now + self._motion_rng.uniform(4.0, 7.0)
            nod_pulse = 0.0
            if nod_started_at is not None:
                nod_progress = (now - nod_started_at) / LISTENING_NOD_DURATION
                nod_pulse = gesture_pulse(nod_progress)
                if nod_progress > 1.0:
                    nod_started_at = None

            can_glance = not speaking and not user_speaking
            if can_glance and glance_started_at is None and now >= next_glance_at:
                glance_started_at = now
                glance_duration = self._motion_rng.uniform(1.4, 2.0)
                glance_yaw = self._motion_rng.choice((-1.0, 1.0)) * self._motion_rng.uniform(
                    3.0, 6.0
                )
                glance_pitch = self._motion_rng.uniform(-1.2, 1.4)
            glance_pulse = 0.0
            if glance_started_at is not None:
                glance_progress = (now - glance_started_at) / glance_duration
                glance_pulse = gesture_pulse(glance_progress)
                if glance_progress > 1.0:
                    glance_started_at = None
                    next_glance_at = now + self._motion_rng.uniform(12.0, 24.0)

            transition_pulse = gesture_pulse((now - state_entered_at) / STATE_TRANSITION_DURATION)
            head_offset, animated_antennas = lifelike_motion_overlay(
                now - started_at,
                effective_state,
                user_speaking=user_speaking,
                transition_pulse=transition_pulse,
                nod_pulse=nod_pulse,
                glance_pulse=glance_pulse,
                glance_yaw_degrees=glance_yaw,
                glance_pitch_degrees=glance_pitch,
            )
            self._target_antennas = (
                listening_antennas.copy() if user_speaking else animated_antennas
            )

            self._command_head = smooth_pose_step(
                self._command_head,
                self._target_head,
                elapsed,
            )
            commanded_head = compose_world_offset(self._command_head, head_offset)
            antenna_delta = np.clip(
                self._target_antennas - self._command_antennas,
                -MAX_ANTENNA_SPEED * elapsed,
                MAX_ANTENNA_SPEED * elapsed,
            )
            self._command_antennas = self._command_antennas + antenna_delta
            try:
                self.mini.set_target(
                    head=commanded_head,
                    antennas=self._command_antennas,
                    body_yaw=None,
                )
            except Exception as exc:
                # reachy-mini 1.9 checks liveness in a separate thread. A
                # missed heartbeat can make send_command() reject commands for
                # one check cycle even though the socket is still receiving
                # and recovers by itself. Keep retrying at the next motion tick
                # and only surface this particular error if it persists beyond
                # the SDK's two-second check cycle. Other command failures are
                # reported immediately.
                transient_liveness_miss = (
                    isinstance(exc, ConnectionError)
                    and str(exc).rstrip(".") == "Lost connection with the server"
                )
                if transient_liveness_miss:
                    if connection_error_started_at is None:
                        connection_error_started_at = now
                    report_error = now - connection_error_started_at >= MOTION_CONNECTION_GRACE
                else:
                    connection_error_started_at = None
                    report_error = True
                if report_error and now - last_command_error >= MOTION_ERROR_LOG_INTERVAL:
                    log.warning("Motion command failed: %s", exc)
                    last_command_error = now
            else:
                connection_error_started_at = None

            next_tick += MOTION_PERIOD
            remaining = next_tick - time.monotonic()
            if remaining < 0.0:
                next_tick = time.monotonic()
                remaining = 0.0
            self._stop_event.wait(remaining)

    def _read_doa(self) -> tuple[float, bool] | None:
        try:
            return self.mini.media.get_DoA()
        except Exception as exc:
            log.debug("DoA unavailable: %s", exc)
            return None

    def _head_target_towards(self, angle: float) -> np.ndarray | None:
        try:
            try:
                head_pose = self.mini.get_current_head_pose()
            except Exception:
                head_pose = self._command_head
            world = doa_world_direction(angle, head_pose)
            return self.mini.look_at_world(*world, perform_movement=False)
        except Exception as exc:
            log.warning("Could not calculate DoA target: %s", exc)
            return None

    def _clear_player(self) -> None:
        clear_player = getattr(getattr(self.mini.media, "audio", None), "clear_player", None)
        if callable(clear_player):
            try:
                clear_player()
            except Exception as exc:
                log.debug("Could not clear speaker buffer: %s", exc)

    def _apply_audio_startup_config(self) -> None:
        audio = getattr(self.mini.media, "audio", None)
        apply_config = getattr(audio, "apply_audio_config", None)
        if not callable(apply_config):
            log.warning("Reachy audio DSP configuration is unavailable")
            return
        try:
            if apply_config(
                AUDIO_STARTUP_CONFIG,
                verify=True,
                write_settle_seconds=AUDIO_CONFIG_SETTLE_SECONDS,
            ):
                log.info("Applied the official conversation audio DSP configuration")
            else:
                log.warning("Reachy audio DSP configuration was not applied")
        except Exception as exc:
            log.warning("Could not apply Reachy audio DSP configuration: %s", exc)

    def _initialize_motion_state(self) -> None:
        try:
            self._command_head = np.asarray(
                self.mini.get_current_head_pose(), dtype=np.float64
            ).copy()
        except Exception as exc:
            log.warning("Could not read initial head pose: %s", exc)
            self._command_head = np.eye(4, dtype=np.float64)
        try:
            self._command_antennas = np.asarray(
                self.mini.get_present_antenna_joint_positions(), dtype=np.float64
            )
        except Exception as exc:
            log.warning("Could not read initial antenna pose: %s", exc)
            self._command_antennas = np.deg2rad(ANTENNA_POSES["idle"])
        try:
            self._target_head = create_head_pose(
                pitch=NATURAL_HEAD_PITCH_DEGREES,
                degrees=True,
            )
        except Exception as exc:
            log.warning("Could not create natural head target: %s", exc)
            self._target_head = np.eye(4, dtype=np.float64)
        self._target_antennas = self._command_antennas.copy()

    def _return_to_neutral(self) -> None:
        try:
            from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE

            start_head = self._command_head.copy()
            start_antennas = self._command_antennas.copy()
            duration = 0.8
            steps = round(duration / MOTION_PERIOD)
            for index in range(1, steps + 1):
                fraction = time_trajectory(index / steps)
                head = linear_pose_interpolation(start_head, INIT_HEAD_POSE, fraction)
                antennas = (
                    start_antennas
                    + (np.asarray(INIT_ANTENNAS_JOINT_POSITIONS) - start_antennas) * fraction
                )
                self.mini.set_target(head=head, antennas=antennas, body_yaw=None)
                time.sleep(MOTION_PERIOD)
            self._command_head = np.asarray(INIT_HEAD_POSE).copy()
            self._command_antennas = np.asarray(INIT_ANTENNAS_JOINT_POSITIONS).copy()
        except Exception as exc:
            log.warning("Could not return Reachy to neutral: %s", exc)
