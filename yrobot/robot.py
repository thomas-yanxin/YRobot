"""Reachy Mini audio, camera, and restrained phase-A motion adapter."""

from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from collections.abc import Mapping
from contextlib import suppress

import numpy as np
from reachy_mini.utils.interpolation import (
    delta_angle_between_mat_rot,
    linear_pose_interpolation,
    time_trajectory,
)
from scipy.signal import resample_poly

from .config import CHUNK_SAMPLES, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE

log = logging.getLogger(__name__)

ANTENNA_POSES: Mapping[str, tuple[float, float]] = {
    "idle": (-10.0, 10.0),
    "listening": (-22.0, 22.0),
    "speaking": (-30.0, 30.0),
}

# Tuned XVF3800 parameters from the official Reachy Mini conversation app.
AUDIO_STARTUP_CONFIG = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)

MOTION_PERIOD = 0.05
BARGE_IN_ARM_DELAY = 0.3
BARGE_IN_CONFIRMATIONS = 3
BARGE_IN_COOLDOWN = 1.0
BARGE_IN_MIN_LEVEL_DB = -32.0
BARGE_IN_RELEASE_SILENCE = 0.7
BARGE_IN_POST_LISTEN_GUARD = 0.4
MAX_HEAD_ANGULAR_STEP = math.radians(2.0)
MAX_HEAD_TRANSLATION_STEP = 0.003
MAX_ANTENNA_STEP = math.radians(4.0)
NATURAL_HEAD_PITCH_DEGREES = -4.0
DOA_GAZE_ELEVATION = math.tan(math.radians(-NATURAL_HEAD_PITCH_DEGREES))


def to_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2 and audio.shape[1] >= 1:
        return audio.mean(axis=1, dtype=np.float32)
    raise ValueError(f"unexpected microphone shape: {audio.shape}")


def audio_level_db(samples: np.ndarray) -> float:
    """Return a stable dBFS RMS level for AEC-processed microphone PCM."""
    audio = np.asarray(samples, dtype=np.float64)
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(audio * audio)))
    return 20.0 * math.log10(max(rms, 1e-6))


def is_near_end_speech(speech_detected: bool, microphone_level_db: float) -> bool:
    """Require hardware speech VAD and audible post-AEC microphone energy."""
    return speech_detected and microphone_level_db >= BARGE_IN_MIN_LEVEL_DB


def resample_audio(
    samples: np.ndarray,
    input_rate: int = OUTPUT_SAMPLE_RATE,
    output_rate: int = INPUT_SAMPLE_RATE,
) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("playback audio must be mono")
    divisor = math.gcd(input_rate, output_rate)
    converted = resample_poly(audio, output_rate // divisor, input_rate // divisor)
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


class RobotIO:
    """Own Reachy media and serialize every application-level motion command."""

    def __init__(self, mini: object) -> None:
        self.mini = mini
        self._audio_chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        # Keep network reception independent from GStreamer.  TTS callbacks can
        # arrive in bursts; pushing them from the WebSocket receive loop makes
        # that loop stop reading while the audio backend catches up.
        self._playback_chunks: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
        self._playback_lock = threading.Lock()
        self._playback_generation = 0
        self._suppress_playback_until = 0.0
        self._barge_in_event = threading.Event()
        self._force_listen_sent = threading.Event()
        self._omni_listen_confirmed = threading.Event()
        self._barge_in_user_done = threading.Event()
        self._expected_listen_response_id: str | None = None
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._playback_thread: threading.Thread | None = None
        self._motion_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._model_state = "idle"
        self._speaking_until = 0.0
        self._barge_in_armed_at = math.inf
        self._last_near_end_speech_at = 0.0
        self._microphone_level_db = -120.0
        self._recording = False
        self._playing = False
        self._wobbling = False
        self._command_head = np.eye(4, dtype=np.float64)
        self._target_head = self._command_head.copy()
        self._command_antennas = np.deg2rad(ANTENNA_POSES["idle"])
        self._target_antennas = self._command_antennas.copy()

    def start(self) -> None:
        self.mini.enable_motors()
        self._initialize_motion_state()
        self._apply_audio_startup_config()
        self.mini.media.start_recording()
        self._recording = True
        self.mini.media.start_playing()
        self._playing = True
        self.mini.enable_wobbling()
        self._wobbling = True
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="yrobot-audio", daemon=True
        )
        self._playback_thread = threading.Thread(
            target=self._playback_loop, name="yrobot-playback", daemon=True
        )
        self._motion_thread = threading.Thread(
            target=self._motion_loop, name="yrobot-motion", daemon=True
        )
        self._capture_thread.start()
        self._playback_thread.start()
        self._motion_thread.start()
        log.info("Reachy media and motion started")

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (self._capture_thread, self._playback_thread, self._motion_thread):
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
        try:
            return self.mini.media.get_frame_jpeg()
        except Exception as exc:
            log.debug("Camera frame unavailable: %s", exc)
            return None

    def play_omni_audio(self, samples: np.ndarray) -> bool:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("playback audio must be mono")
        if audio.size == 0:
            return False
        # decode_pcm owns its array, but copying here keeps this port safe for
        # callers that reuse a mutable input buffer after returning.
        with self._playback_lock:
            if time.monotonic() < self._suppress_playback_until:
                return False
            generation = self._playback_generation
            self._playback_chunks.put((generation, audio.copy()))
        return True

    def interrupt_omni_audio(self) -> bool:
        """Atomically drop pending and device-buffered speech for barge-in."""
        now = time.monotonic()
        with self._playback_lock:
            with self._state_lock:
                active = now < self._speaking_until or not self._playback_chunks.empty()
            if not active:
                return False

            # A generation invalidates a chunk even when the playback worker
            # has already removed it from Queue and is resampling it.
            self._playback_generation += 1
            self._suppress_playback_until = math.inf
            self._force_listen_sent.clear()
            self._omni_listen_confirmed.clear()
            self._barge_in_user_done.clear()
            self._expected_listen_response_id = None
            self._barge_in_event.set()
            while True:
                try:
                    self._playback_chunks.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._playback_chunks.task_done()

            with self._state_lock:
                self._speaking_until = now
                self._barge_in_armed_at = math.inf
                self._last_near_end_speech_at = now
                self._model_state = "listening"
            # Serialize the SDK flush with playback pushes so an old chunk
            # cannot be submitted immediately after clear_player().
            self._clear_player()

        log.info("User barge-in: cleared queued speech and resumed listening")
        return True

    def force_listen_active(self) -> bool:
        """Keep force_listen on every input until both sides finish barge-in."""
        return self._barge_in_event.is_set()

    def note_force_listen_sent(self, response_id: str) -> None:
        if not self._barge_in_event.is_set():
            return
        with self._state_lock:
            expected = self._expected_listen_response_id
            expected_session = expected.rpartition("_resp_")[0] if expected else ""
            response_session = response_id.rpartition("_resp_")[0]
            if expected is None or expected_session != response_session:
                self._expected_listen_response_id = response_id
        self._force_listen_sent.set()

    def confirm_omni_listening(self, response_id: str) -> None:
        """Record a server listen event that follows a force_listen input."""
        with self._state_lock:
            expected = self._expected_listen_response_id
        if (
            self._barge_in_event.is_set()
            and self._force_listen_sent.is_set()
            and response_id == expected
        ):
            self._omni_listen_confirmed.set()
            self._try_finish_barge_in()

    def _update_barge_in_release(self, near_end_speech: bool, now: float) -> None:
        if not self._barge_in_event.is_set():
            return
        if near_end_speech:
            with self._state_lock:
                self._last_near_end_speech_at = now
            self._barge_in_user_done.clear()
            return
        with self._state_lock:
            silent_for = now - self._last_near_end_speech_at
        if silent_for >= BARGE_IN_RELEASE_SILENCE:
            self._barge_in_user_done.set()
            self._try_finish_barge_in()

    def _try_finish_barge_in(self) -> None:
        if not (
            self._barge_in_event.is_set()
            and self._barge_in_user_done.is_set()
            and self._force_listen_sent.is_set()
            and self._omni_listen_confirmed.is_set()
        ):
            return
        with self._playback_lock:
            if not self._barge_in_event.is_set():
                return
            self._barge_in_event.clear()
            self._force_listen_sent.clear()
            self._omni_listen_confirmed.clear()
            self._barge_in_user_done.clear()
            with self._state_lock:
                self._expected_listen_response_id = None
            # Drop any asynchronous TTS callback already in flight after the
            # listen acknowledgement. A new response cannot arrive this soon.
            self._suppress_playback_until = (
                time.monotonic() + BARGE_IN_POST_LISTEN_GUARD
            )
        log.info("Omni listen confirmed; barge-in complete")

    def set_conversation_state(self, state: str) -> None:
        if state not in ANTENNA_POSES:
            raise ValueError(f"unknown conversation state: {state}")
        with self._state_lock:
            self._model_state = state

    def _capture_loop(self) -> None:
        pending = np.empty(0, dtype=np.float32)
        while not self._stop_event.is_set():
            try:
                sample = self.mini.media.get_audio_sample()
                if sample is None:
                    self._stop_event.wait(0.01)
                    continue
                mono = to_mono(sample)
                with self._state_lock:
                    self._microphone_level_db = audio_level_db(mono)
                pending = np.concatenate((pending, mono))
                while pending.size >= CHUNK_SAMPLES:
                    chunk = pending[:CHUNK_SAMPLES].copy()
                    pending = pending[CHUNK_SAMPLES:]
                    self._put_latest(chunk)
            except Exception as exc:
                log.warning("Microphone read failed: %s", exc)
                self._stop_event.wait(1.0)

    def _playback_loop(self) -> None:
        """Feed GStreamer in order without ever blocking the WebSocket reader."""
        while not self._stop_event.is_set():
            try:
                generation, samples = self._playback_chunks.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                playback = resample_audio(samples)
                if playback.size == 0:
                    continue
                with self._playback_lock:
                    now = time.monotonic()
                    if (
                        generation != self._playback_generation
                        or now < self._suppress_playback_until
                    ):
                        continue
                    with self._state_lock:
                        if now >= self._speaking_until:
                            self._barge_in_armed_at = now + BARGE_IN_ARM_DELAY
                        start = max(now, self._speaking_until)
                        self._speaking_until = start + playback.size / INPUT_SAMPLE_RATE
                        self._model_state = "speaking"
                    self.mini.media.push_audio_sample(playback)
            except Exception as exc:
                log.warning("Speaker playback failed: %s", exc)
            finally:
                self._playback_chunks.task_done()

    def _put_latest(self, chunk: np.ndarray) -> None:
        try:
            self._audio_chunks.put_nowait(chunk)
        except queue.Full:
            with suppress(queue.Empty):
                self._audio_chunks.get_nowait()
            self._audio_chunks.put_nowait(chunk)

    def _motion_loop(self) -> None:
        last_effective_state = ""
        last_doa: float | None = None
        last_turn = 0.0
        next_idle_motion = time.monotonic() + random.uniform(5.0, 8.0)
        last_command_error = 0.0
        barge_in_frames = 0
        barge_in_cooldown_until = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            with self._state_lock:
                speaking = now < self._speaking_until
                state = self._model_state
                barge_in_armed_at = self._barge_in_armed_at
                microphone_level_db = self._microphone_level_db
            effective_state = "speaking" if speaking else state

            if effective_state != last_effective_state:
                self._target_antennas = np.deg2rad(
                    ANTENNA_POSES.get(effective_state, ANTENNA_POSES["idle"])
                )
                last_effective_state = effective_state

            # DoA's hardware speech bit also sees Reachy's own speaker. It may
            # assist barge-in, but only post-AEC microphone PCM is allowed to
            # confirm that audible near-end speech remains.
            doa = self._read_doa()
            speech_detected = doa is not None and doa[1]
            near_end_speech = is_near_end_speech(
                speech_detected,
                microphone_level_db,
            )
            self._update_barge_in_release(near_end_speech, now)
            if speaking:
                if (
                    near_end_speech
                    and now >= barge_in_armed_at
                    and now >= barge_in_cooldown_until
                ):
                    barge_in_frames += 1
                    if barge_in_frames >= BARGE_IN_CONFIRMATIONS:
                        if self.interrupt_omni_audio():
                            barge_in_cooldown_until = now + BARGE_IN_COOLDOWN
                            log.info(
                                "Near-end speech confirmed at %.1f dBFS",
                                microphone_level_db,
                            )
                        barge_in_frames = 0
                else:
                    barge_in_frames = 0
            else:
                barge_in_frames = 0
                if doa is not None:
                    angle = doa[0]
                    moved_enough = last_doa is None or angular_distance(angle, last_doa) >= 0.12
                    if speech_detected and moved_enough and now - last_turn >= 0.8:
                        target = self._head_target_towards(angle)
                        if target is not None:
                            self._target_head = target
                            last_doa = angle
                            last_turn = now
                            next_idle_motion = last_turn + random.uniform(5.0, 8.0)
                if not speech_detected and effective_state == "idle" and now >= next_idle_motion:
                    target = self._idle_glance_target()
                    if target is not None:
                        self._target_head = target
                    next_idle_motion = time.monotonic() + random.uniform(6.0, 10.0)

            self._command_head = step_pose(self._command_head, self._target_head)
            antenna_delta = np.clip(
                self._target_antennas - self._command_antennas,
                -MAX_ANTENNA_STEP,
                MAX_ANTENNA_STEP,
            )
            self._command_antennas = self._command_antennas + antenna_delta
            try:
                self.mini.set_target(
                    head=self._command_head,
                    antennas=self._command_antennas,
                    body_yaw=None,
                )
            except Exception as exc:
                if now - last_command_error >= 5.0:
                    log.warning("Motion command failed: %s", exc)
                    last_command_error = now

            self._stop_event.wait(MOTION_PERIOD)

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

    def _idle_glance_target(self) -> np.ndarray | None:
        try:
            from reachy_mini.utils import create_head_pose

            return create_head_pose(
                roll=random.uniform(-2.0, 2.0),
                pitch=random.uniform(
                    NATURAL_HEAD_PITCH_DEGREES - 2.0,
                    NATURAL_HEAD_PITCH_DEGREES + 2.0,
                ),
                yaw=random.uniform(-4.0, 4.0),
                degrees=True,
            )
        except Exception as exc:
            log.debug("Idle motion unavailable: %s", exc)
            return None

    def _clear_player(self) -> None:
        audio = getattr(self.mini.media, "audio", None)
        clear_player = getattr(audio, "clear_player", None)
        if not callable(clear_player):
            clear_player = getattr(audio, "clear_output_buffer", None)
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
            if apply_config(AUDIO_STARTUP_CONFIG, verify=True):
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
            from reachy_mini.utils import create_head_pose

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
