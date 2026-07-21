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


def to_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2 and audio.shape[1] >= 1:
        return audio.mean(axis=1, dtype=np.float32)
    raise ValueError(f"unexpected microphone shape: {audio.shape}")


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
    """Apply the official DoA head-to-world transform."""
    pose = np.asarray(head_pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("head_pose must be a 4x4 transform")
    direction_head = np.array([math.sin(angle), math.cos(angle), 0.0])
    return pose[:3, :3] @ direction_head


def angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class RobotIO:
    """Own Reachy media and serialize every application-level motion command."""

    def __init__(self, mini: object) -> None:
        self.mini = mini
        self._audio_chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._motion_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._model_state = "idle"
        self._speaking_until = 0.0
        self._recording = False
        self._playing = False
        self._wobbling = False

    def start(self) -> None:
        self.mini.enable_motors()
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
        self._motion_thread = threading.Thread(
            target=self._motion_loop, name="yrobot-motion", daemon=True
        )
        self._capture_thread.start()
        self._motion_thread.start()
        log.info("Reachy media and motion started")

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (self._capture_thread, self._motion_thread):
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

    def play_omni_audio(self, samples: np.ndarray) -> None:
        playback = resample_audio(samples)
        if playback.size == 0:
            return
        with self._state_lock:
            start = max(time.monotonic(), self._speaking_until)
            self._speaking_until = start + playback.size / INPUT_SAMPLE_RATE
            self._model_state = "speaking"
        self.mini.media.push_audio_sample(playback)

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
                pending = np.concatenate((pending, to_mono(sample)))
                while pending.size >= CHUNK_SAMPLES:
                    chunk = pending[:CHUNK_SAMPLES].copy()
                    pending = pending[CHUNK_SAMPLES:]
                    self._put_latest(chunk)
            except Exception as exc:
                log.warning("Microphone read failed: %s", exc)
                self._stop_event.wait(1.0)

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

        while not self._stop_event.is_set():
            now = time.monotonic()
            with self._state_lock:
                speaking = now < self._speaking_until
                state = self._model_state
            effective_state = "speaking" if speaking else state

            if effective_state != last_effective_state:
                self._set_antennas(effective_state)
                last_effective_state = effective_state

            if not speaking:
                doa = self._read_doa()
                if doa is not None:
                    angle, speech_detected = doa
                    moved_enough = last_doa is None or angular_distance(angle, last_doa) >= 0.12
                    if speech_detected and moved_enough and now - last_turn >= 0.8:
                        if self._look_towards(angle):
                            last_doa = angle
                            last_turn = time.monotonic()
                            next_idle_motion = last_turn + random.uniform(5.0, 8.0)
                elif effective_state == "idle" and now >= next_idle_motion:
                    self._idle_glance()
                    next_idle_motion = time.monotonic() + random.uniform(6.0, 10.0)

            self._stop_event.wait(0.1)

    def _read_doa(self) -> tuple[float, bool] | None:
        try:
            return self.mini.media.get_DoA()
        except Exception as exc:
            log.debug("DoA unavailable: %s", exc)
            return None

    def _look_towards(self, angle: float) -> bool:
        try:
            world = doa_world_direction(angle, self.mini.get_current_head_pose())
            target = self.mini.look_at_world(*world, perform_movement=False)
            self.mini.goto_target(head=target, duration=0.45, body_yaw=None)
            return True
        except Exception as exc:
            log.warning("DoA motion failed: %s", exc)
            return False

    def _idle_glance(self) -> None:
        try:
            from reachy_mini.utils import create_head_pose

            target = create_head_pose(
                roll=random.uniform(-2.0, 2.0),
                pitch=random.uniform(-2.0, 2.0),
                yaw=random.uniform(-4.0, 4.0),
                degrees=True,
            )
            self.mini.goto_target(head=target, duration=0.8, body_yaw=None)
        except Exception as exc:
            log.debug("Idle motion unavailable: %s", exc)

    def _set_antennas(self, state: str) -> None:
        angles = ANTENNA_POSES.get(state, ANTENNA_POSES["idle"])
        try:
            self.mini.goto_target(antennas=np.deg2rad(angles), duration=0.35, body_yaw=None)
        except Exception as exc:
            log.debug("Antenna posture unavailable: %s", exc)

    def _clear_player(self) -> None:
        audio = getattr(self.mini.media, "audio", None)
        clear_player = getattr(audio, "clear_player", None)
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

    def _return_to_neutral(self) -> None:
        try:
            from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE

            self.mini.goto_target(
                head=INIT_HEAD_POSE,
                antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                duration=0.8,
                body_yaw=None,
            )
        except Exception as exc:
            log.warning("Could not return Reachy to neutral: %s", exc)
