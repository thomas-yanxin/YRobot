"""A mock ``ReachyMini`` that mirrors the subset of the SDK this app uses.

Goals:
* Let the whole pipeline run with **no robot and no daemon**.
* When possible, use the Mac's own **microphone / speaker / webcam** (via ``sounddevice`` /
  ``opencv``) so ``--sim`` is a genuine end-to-end demo, not just a stub.
* Degrade gracefully to synthetic silence / frames when those libs are missing.

The API surface here is intentionally the same shape as ``reachy_mini.ReachyMini`` and
``mini.media.*`` so :class:`~reachy_mini_live_chat.pipeline.Pipeline` never branches on sim vs real.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import List, Optional

import numpy as np

log = logging.getLogger("live_chat.sim")

_SR = 16000  # Reachy media is 16 kHz float32


class _FakeMedia:
    """Mimics ``mini.media``: mic capture, speaker playback, DOA, camera."""

    def __init__(self) -> None:
        self._sr = _SR
        self._in_ch = 1
        self._out_ch = 1
        self._recording = False
        self._playing = False

        self._sd = None       # sounddevice module, if available
        self._in_stream = None
        self._out_stream = None
        self._mic_buf: List[np.ndarray] = []
        self._mic_lock = threading.Lock()

        self._cv2 = None
        self._cap = None
        self._t0 = time.monotonic()

    # -- microphone ---------------------------------------------------------
    def start_recording(self) -> None:
        if self._recording:
            return
        self._recording = True
        try:
            import sounddevice as sd  # lazy

            self._sd = sd

            def _cb(indata, frames, time_info, status):  # noqa: ANN001
                with self._mic_lock:
                    self._mic_buf.append(indata.copy().reshape(-1, 1).astype(np.float32))

            self._in_stream = sd.InputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                blocksize=int(self._sr * 0.02), callback=_cb,
            )
            self._in_stream.start()
            log.info("sim mic: using Mac microphone @ %d Hz", self._sr)
        except Exception as e:  # no sounddevice / no mic
            log.warning("sim mic: no real microphone (%s); returning synthetic silence", e)
            self._sd = None

    def get_audio_sample(self) -> Optional[np.ndarray]:
        """Return buffered mic audio as (N, 2) float32 @ 16 kHz, or None if empty."""
        if not self._recording:
            return None
        if self._sd is not None:
            with self._mic_lock:
                if not self._mic_buf:
                    return None
                mono = np.concatenate(self._mic_buf, axis=0)
                self._mic_buf.clear()
            return np.repeat(mono, 2, axis=1)  # (N,1) -> (N,2)
        # synthetic: a short block of silence, paced to real time so the capture
        # loop doesn't busy-spin a core when there's no real microphone.
        time.sleep(0.02)
        n = int(self._sr * 0.02)
        return np.zeros((n, 2), dtype=np.float32)

    def stop_recording(self) -> None:
        self._recording = False
        if self._in_stream is not None:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

    # -- speaker ------------------------------------------------------------
    def start_playing(self) -> None:
        if self._playing:
            return
        self._playing = True
        try:
            import sounddevice as sd

            self._sd = self._sd or sd
            self._out_stream = sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                blocksize=int(self._sr * 0.02),
            )
            self._out_stream.start()
            log.info("sim speaker: using Mac speaker @ %d Hz", self._sr)
        except Exception as e:
            log.warning("sim speaker: no real output (%s); playback is silent", e)
            self._out_stream = None

    def push_audio_sample(self, data: np.ndarray) -> None:
        """Non-blocking write. ``data`` is (N,1|2) float32 @ 16 kHz."""
        if data is None or len(data) == 0:
            return
        mono = data.mean(axis=1) if data.ndim == 2 else data
        mono = np.clip(mono, -1.0, 1.0).astype(np.float32)
        if self._out_stream is not None:
            try:
                self._out_stream.write(mono.reshape(-1, 1))
            except Exception:
                pass

    def stop_playing(self) -> None:
        self._playing = False
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

    def play_sound(self, path: str) -> None:  # noqa: D401
        log.info("sim: play_sound(%s) [no-op]", path)

    # -- formats ------------------------------------------------------------
    def get_input_audio_samplerate(self) -> int:
        return self._sr

    def get_output_audio_samplerate(self) -> int:
        return self._sr

    def get_input_channels(self) -> int:
        return 2

    def get_output_channels(self) -> int:
        return self._out_ch

    # -- DOA ----------------------------------------------------------------
    def get_DoA(self):  # -> tuple[float, bool] | None
        # A single-mic Mac cannot localize; real DOA needs the robot's array.
        # Return None so the motion controller simply skips DOA in sim.
        return None

    # -- camera -------------------------------------------------------------
    def _ensure_cam(self) -> None:
        if self._cap is not None:
            return
        try:
            import cv2  # lazy

            self._cv2 = cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self._cap = cap
                log.info("sim camera: using Mac webcam")
            else:
                cap.release()
                self._cap = None
        except Exception as e:
            log.warning("sim camera: no webcam (%s); returning synthetic frames", e)
            self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Return an (H, W, 3) uint8 BGR frame."""
        self._ensure_cam()
        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok:
                return frame
        # synthetic gradient frame with a moving bar (proves the vision path works)
        h, w = 240, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)
        t = time.monotonic() - self._t0
        x = int((math.sin(t) * 0.5 + 0.5) * (w - 40))
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img[:, x:x + 40, 2] = 255
        return img

    def get_frame_jpeg(self) -> Optional[bytes]:
        frame = self.get_frame()
        if frame is None:
            return None
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                return buf.tobytes()
        except Exception:
            pass
        try:  # PIL fallback (frame is BGR -> RGB)
            import io

            from PIL import Image

            bio = io.BytesIO()
            Image.fromarray(frame[:, :, ::-1]).save(bio, format="JPEG", quality=80)
            return bio.getvalue()
        except Exception:
            return None

    def close(self) -> None:
        self.stop_recording()
        self.stop_playing()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class FakeMini:
    """Mock ``ReachyMini``. Records the last commanded pose so tests/UI can inspect it."""

    def __init__(self, **_kwargs) -> None:
        self.media = _FakeMedia()
        self._head = _identity()
        self._antennas = [0.0, 0.0]
        self._body_yaw = 0.0
        self._move_cancel = threading.Event()
        self.log: List[dict] = []  # motion command history (for the web UI / debugging)

    # context manager
    def __enter__(self) -> "FakeMini":
        return self

    def __exit__(self, *exc) -> None:
        self.media.close()

    # motors / lifecycle
    def enable_motors(self, ids=None) -> None: ...
    def disable_motors(self, ids=None) -> None: ...
    def wake_up(self) -> None:
        log.info("sim: wake_up")

    def goto_sleep(self) -> None:
        log.info("sim: goto_sleep")

    # motion
    def set_target(self, head=None, antennas=None, body_yaw=None) -> None:
        if head is not None:
            self._head = np.asarray(head)
        if antennas is not None:
            self._antennas = list(antennas)
        if body_yaw is not None:
            self._body_yaw = float(body_yaw)

    def goto_target(self, head=None, antennas=None, duration=0.5, method="minjerk", body_yaw=0.0) -> None:
        self.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
        # non-blocking-ish in sim: don't actually sleep the whole duration
        time.sleep(min(0.02, max(0.0, duration)))

    def look_at_world(self, x, y, z, duration=1.0, perform_movement=True):
        self.log.append({"look_at_world": [x, y, z], "duration": duration})
        return self._head

    def get_current_head_pose(self):
        return self._head

    def get_current_joint_positions(self):
        return [0.0] * 7, list(self._antennas)

    def get_present_antenna_joint_positions(self):
        return list(self._antennas)

    def cancel_move(self) -> None:
        self._move_cancel.set()

    def play_move(self, move, play_frequency=100.0, initial_goto_duration=0.0, sound=True) -> None:
        """Play a recorded move by sampling ``move.evaluate(t)`` at ``play_frequency``."""
        self._move_cancel.clear()
        dur = float(getattr(move, "duration", 1.0) or 1.0)
        dt = 1.0 / max(1.0, play_frequency)
        t = 0.0
        while t < dur and not self._move_cancel.is_set():
            try:
                head, antennas, body_yaw = move.evaluate(t)
                self.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            except Exception:
                pass
            time.sleep(dt)
            t += dt


def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)
