"""Reachy Mini lifecycle and realtime subsystem composition."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

from .audio import (
    AudioCaptureWorker,
    AudioUnit,
    EchoReference,
    NearEndDetector,
    PlaybackEngine,
    PlaybackPacket,
    VoiceDecision,
)
from .config import Settings
from .motion import MotionController
from .perception import CameraWorker, DoATracker, DoAWorker, LatestFrame
from .realtime import RealtimeClient
from .state import InteractionPhase, TurnCoordinator

log = logging.getLogger(__name__)


class _NearEndState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = False
        self._updated_at = 0.0

    def update(self, decision: VoiceDecision) -> None:
        with self._lock:
            self._current = decision.current_near_end
            self._updated_at = decision.timestamp

    def current(self) -> bool:
        with self._lock:
            return self._current and time.monotonic() - self._updated_at < 0.1


class _Transcript:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = -1
        self._chunks: list[str] = []

    def append(self, text: str, epoch: int) -> None:
        with self._lock:
            if epoch != self._epoch:
                self._epoch = epoch
                self._chunks.clear()
            self._chunks.append(text)

    def finish(self) -> str:
        with self._lock:
            text = "".join(self._chunks).strip()
            self._chunks.clear()
            return text

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._epoch = -1


class YRobotRuntime:
    """Own one complete app run without owning Reachy's outer connection."""

    def __init__(
        self,
        mini: Any,
        settings: Settings,
        stop_event: threading.Event,
    ) -> None:
        settings.validate()
        self.mini = mini
        self.settings = settings
        self.stop_event = stop_event
        self.coordinator = TurnCoordinator()
        self.latest_frame = LatestFrame()
        self.doa_tracker = DoATracker(hold_seconds=settings.doa_hold_seconds)
        self.near_end = _NearEndState()
        self.transcript = _Transcript()
        self._barge_count = 0

        media = mini.media
        echo = EchoReference(sample_rate=settings.input_sample_rate)
        self.playback = PlaybackEngine(
            media,
            lambda: self.coordinator.snapshot().epoch,
            echo,
            input_sample_rate=settings.output_sample_rate,
            output_sample_rate=settings.input_sample_rate,
            max_queue=settings.playback_buffers,
            preroll_ms=settings.playback_preroll_ms,
        )
        detector = NearEndDetector(
            sample_rate=settings.input_sample_rate,
            frame_ms=settings.local_frame_ms,
            vad_mode=settings.vad_mode,
            min_rms=settings.vad_min_rms,
            noise_ratio=settings.vad_noise_ratio,
            barge_attack_ms=settings.barge_attack_ms,
            barge_debounce_ms=settings.barge_debounce_ms,
            near_end_hold_ms=settings.near_end_hold_ms,
            echo_correlation=settings.echo_correlation,
            echo_reference=echo,
        )
        self.motion = MotionController(
            mini,
            phase_source=self.coordinator.snapshot,
            doa_source=self.doa_tracker.snapshot,
            hz=settings.motion_hz,
        )
        self.camera = CameraWorker(
            media,
            self.latest_frame,
            width=settings.camera_width,
            quality=settings.camera_jpeg_quality,
            fps=settings.camera_fps,
        )
        self.doa = DoAWorker(
            media,
            self.doa_tracker,
            self.near_end.current,
            head_pose=mini.get_current_head_pose,
            playback_active=self.playback.echo_guard_active,
            hz=settings.doa_hz,
        )
        self.client = RealtimeClient(
            settings,
            self.coordinator,
            latest_frame=self._latest_jpeg,
            on_audio=self._on_audio,
            on_listen=self._on_listen,
            on_text=self._on_text,
            on_session=self._on_session,
        )
        self.capture = AudioCaptureWorker(
            media,
            channel=settings.mic_channel,
            detector=detector,
            output_active=self._model_output_active,
            echo_guard_active=self.playback.echo_guard_active,
            on_unit=self._on_unit,
            on_voice=self.near_end.update,
            on_barge_in=self._on_barge_in,
            sample_rate=settings.input_sample_rate,
            frame_ms=settings.local_frame_ms,
            unit_ms=settings.input_unit_ms,
        )

    def run(self) -> None:
        if self.stop_event.is_set():
            return

        media = self.mini.media
        playing_started = False
        recording_started = False
        playback_started = False
        wobbling = False
        try:
            self.mini.enable_motors()
            try:
                self.mini.enable_wobbling()
                wobbling = True
            except Exception:
                log.warning("audio-reactive wobbling is unavailable", exc_info=True)

            media.start_playing()
            playing_started = True
            media.start_recording()
            recording_started = True
            self.playback.start()
            playback_started = True
            self.motion.start()
            if not self.motion.wait_ready(1.0):
                raise RuntimeError("motion controller did not become ready")
            self.camera.start()
            self.doa.start()
            self.capture.start()

            log.info(
                "YRobot ready: %s, input=%d ms, camera=%.1f fps, motion=%.0f Hz",
                self.settings.realtime_url,
                self.settings.input_unit_ms,
                self.settings.camera_fps,
                self.settings.motion_hz,
            )
            asyncio.run(self.client.run(self.stop_event))
        finally:
            stop_epoch = self.coordinator.stop()
            if playback_started and not self.playback.interrupt(stop_epoch):
                log.error("Reachy playback could not be flushed during shutdown")
            workers = {
                "capture": self.capture.stop(),
                "doa": self.doa.stop(),
                "camera": self.camera.stop(),
                "motion": self.motion.stop(),
            }
            if playback_started:
                workers["playback"] = self.playback.stop(flush=False)
            retry = {
                "capture": lambda: self.capture.stop(timeout=1.0),
                "doa": lambda: self.doa.stop(timeout=1.0),
                "camera": lambda: self.camera.stop(timeout=1.0),
                "motion": lambda: self.motion.stop(timeout=1.0),
                "playback": lambda: self.playback.stop(
                    flush=False,
                    timeout=1.0,
                ),
            }
            for name, stopped in workers.items():
                if not stopped and not retry[name]():
                    log.critical("%s worker did not stop before media teardown", name)
            if wobbling:
                try:
                    self.mini.disable_wobbling()
                except Exception:
                    log.warning("could not disable wobbling cleanly", exc_info=True)
            if playing_started:
                media.stop_playing()
            if recording_started:
                media.stop_recording()
            log.info(
                "YRobot stopped: barge_ins=%d realtime=%s playback=%s motion=%s",
                self._barge_count,
                self.client.metrics,
                self.playback.stats(),
                self.motion.stats(),
            )

    def _on_unit(self, unit: AudioUnit) -> None:
        self.client.submit_audio(
            unit.f32le,
            captured_at=unit.captured_at,
            sequence=unit.sequence,
        )

    def _on_audio(
        self,
        samples: np.ndarray,
        epoch: int,
        response_id: str | None,
    ) -> None:
        self.playback.enqueue(PlaybackPacket(epoch, samples, response_id))

    def _on_text(
        self,
        text: str,
        epoch: int,
        response_id: str | None,
    ) -> None:
        del response_id
        self.transcript.append(text, epoch)

    def _on_listen(self, epoch: int) -> None:
        del epoch
        self.playback.mark_response_boundary()
        text = self.transcript.finish()
        if text:
            log.info("MiniCPM-o: %s", text)

    def _on_session(self, ready: bool, epoch: int) -> None:
        self.transcript.clear()
        if not self.playback.interrupt(epoch):
            log.critical("cannot guarantee stale audio is silent; stopping YRobot")
            self.stop_event.set()
        log.info("Realtime session %s", "ready" if ready else "disconnected")

    def _on_barge_in(self, decision: VoiceDecision) -> None:
        started = time.perf_counter()
        epoch = self.coordinator.interrupt()
        if epoch is None:
            return
        self._barge_count += 1
        self.transcript.clear()
        if not self.playback.interrupt(epoch):
            log.critical("barge-in flush failed; stopping to prevent stale playback")
            self.stop_event.set()
            return
        log.info(
            "Barge-in: local silence in %.1f ms (rms=%.4f echo=%.2f)",
            (time.perf_counter() - started) * 1_000,
            decision.rms,
            decision.echo_similarity,
        )

    def _latest_jpeg(self) -> bytes | None:
        snapshot = self.latest_frame.snapshot(max_age_seconds=2.0)
        return None if snapshot is None else snapshot.jpeg

    def _model_output_active(self) -> bool:
        return (
            self.playback.output_active()
            or self.coordinator.snapshot().phase is InteractionPhase.SPEAKING
        )
