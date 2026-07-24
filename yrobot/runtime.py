"""Reachy Mini lifecycle and realtime subsystem composition."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
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
from .audio_config import apply_audio_startup_config
from .config import Settings
from .motion import MotionController
from .perception import CameraWorker, DoATracker, DoAWorker, LatestFrame
from .realtime import RealtimeClient
from .state import TurnCoordinator

log = logging.getLogger(__name__)


class _VisionUplink:
    """Return only fresh camera frames at the configured model cadence."""

    def __init__(
        self,
        latest: LatestFrame,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._latest = latest
        self._interval = interval_seconds
        self._tolerance = min(0.05, interval_seconds * 0.05)
        self._clock = clock
        self._last_sequence = -1
        self._next_at = -float("inf")

    def reset(self) -> None:
        self._last_sequence = -1
        self._next_at = -float("inf")

    def next_jpeg(self) -> bytes | None:
        now = self._clock()
        snapshot = self._latest.snapshot(max_age_seconds=2.0, now=now)
        if (
            snapshot is None
            or snapshot.sequence == self._last_sequence
            or now + self._tolerance < self._next_at
        ):
            return None
        self._last_sequence = snapshot.sequence
        if self._next_at == -float("inf"):
            self._next_at = now + self._interval
        else:
            while self._next_at <= now + self._tolerance:
                self._next_at += self._interval
        return snapshot.jpeg


class _NearEndState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = False
        self._updated_at = 0.0
        self.wake_event = threading.Event()

    def update(self, decision: VoiceDecision) -> None:
        with self._lock:
            rising = decision.near_end and not self._current
            self._current = decision.near_end
            self._updated_at = decision.timestamp
        if rising:
            self.wake_event.set()

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
        self.vision_uplink = _VisionUplink(
            self.latest_frame,
            settings.vision_send_interval_seconds,
        )
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
            self._daemon_doa_url(),
            self.doa_tracker,
            self.near_end.current,
            head_pose=mini.get_current_head_pose,
            playback_active=self.playback.echo_guard_active,
            hz=settings.doa_hz,
            wake_event=self.near_end.wake_event,
        )
        self.client = RealtimeClient(
            settings,
            self.coordinator,
            latest_frame=self.vision_uplink.next_jpeg,
            on_audio=self._on_audio,
            on_listen=self._on_listen,
            on_text=self._on_text,
            on_session=self._on_session,
        )
        self.capture = AudioCaptureWorker(
            media,
            channel=settings.mic_channel,
            detector=detector,
            output_active=self.playback.output_active,
            echo_guard_active=self.playback.echo_guard_active,
            playback_gate=self.playback.gate_snapshot,
            on_unit=self._on_unit,
            on_voice=self.near_end.update,
            on_barge_in=self._on_barge_in,
            sample_rate=settings.input_sample_rate,
            frame_ms=settings.local_frame_ms,
            unit_ms=settings.input_unit_ms,
        )

    def run(self) -> None:
        if self.stop_event.is_set():
            self.doa.stop()
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
            apply_audio_startup_config(self.mini, logger=log)
            self.playback.start()
            playback_started = True
            self.motion.start()
            if not self.motion.wait_ready(1.0):
                raise RuntimeError("motion controller did not become ready")
            self.camera.start()
            self.doa.start()
            self.capture.start()

            log.info(
                "YRobot ready: %s, input=%d ms, camera=%.1f fps, vision=%.2f fps, "
                "DoA=adaptive up to %.0f Hz, motion=%.0f Hz",
                self.settings.realtime_url,
                self.settings.input_unit_ms,
                self.settings.camera_fps,
                1.0 / self.settings.vision_send_interval_seconds,
                self.settings.doa_hz,
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
        received_at: float,
    ) -> None:
        self.playback.enqueue(
            PlaybackPacket(
                epoch,
                samples,
                response_id,
                received_at=received_at,
            )
        )

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
        if ready:
            self.vision_uplink.reset()
        if not self.playback.interrupt(epoch):
            log.critical("cannot guarantee stale audio is silent; stopping YRobot")
            self.stop_event.set()
        log.info("Realtime session %s", "ready" if ready else "disconnected")

    def _on_barge_in(self, decision: VoiceDecision) -> None:
        started = time.perf_counter()
        if decision.playback_epoch is None or decision.speaker_started_age_ms is None:
            log.debug("ignored unarmed barge-in decision")
            return
        epoch = self.playback.commit_if_gate_current(
            decision.playback_generation,
            decision.playback_epoch,
            decision.playback_response_id,
            lambda playback_audible: self.coordinator.interrupt_if_epoch(
                decision.playback_epoch,
                playback_audible=playback_audible,
            ),
        )
        if epoch is None:
            log.debug(
                "ignored stale barge-in decision: epoch=%d response_id=%s generation=%d",
                decision.playback_epoch,
                decision.playback_response_id or "-",
                decision.playback_generation,
            )
            return
        self._barge_count += 1
        self.transcript.clear()
        if not self.playback.interrupt(epoch):
            log.critical("barge-in flush failed; stopping to prevent stale playback")
            self.stop_event.set()
            return
        log.info(
            "Barge-in: fresh_vad=%d ms, speaker_age=%.1f ms, "
            "player_flush=%.1f ms, stop_after_fresh_speech≈%.1f ms "
            "(rms=%.4f echo=%.2f fit=%.2f generation=%d)",
            decision.fresh_attack_ms,
            decision.speaker_started_age_ms,
            (flush_ms := (time.perf_counter() - started) * 1_000),
            decision.fresh_attack_ms + flush_ms,
            decision.rms,
            decision.echo_similarity,
            decision.echo_fit,
            decision.playback_generation,
        )

    def _daemon_doa_url(self) -> str:
        client = getattr(self.mini, "client", None)
        host = getattr(client, "host", None)
        port = getattr(client, "port", None)
        if not isinstance(host, str) or not host or not isinstance(port, int):
            raise RuntimeError("Reachy Mini daemon address is unavailable")
        return f"http://{host}:{port}/api/state/doa"
