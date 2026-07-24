"""Application wiring: Reachy Mini app, session lifecycle and CLI.

Thread map (all communication is immutable data + atomic flags):

    main loop      microphone → VAD → turn gate → 500 ms uplink chunks
    yrobot-recv    gateway deltas → gate check → speaker queue / captions
    yrobot-speaker paced, interruptible playback (owns the audio pipeline)
    yrobot-motion  50 Hz choreographer (owns the robot pose)
    yrobot-doa     12 Hz sound compass → gaze targets
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from dotenv import load_dotenv
from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini.reachy_mini import ReachyMini

from yrobot.audio import Microphone, Speaker, VoiceDetector
from yrobot.config import Settings
from yrobot.motion import IDLE, LISTEN, SPEAK, Choreographer, SoundCompass
from yrobot.realtime import Delta, RealtimeClient, ThinkFilter
from yrobot.turn import TurnGate

logger = logging.getLogger(__name__)

FRAME_S = 0.02
ACTIVE_WINDOW_S = 10.0  # camera stays at 1 fps this long after the user spoke
# kv-cache burn rates measured on the live gateway (tokens/second, per frame).
KV_PER_S_IDLE, KV_PER_S_ACTIVE, KV_PER_FRAME = 13.0, 85.0, 64.0


class Conversation:
    """One full-duplex conversation across rotating gateway sessions."""

    def __init__(self, settings: Settings, mini: ReachyMini, stop: threading.Event) -> None:
        self._s = settings
        self._mini = mini
        self._stop = stop
        self._mic = Microphone(mini.media)
        self._detector = VoiceDetector(settings.vad_aggressiveness)
        self._speaker = Speaker(mini.media)
        self._gate = TurnGate()
        self._choreo = Choreographer(mini)
        self._compass = SoundCompass(
            mini.media,
            current_head_yaw=self._choreo.current_yaw,
            user_active=lambda: self._detector.active(time.monotonic()),
            on_target=self._choreo.set_gaze_target,
        )
        self._captions = ThinkFilter()
        self._session_dead = threading.Event()
        self._last_voice_at = -1e9
        self._last_frame_at = -1e9

    def run(self) -> None:
        self._mini.media.start_recording()
        self._mini.media.start_playing()
        self._mini.enable_wobbling()
        if self._s.head_tracking_weight > 0:
            self._mini.start_head_tracking(weight=self._s.head_tracking_weight)
        self._speaker.start()
        self._choreo.start()
        self._compass.start()
        try:
            while not self._stop.is_set():
                self._one_session()
                self._stop.wait(self._s.reconnect_delay_s)
        finally:
            self._compass.close()
            self._choreo.close()
            self._speaker.close()
            self._mini.media.stop_recording()

    # -- session ------------------------------------------------------------

    def _one_session(self) -> None:
        self._session_dead.clear()
        self._captions = ThinkFilter()
        client = RealtimeClient(self._s, on_delta=self._on_delta, on_closed=self._on_closed)
        try:
            client.open()
        except Exception as exc:  # noqa: BLE001 — queue/backend failures are routine
            logger.warning("session open failed: %s", exc)
            client.close()
            return
        try:
            self._uplink_loop(client)
        finally:
            client.close(reason="rollover" if not self._stop.is_set() else "user_stop")

    def _uplink_loop(self, client: RealtimeClient) -> None:
        chunk_frames = self._s.chunk_ms // 20
        frames: list[np.ndarray] = []
        while self._mic.read_frames():  # drop audio captured during session setup
            pass
        t0 = time.monotonic()
        kv_est = 0.0
        while not self._stop.is_set() and not self._session_dead.is_set():
            frames.extend(self._process_mic())
            if len(frames) < chunk_frames:
                continue
            now = time.monotonic()
            chunk = np.concatenate(frames[:chunk_frames])
            del frames[:chunk_frames]
            jpeg = self._next_frame(now)
            try:
                client.send_chunk(chunk, jpeg, self._gate.chunk_force_listen(now))
            except Exception as exc:  # noqa: BLE001
                logger.info("uplink ended: %s", exc)
                return
            busy = self._speaker.audible(now) or self._detector.active(now)
            kv_est += (KV_PER_S_ACTIVE if busy else KV_PER_S_IDLE) * self._s.chunk_ms / 1000
            kv_est += KV_PER_FRAME if jpeg is not None else 0.0
            if self._should_rotate(now - t0, kv_est, now):
                logger.info("rotating session (%.0f s, ~%.0f kv tokens)", now - t0, kv_est)
                return

    def _process_mic(self) -> list[np.ndarray]:
        """Read mic frames; run VAD, barge-in and posture per 20 ms frame."""
        out = self._mic.read_frames()
        now = time.monotonic()
        for frame in out:
            voiced = self._detector.process(frame, now)
            if voiced:
                self._last_voice_at = now
            if self._gate.user_frame(voiced, self._speaker.audible(now), now):
                self._speaker.interrupt()  # user barged in: silence within one tick
                logger.info("barge-in: playback flushed")
            now += FRAME_S
        now = time.monotonic()
        if self._speaker.audible(now):
            self._choreo.set_mode(SPEAK)
        elif self._detector.active(now) or self._gate.latched:
            self._choreo.set_mode(LISTEN)
        else:
            self._choreo.set_mode(IDLE)
        return out

    def _next_frame(self, now: float) -> bytes | None:
        """Attach a camera frame unless only the robot is talking.

        Vision costs ~64 kv tokens per frame; streaming frames while the
        robot monologues forced context rotations mid-conversation.
        """
        if not self._s.send_video or self._speaker.audible(now):
            return None
        active = now - self._last_voice_at < ACTIVE_WINDOW_S
        period = self._s.frame_period_active_s if active else self._s.frame_period_idle_s
        if now - self._last_frame_at < period:
            return None
        jpeg = self._mini.media.get_frame_jpeg()
        if jpeg:
            self._last_frame_at = now
        return jpeg

    def _should_rotate(self, elapsed: float, kv_est: float, now: float) -> bool:
        over = elapsed > self._s.session_budget_s or kv_est > self._s.kv_budget_tokens
        if not over:
            return False
        quiet = (
            not self._gate.latched
            and not self._speaker.audible(now)
            and not self._detector.active(now)
        )
        return quiet or elapsed > self._s.session_budget_s + 30.0

    # -- gateway callbacks (yrobot-recv thread) -------------------------------

    def _on_delta(self, delta: Delta) -> None:
        now = time.monotonic()
        if delta.kind == "listen":
            self._gate.model_listen(now)
            self._speaker.utterance_end()
        elif delta.kind == "audio":
            if self._gate.model_audio(now):
                self._speaker.play(self._speaker.epoch, delta.audio)
        elif delta.kind == "text":
            caption = self._captions.feed(delta.text).strip()
            if caption:
                logger.info("robot: %s", caption)

    def _on_closed(self, reason: str) -> None:
        logger.info("session closed: %s", reason)
        self._session_dead.set()


class Yrobot(ReachyMiniApp):
    """Reachy Mini app entry point (``reachy_mini_apps`` group)."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        load_dotenv()
        Conversation(Settings.from_env(), reachy_mini, stop_event).run()


def cli() -> None:
    """Run YRobot from a terminal (Ctrl-C to stop)."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s"
    )
    app = Yrobot()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    cli()
