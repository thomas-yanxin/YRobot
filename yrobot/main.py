"""Application wiring: Reachy Mini app, session lifecycle and CLI.

Thread map (all communication is immutable data + atomic flags):

    main loop      microphone → VAD → turn controller → bounded audio queue
    yrobot-uplink  audio queue + latest JPEG → websocket (may block safely)
    yrobot-camera  camera → resize/JPEG → replaceable latest-frame slot
    yrobot-recv    gateway deltas → gate check → speaker queue / captions
    yrobot-speaker paced, interruptible playback (owns the audio pipeline)
    yrobot-motion  50 Hz choreographer (owns the robot pose)
    yrobot-doa     12 Hz sound compass → gaze targets
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini.reachy_mini import ReachyMini

try:  # optional: shrinks camera frames before upload (weak networks)
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from yrobot.audio import (
    Microphone,
    Speaker,
    UplinkGain,
    VoiceDetector,
    apply_audio_startup_config,
)
from yrobot.config import Settings
from yrobot.motion import IDLE, LISTEN, SPEAK, Choreographer, SoundCompass, head_yaw_of
from yrobot.realtime import Delta, RealtimeClient, ThinkFilter
from yrobot.turn import TurnGate

logger = logging.getLogger(__name__)

FRAME_S = 0.02
AUDIO_PIPELINE_WARMUP_S = 1.0
ACTIVE_WINDOW_S = 10.0  # camera stays at 1 fps this long after the user spoke
# kv-cache burn rates measured on the live gateway (tokens/second, per frame).
# "Active chat ~85 tok/s" was measured WITH 1 fps vision; frames are counted
# separately here, so busy audio-only burn is ~28. Estimating 85 rotated
# sessions every ~85 s and wiped the model's memory mid-conversation
# (hardware log 2026-07-24).
KV_PER_S_IDLE, KV_PER_S_BUSY, KV_PER_FRAME = 13.0, 28.0, 64.0
# The model reads frames at max_slice_nums=1 (~448 px): a full-resolution
# JPEG is pure uplink waste and stalls audio on weak wifi.
FRAME_MAX_DIM = 448
FRAME_JPEG_QUALITY = 80


def shrink_jpeg(bgr_frame: np.ndarray | None) -> bytes | None:
    """Encode a camera frame at the model's native vision scale."""
    if bgr_frame is None or cv2 is None:
        return None
    height, width = bgr_frame.shape[:2]
    scale = FRAME_MAX_DIM / max(height, width)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        bgr_frame = cv2.resize(bgr_frame, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
    return encoded.tobytes() if ok else None


@dataclass(frozen=True)
class UplinkPacket:
    """One captured audio unit waiting for the network sender."""

    audio: np.ndarray
    force_listen: bool
    captured_at: float
    input_id: str


class LatestCamera(threading.Thread):
    """Capture and encode video away from the realtime audio path.

    Only the newest JPEG is retained: video is context, not a lossless stream,
    so a slow network must never build a stale frame backlog.
    """

    def __init__(
        self,
        media,
        active: Callable[[float], bool],
        robot_audible: Callable[[float], bool],
        active_period_s: float,
        idle_period_s: float,
    ) -> None:
        super().__init__(name="yrobot-camera", daemon=True)
        self._media = media
        self._active = active
        self._robot_audible = robot_audible
        self._active_period_s = active_period_s
        self._idle_period_s = idle_period_s
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[int, bytes] | None = None
        self._sequence = 0
        self._taken_sequence = 0

    def close(self) -> None:
        self._halt.set()

    def take_latest(self) -> bytes | None:
        """Return each encoded frame at most once."""
        with self._lock:
            if self._latest is None or self._latest[0] == self._taken_sequence:
                return None
            self._taken_sequence, jpeg = self._latest
            return jpeg

    def run(self) -> None:
        next_capture = 0.0
        while not self._halt.wait(0.02):
            now = time.monotonic()
            if now < next_capture or self._robot_audible(now):
                continue
            period = self._active_period_s if self._active(now) else self._idle_period_s
            next_capture = now + period
            try:
                if cv2 is not None:
                    jpeg = shrink_jpeg(self._media.get_frame())
                else:
                    jpeg = self._media.get_frame_jpeg()
            except Exception as exc:  # noqa: BLE001
                logger.debug("camera capture skipped: %s", exc)
                continue
            if not jpeg:
                continue
            with self._lock:
                self._sequence += 1
                self._latest = (self._sequence, jpeg)


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
        self._turn_lock = threading.Lock()
        self._agc = UplinkGain()
        self._choreo = Choreographer(mini)
        self._compass = SoundCompass(
            mini.media,
            current_head_yaw=self._current_head_yaw,
            user_active=self._confirmed_user_active,
            on_target=self._choreo.set_gaze_target,
        )
        self._captions = ThinkFilter()
        self._session_dead = threading.Event()
        self._last_voice_at = -1e9
        self._last_user_onset_at = -1e9
        self._confirmed_voice_until = -1e9
        self._server_kv: float | None = None
        self._video_kv_est = 0.0
        self._last_logged_audio_onset_at = -1e9
        self._input_sequence = 0

    def run(self) -> None:
        self._mini.media.start_recording()
        self._mini.media.start_playing()
        # Pollen's reference waits for the GStreamer pipelines to materialize
        # before writing/reading back XVF controls.
        self._stop.wait(AUDIO_PIPELINE_WARMUP_S)
        apply_audio_startup_config(self._mini.media)
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
            self._compass.join(timeout=2)
            self._choreo.join(timeout=2)
            self._speaker.join(timeout=2)
            self._mini.media.stop_recording()

    # -- session ------------------------------------------------------------

    def _one_session(self) -> None:
        self._session_dead.clear()
        self._captions = ThinkFilter()
        self._server_kv = None
        self._video_kv_est = 0.0
        self._last_logged_audio_onset_at = -1e9
        self._input_sequence = 0
        with self._turn_lock:
            self._gate = TurnGate()
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
            # A transport/session boundary is also a playback boundary.
            # Never let buffered deltas from a dead session leak into the
            # reconnecting one.
            with self._turn_lock:
                final_epoch = self._speaker.interrupt()
            client.close(reason="rollover" if not self._stop.is_set() else "user_stop")
            self._speaker.wait_flushed(final_epoch, timeout=0.5)

    def _uplink_loop(self, client: RealtimeClient) -> None:
        chunk_frames = self._s.chunk_ms // 20
        frames: list[np.ndarray] = []
        packets: queue.Queue[UplinkPacket] = queue.Queue(maxsize=4)
        sender_halt = threading.Event()
        camera = (
            LatestCamera(
                self._mini.media,
                active=lambda now: now - self._last_voice_at < ACTIVE_WINDOW_S,
                robot_audible=self._speaker.playing,
                active_period_s=self._s.frame_period_active_s,
                idle_period_s=self._s.frame_period_idle_s,
            )
            if self._s.send_video
            else None
        )
        sender = threading.Thread(
            target=self._send_loop,
            args=(client, packets, sender_halt, camera),
            name="yrobot-uplink",
            daemon=True,
        )
        if camera is not None:
            camera.start()
        sender.start()
        while self._mic.read_frames():  # drop audio captured during session setup
            pass
        t0 = time.monotonic()
        kv_est = 0.0
        last_poll = time.monotonic()
        last_gap_log = -1e9
        try:
            while not self._stop.is_set() and not self._session_dead.is_set():
                poll_at = time.monotonic()
                capture_gap = poll_at - last_poll
                last_poll = poll_at
                if capture_gap > 0.06 and poll_at - last_gap_log > 2.0:
                    logger.warning("microphone loop gap %.0f ms", capture_gap * 1000)
                    last_gap_log = poll_at
                frames.extend(self._process_mic())
                now = time.monotonic()
                with self._turn_lock:
                    timed_out = self._gate.timed_out(now)
                if timed_out:
                    logger.error("barge-in boundary timed out; reconnecting instead of replaying")
                    return
                # MiniCPM-o 4.5 only advances on complete one-second units.
                # A partial force packet can produce a synthetic listen while
                # never executing the force override, so never flush early.
                if len(frames) < chunk_frames:
                    continue
                raw_chunk = np.concatenate(frames[:chunk_frames])
                del frames[:chunk_frames]
                chunk = self._agc.process(
                    raw_chunk,
                    playback_active=self._speaker.playing(now),
                    confirmed_user_voice=self._confirmed_user_active(now),
                )
                self._input_sequence += 1
                input_id = f"input_{self._input_sequence:08d}"
                with self._turn_lock:
                    force_listen = self._gate.chunk_force_listen(now)
                self._enqueue_packet(
                    packets,
                    UplinkPacket(
                        chunk,
                        force_listen=force_listen,
                        captured_at=now,
                        input_id=input_id,
                    ),
                    # Prioritize the first causally valid force unit over stale
                    # queued silence/echo on a congested wireless uplink.
                    flush_backlog=force_listen,
                )
                busy = self._speaker.audible(now) or self._confirmed_user_active(now)
                kv_est += (KV_PER_S_BUSY if busy else KV_PER_S_IDLE) * len(raw_chunk) / 16_000
                rotation_kv = self._server_kv
                if rotation_kv is None:
                    rotation_kv = kv_est + self._video_kv_est
                if self._should_rotate(now - t0, rotation_kv, now):
                    logger.info(
                        "rotating session (%.0f s, %.0f kv tokens%s)",
                        now - t0,
                        rotation_kv,
                        " server" if self._server_kv is not None else " estimated",
                    )
                    return
        finally:
            sender_halt.set()
            if camera is not None:
                camera.close()
                camera.join(timeout=2)
            sender.join(timeout=2)

    def _send_loop(
        self,
        client: RealtimeClient,
        packets: queue.Queue[UplinkPacket],
        halt: threading.Event,
        camera: LatestCamera | None,
    ) -> None:
        """Serialize network writes without ever blocking mic capture."""
        while not halt.is_set() and not self._session_dead.is_set():
            try:
                packet = packets.get(timeout=0.05)
            except queue.Empty:
                continue
            jpeg = camera.take_latest() if camera is not None else None
            started = time.monotonic()
            queue_ms = (started - packet.captured_at) * 1000
            try:
                if packet.force_listen:
                    with self._turn_lock:
                        self._gate.force_sent(packet.input_id, started)
                client.send_chunk(
                    packet.audio,
                    jpeg,
                    packet.force_listen,
                    packet.input_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("uplink ended: %s", exc)
                self._session_dead.set()
                return
            send_ms = (time.monotonic() - started) * 1000
            if packet.force_listen:
                logger.info(
                    "force_listen sent: %.0f ms queue age, %.0f ms websocket",
                    queue_ms,
                    send_ms,
                )
            if jpeg is not None:
                self._video_kv_est += KV_PER_FRAME
            if send_ms > 300 or queue_ms > 150:
                logger.warning(
                    "slow uplink: websocket %.0f ms, queue age %.0f ms, depth %d, video=%s",
                    send_ms,
                    queue_ms,
                    packets.qsize(),
                    jpeg is not None,
                )

    @staticmethod
    def _enqueue_packet(
        packets: queue.Queue[UplinkPacket],
        packet: UplinkPacket,
        *,
        flush_backlog: bool = False,
    ) -> None:
        """Bound realtime backlog; retain the freshest audio under overload."""
        if flush_backlog:
            discarded = 0
            while True:
                try:
                    packets.get_nowait()
                    discarded += 1
                except queue.Empty:
                    break
            if discarded:
                logger.info("barge-in dropped %d stale queued uplink chunks", discarded)
        try:
            packets.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            packets.get_nowait()
        except queue.Empty:
            pass
        packets.put_nowait(packet)
        logger.error("uplink backlog full: dropped oldest audio chunk")

    def _process_mic(self) -> list[np.ndarray]:
        """Read mic frames; run VAD, barge-in and posture per 20 ms frame.

        The XVF3800 profile conditions double-talk before WebRTC VAD. Once
        VAD confirms 100 ms of speech over a live robot turn, interruption is
        destructive and immediate: advance the playback epoch, request a
        device flush, latch output suppression and keep force_listen active
        on complete model units. Old audio is never resumed after a qualified
        human onset.
        """
        out = self._mic.read_frames()
        # A device read may return several frames. Approximate their capture
        # times instead of assigning the oldest frame a timestamp in the
        # future after a delayed device read.
        now = time.monotonic() - FRAME_S * max(0, len(out) - 1)
        for frame in out:
            robot_sounding = self._speaker.sounding(now)
            robot_turn_live = self._speaker.audible(now)
            voiced = self._detector.process(frame, now, floor_frozen=robot_sounding)
            if (
                voiced
                and self._detector.streak >= 5
                and robot_turn_live
                and not self._gate_latched()
            ):
                self._begin_barge(now)
            # Once a barge is latched, every voiced frame keeps force sticky.
            # While the robot is silent, ordinary user speech still drives
            # DoA/camera activity without creating an interruption.
            if voiced and (not robot_sounding or self._gate_latched()):
                self._mark_user_voice(now)
                with self._turn_lock:
                    self._gate.user_frame(True, False, now)
            now += FRAME_S
        now = time.monotonic()
        if self._speaker.playing(now):
            self._choreo.set_mode(SPEAK)
        elif self._confirmed_user_active(now) or self._gate_latched():
            self._choreo.set_mode(LISTEN)
        else:
            self._choreo.set_mode(IDLE)
        return out

    def _begin_barge(self, now: float) -> None:
        """Atomically suppress output and hard-stop the interrupted local turn."""
        with self._turn_lock:
            started = self._gate.user_frame(True, True, now)
            if started:
                self._captions = ThinkFilter()
                epoch = self._speaker.interrupt()
        if started:
            self._mark_user_voice(now)
            logger.info(
                "barge-in: local playback discarded at epoch %d (mic %.1f dB); "
                "force latched for next complete unit",
                epoch,
                self._detector.last_db,
            )

    def _mark_user_voice(self, now: float) -> None:
        if now - self._last_voice_at >= VoiceDetector.HANGOVER_S:
            self._last_user_onset_at = now
        self._last_voice_at = now
        self._confirmed_voice_until = max(
            self._confirmed_voice_until,
            now + VoiceDetector.HANGOVER_S,
        )

    def _confirmed_user_active(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now < self._confirmed_voice_until

    def _gate_latched(self) -> bool:
        with self._turn_lock:
            return self._gate.latched

    def _current_head_yaw(self) -> float:
        """Read the daemon's cached physical pose; fall back during startup."""
        try:
            return head_yaw_of(np.asarray(self._mini.get_current_head_pose()))
        except Exception:  # noqa: BLE001
            return self._choreo.current_yaw()

    def _should_rotate(self, elapsed: float, kv_est: float, now: float) -> bool:
        over = elapsed > self._s.session_budget_s or kv_est > self._s.kv_budget_tokens
        if not over:
            return False
        quiet = not self._gate_latched() and not self._speaker.audible(
            now
        ) and not self._confirmed_user_active(now)
        return quiet or elapsed > self._s.session_budget_s + 30.0

    # -- gateway callbacks (yrobot-recv thread) -------------------------------

    def _on_delta(self, delta: Delta) -> None:
        now = delta.received_at
        kv = delta.metrics.get("kv_cache_length")
        if isinstance(kv, int | float):
            self._server_kv = float(kv)
        if delta.kind == "listen":
            with self._turn_lock:
                was_latched = self._gate.latched
                acknowledged = self._gate.model_listen(now, delta.input_id)
                if not self._gate.latched:
                    self._speaker.utterance_end()
            if acknowledged:
                logger.info(
                    "force_listen acknowledged by %s: waiting for user turn end",
                    delta.input_id,
                )
            elif was_latched:
                logger.info(
                    "ignored unmatched listen from %s; interrupted turn remains suppressed",
                    delta.input_id or "missing input_id",
                )
        elif delta.kind == "audio":
            with self._turn_lock:
                was_latched = self._gate.latched
                allowed = self._gate.model_audio(now, delta.response_id)
                if allowed:
                    epoch = self._speaker.epoch
                    self._speaker.play(epoch, delta.audio)
            if was_latched and allowed:
                logger.info("barge-in boundary complete: accepting new model response")
            if (
                allowed
                and self._last_user_onset_at > self._last_logged_audio_onset_at
                and now - self._last_user_onset_at < 30.0
            ):
                self._last_logged_audio_onset_at = self._last_user_onset_at
                logger.info(
                    "first accepted audio %.0f ms after confirmed voice onset (response %s)",
                    (now - self._last_user_onset_at) * 1000,
                    delta.response_id or "unknown",
                )
        elif delta.kind == "text":
            with self._turn_lock:
                was_latched = self._gate.latched
                allowed = self._gate.model_text(now, delta.response_id)
                caption = self._captions.feed(delta.text).strip() if allowed else ""
            if was_latched and allowed:
                logger.info("barge-in boundary complete: accepting new model response")
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
