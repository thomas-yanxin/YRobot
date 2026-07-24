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

from yrobot.audio import EchoGuard, Microphone, Speaker, UplinkGain, VoiceDetector
from yrobot.config import Settings
from yrobot.motion import IDLE, LISTEN, SPEAK, Choreographer, SoundCompass, head_yaw_of
from yrobot.realtime import Delta, RealtimeClient, ThinkFilter
from yrobot.turn import DuckVerifier, TurnGate

logger = logging.getLogger(__name__)

FRAME_S = 0.02
ACTIVE_WINDOW_S = 10.0  # camera stays at 1 fps this long after the user spoke
# kv-cache burn rates measured on the live gateway (tokens/second, per frame).
# "Active chat ~85 tok/s" was measured WITH 1 fps vision; frames are counted
# separately here, so busy audio-only burn is ~28. Estimating 85 rotated
# sessions every ~85 s and wiped the model's memory mid-conversation
# (hardware log 2026-07-24).
KV_PER_S_IDLE, KV_PER_S_BUSY, KV_PER_FRAME = 13.0, 28.0, 64.0
# The robot is only worth interrupting when it is audibly talking; against
# near-silence the model's own duplex turn-taking handles user speech.
BARGE_PLAYOUT_MIN_DB = -40.0
STRONG_VOICE_MIN_DB = -35.0  # absolute floor for fast settle-window commits
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
        self._verifier = DuckVerifier()
        self._echo_guard = EchoGuard()
        self._agc = UplinkGain()
        self._barge_flush = False
        self._duck_pending = False
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
        self._last_block_log = -1e9
        self._voiced_run = 0
        self._server_kv: float | None = None
        self._video_kv_est = 0.0
        self._seen_audio_responses: set[str] = set()

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
        self._seen_audio_responses.clear()
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
                # A confirmed barge flushes the partial chunk immediately so
                # force_listen and the user's onset do not wait for 500 ms.
                urgent = self._barge_flush and frames
                if len(frames) < chunk_frames and not urgent:
                    continue
                self._barge_flush = False
                now = time.monotonic()
                n = min(len(frames), chunk_frames)
                raw_chunk = np.concatenate(frames[:n])
                del frames[:n]
                chunk = self._agc.process(
                    raw_chunk,
                    playback_active=self._speaker.playing(now),
                    confirmed_user_voice=self._confirmed_user_active(now),
                )
                with self._turn_lock:
                    force_listen = self._gate.chunk_force_listen(now)
                self._enqueue_packet(
                    packets,
                    UplinkPacket(chunk, force_listen=force_listen, captured_at=now),
                    flush_backlog=urgent,
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
                client.send_chunk(packet.audio, jpeg, packet.force_listen)
            except Exception as exc:  # noqa: BLE001
                logger.info("uplink ended: %s", exc)
                self._session_dead.set()
                return
            send_ms = (time.monotonic() - started) * 1000
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

        Barge-in is two-staged: a voiced frame that beats the echo guard's
        residual prediction only ducks playback; ``DuckVerifier`` then
        listens with the speaker silent and either commits the destructive
        flush or resumes the held audio (false trigger → the guard learns).
        """
        out = self._mic.read_frames()
        # A device read may return several frames. Approximate their capture
        # times instead of assigning the oldest frame a timestamp in the
        # future, which distorted settle windows after a delayed read.
        now = time.monotonic() - FRAME_S * max(0, len(out) - 1)
        for frame in out:
            self._start_verifier_after_clear(now)
            robot_sounding = self._speaker.sounding(now)
            voiced = self._detector.process(frame, now, floor_frozen=robot_sounding)
            verdict = (
                self._verifier.frame(voiced, now, strong=self._strong_voice(voiced, now))
                if self._verifier.active
                else None
            )
            if verdict == "commit":
                self._commit_barge(now, "verified")
                now += FRAME_S
                continue
            if verdict == "resume":
                self._echo_guard.penalize()
                self._speaker.release_hold()
                self._choreo.release_still()
                self._set_tracking(self._s.head_tracking_weight)
                logger.info(
                    "barge candidate was not a voice: resumed (leakage %.1f dB)",
                    self._echo_guard.offset_db,
                )
            if (
                not self._verifier.active
                and not self._duck_pending
                and self._speaker.playing(now)
                and self._speaker.playout_db(now) > BARGE_PLAYOUT_MIN_DB
            ):
                mic_db = self._detector.last_db
                playout = self._speaker.playout_db(now)
                real_voice = self._echo_guard.observe(mic_db, playout)
                self._voiced_run = self._voiced_run + 1 if voiced else 0
                # Fallback: 350 ms of sustained voice within 10 dB of the
                # predicted echo also earns a duck — the verify stage is
                # the arbiter, and hardware double-talk suppression leaves
                # real interrupting speech hovering around the gate.
                insistent = (
                    self._voiced_run >= 17 and mic_db >= playout + self._echo_guard.offset_db - 10.0
                )
                # 100 ms of continuous voice for a candidate: impulsive
                # servo knocks confirm the 60 ms VAD streak but rarely this.
                strong = self._detector.streak >= 5
                may_duck = self._verifier.ready(now) and not self._speaker.holding
                if strong and (real_voice or insistent) and may_duck:
                    if self._verifier.in_retry(now):
                        # The user is insisting right after a wrong resume:
                        # two independent candidates within seconds — commit
                        # without a second verify.
                        self._commit_barge(now, "retry")
                        logger.info(
                            "barge-in (retry after resume): committed (mic %.1f dB)", mic_db
                        )
                        now += FRAME_S
                        continue
                    self._speaker.hold()
                    self._duck_pending = True
                    # Silence every self-noise source for the verify: our
                    # motion freezes and daemon-side face tracking pauses —
                    # its servo noise confirmed false barges (hardware log
                    # 2026-07-24, 7th run: mic -19.7 dB during a hold).
                    self._choreo.hold_still(now + DuckVerifier.WINDOW_S + 0.3)
                    self._set_tracking(0.0)
                    logger.info(
                        "barge candidate (%s): duck requested (mic %.1f dB, playout %.1f dB)",
                        "level" if real_voice else "sustained",
                        mic_db,
                        playout,
                    )
                elif (
                    voiced
                    and may_duck
                    and not (real_voice or insistent)
                    and now - self._last_block_log > 2.0
                ):
                    self._last_block_log = now
                    logger.info(
                        "voice gated as echo: mic %.1f dB < %.1f dB (playout %.1f, leak %.1f)",
                        mic_db,
                        playout + self._echo_guard.offset_db,
                        playout,
                        self._echo_guard.offset_db,
                    )
            else:
                self._voiced_run = 0
            # Only post-echo speech drives DoA, camera activity and repeated
            # force-listen requests. Raw VAD may be our own loudspeaker.
            if voiced and not robot_sounding:
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

    def _start_verifier_after_clear(self, now: float) -> None:
        """Arm verification only after the device confirms it is silent."""
        if not self._duck_pending:
            return
        cleared_at = self._speaker.hold_completed_at()
        if cleared_at is None or now < cleared_at:
            return
        self._duck_pending = False
        self._verifier.start(cleared_at)
        logger.debug("barge verifier armed %.0f ms after physical clear", (now - cleared_at) * 1000)

    def _commit_barge(self, now: float, source: str) -> None:
        """Atomically invalidate the response and advance playback generation."""
        with self._turn_lock:
            self._gate.user_frame(True, True, now)
            epoch = self._speaker.interrupt()
            self._captions = ThinkFilter()
        self._duck_pending = False
        self._barge_flush = True
        self._mark_user_voice(now)
        self._choreo.release_still()
        self._set_tracking(self._s.head_tracking_weight)
        logger.info("barge-in confirmed (%s): turn discarded at epoch %d", source, epoch)

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

    def _strong_voice(self, voiced: bool, now: float) -> bool:
        """Voice whose level cannot be the in-flight echo of what we played
        (the playout envelope keeps pre-duck blocks in view for 1.6 s).

        The absolute floor matters: in a silent room a -44 dB rustle over a
        near-silent playout counted as "strong" and destroyed the model's
        reply within 60 ms (hardware log 2026-07-24, 9th run).
        """
        if not voiced or self._detector.last_db < STRONG_VOICE_MIN_DB:
            return False
        playout = self._speaker.playout_db(now)
        if playout <= -90.0:
            return True
        return self._detector.last_db >= playout + self._echo_guard.offset_db + 6.0

    def _set_tracking(self, weight: float) -> None:
        """Adjust daemon face tracking; weight 0 pauses it cheaply."""
        if self._s.head_tracking_weight <= 0:
            return
        try:
            self._mini.start_head_tracking(weight=weight)
        except Exception as exc:  # noqa: BLE001
            logger.debug("head tracking adjust failed: %s", exc)

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
                self._gate.model_listen(now)
                if not self._gate.latched:
                    self._speaker.utterance_end()
        elif delta.kind == "audio":
            with self._turn_lock:
                if self._gate.model_audio(now, delta.response_id):
                    epoch = self._speaker.epoch
                    self._speaker.play(epoch, delta.audio)
                    if delta.response_id and delta.response_id not in self._seen_audio_responses:
                        self._seen_audio_responses.add(delta.response_id)
                        if now - self._last_user_onset_at < 30.0:
                            logger.info(
                                "response %s first audio %.0f ms after confirmed voice onset",
                                delta.response_id,
                                (now - self._last_user_onset_at) * 1000,
                            )
        elif delta.kind == "text":
            with self._turn_lock:
                allowed = self._gate.model_text(now, delta.response_id)
                caption = self._captions.feed(delta.text).strip() if allowed else ""
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
