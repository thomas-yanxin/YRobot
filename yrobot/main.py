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

try:  # optional: shrinks camera frames before upload (weak networks)
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from yrobot.audio import EchoGuard, Microphone, Speaker, UplinkGain, VoiceDetector
from yrobot.config import Settings
from yrobot.motion import IDLE, LISTEN, SPEAK, Choreographer, SoundCompass
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
        self._verifier = DuckVerifier()
        self._echo_guard = EchoGuard()
        self._agc = UplinkGain()
        self._barge_flush = False
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
        self._last_block_log = -1e9
        self._voiced_run = 0
        self._frames_paused_until = -1e9
        self._frame_pause_s = 10.0

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
            # A confirmed barge flushes the partial chunk immediately so
            # force_listen and the user's onset reach the model without
            # waiting out the chunk boundary.
            urgent = self._barge_flush and frames
            if len(frames) < chunk_frames and not urgent:
                continue
            self._barge_flush = False
            now = time.monotonic()
            n = min(len(frames), chunk_frames)
            chunk = self._agc.process(np.concatenate(frames[:n]))
            del frames[:n]
            jpeg = self._next_frame(now)
            try:
                client.send_chunk(chunk, jpeg, self._gate.chunk_force_listen(now))
            except Exception as exc:  # noqa: BLE001
                logger.info("uplink ended: %s", exc)
                return
            sent_in = time.monotonic() - now
            if sent_in > 0.3:
                # Weak network: frames serialize behind audio on the same
                # websocket and a single JPEG can stall the uplink cadence.
                # The pause backs off exponentially — a fixed 10 s retried a
                # doomed frame every 10 s for as long as the wifi stayed bad.
                self._frames_paused_until = now + self._frame_pause_s
                logger.info(
                    "slow uplink (%.2f s send): pausing camera frames %.0f s",
                    sent_in,
                    self._frame_pause_s,
                )
                self._frame_pause_s = min(self._frame_pause_s * 2, 120.0)
            elif jpeg is not None:
                self._frame_pause_s = 10.0  # a frame went through cleanly
            busy = self._speaker.audible(now) or self._detector.active(now)
            kv_est += (KV_PER_S_BUSY if busy else KV_PER_S_IDLE) * self._s.chunk_ms / 1000
            kv_est += KV_PER_FRAME if jpeg is not None else 0.0
            if self._should_rotate(now - t0, kv_est, now):
                logger.info("rotating session (%.0f s, ~%.0f kv tokens)", now - t0, kv_est)
                return

    def _process_mic(self) -> list[np.ndarray]:
        """Read mic frames; run VAD, barge-in and posture per 20 ms frame.

        Barge-in is two-staged: a voiced frame that beats the echo guard's
        residual prediction only ducks playback; ``DuckVerifier`` then
        listens with the speaker silent and either commits the destructive
        flush or resumes the held audio (false trigger → the guard learns).
        """
        out = self._mic.read_frames()
        now = time.monotonic()
        for frame in out:
            robot_audible = self._speaker.audible(now)
            voiced = self._detector.process(frame, now, floor_frozen=robot_audible)
            if voiced:
                self._last_voice_at = now
            verdict = self._verifier.frame(voiced, now, strong=self._strong_voice(voiced, now))
            if verdict == "commit":
                self._gate.user_frame(True, True, now)
                self._speaker.interrupt()
                self._barge_flush = True
                self._set_tracking(self._s.head_tracking_weight)
                logger.info("barge-in confirmed: turn discarded")
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
                and self._speaker.sounding(now)
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
                        self._gate.user_frame(True, True, now)
                        self._speaker.interrupt()
                        self._barge_flush = True
                        logger.info(
                            "barge-in (retry after resume): committed (mic %.1f dB)", mic_db
                        )
                        now += FRAME_S
                        continue
                    self._speaker.hold()  # silent within one tick, lossless
                    # Silence every self-noise source for the verify: our
                    # motion freezes and daemon-side face tracking pauses —
                    # its servo noise confirmed false barges (hardware log
                    # 2026-07-24, 7th run: mic -19.7 dB during a hold).
                    self._choreo.hold_still(now + DuckVerifier.WINDOW_S + 0.3)
                    self._set_tracking(0.0)
                    self._verifier.start(now)
                    logger.info(
                        "barge candidate (%s): ducked (mic %.1f dB, playout %.1f dB)",
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
            self._gate.user_frame(voiced, False, now)
            now += FRAME_S
        now = time.monotonic()
        if self._speaker.audible(now):
            self._choreo.set_mode(SPEAK)
        elif self._detector.active(now) or self._gate.latched:
            self._choreo.set_mode(LISTEN)
        else:
            self._choreo.set_mode(IDLE)
        return out

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

    def _next_frame(self, now: float) -> bytes | None:
        """Attach a camera frame unless only the robot is talking.

        Vision costs ~64 kv tokens per frame; streaming frames while the
        robot monologues forced context rotations mid-conversation.
        """
        if not self._s.send_video or self._speaker.audible(now):
            return None
        if now < self._frames_paused_until:
            return None
        active = now - self._last_voice_at < ACTIVE_WINDOW_S
        period = self._s.frame_period_active_s if active else self._s.frame_period_idle_s
        if now - self._last_frame_at < period:
            return None
        if cv2 is not None:
            jpeg = shrink_jpeg(self._mini.media.get_frame())
        else:
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
