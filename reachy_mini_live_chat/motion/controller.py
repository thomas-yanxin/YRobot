"""The one thread that owns ``set_target`` (per the SDK control-loop guidance).

Every other thread only enqueues :class:`MotionIntent`s; this loop is the single writer.
Each ~100 Hz tick it computes a target pose and commands it, blending:

* **mood** motion for the conversation state (idle breathing, attentive listen lean,
  thoughtful tilt, speaking sway),
* **DOA** head-yaw toward the current speaker,
* a currently-playing **emotion** move, sampled frame-by-frame so a barge-in ``cancel``
  stops it instantly (procedural fallback when the library isn't available).

All output goes through :mod:`.safety` clamps, so nothing ever exceeds the joint limits.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Optional, Tuple

import numpy as np

from ..bus import Bus, ConvState, MotionIntent
from ..config import Config
from . import safety
from .doa import DoaTracker
from .emotions import EmotionLibrary

log = logging.getLogger("live_chat.motion")


def _head_pose(roll: float, pitch: float, yaw: float, z: float = 0.0) -> np.ndarray:
    """Build a 4x4 head pose from rpy degrees (matches reachy create_head_pose xyz order)."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = safety.rpy_to_matrix(roll, pitch, yaw, degrees=True)
    pose[2, 3] = z
    return pose


class MotionController:
    def __init__(self, mini, cfg: Config, bus: Bus) -> None:
        self.mini = mini
        self.cfg = cfg
        self.bus = bus
        self.emotions = EmotionLibrary(cfg)
        self._doa = DoaTracker()
        self._doa_yaw = 0.0
        self._doa_last_poll = 0.0

        self._active_move = None      # (move_or_None, name, t0, duration, intensity)
        self._thread: Optional[threading.Thread] = None
        self._t0 = time.monotonic()

        # Output smoothing: EMA the commanded antennas (and head rpy) so state changes
        # and any control-loop jitter don't show up as jerky steps. Time constant is a
        # few tens of ms at 100 Hz — smooth to the eye, still responsive for gestures.
        self._ant_cmd = [0.0, 0.0]
        self._head_rpy_cmd = None     # (roll, pitch, yaw) degrees, smoothed
        self._ant_alpha = 0.25
        self._head_alpha = 0.35
        # Speech envelope (EMA of bus.speech_level): the talking wobble follows the
        # actual voice instead of a fixed metronome sine — silence between sentences
        # stills the head, stressed words move it. This is what makes it look alive.
        self._speech_lvl = 0.0
        # Random per-session phases for the speaking oscillators (the official head
        # wobbler does the same) so the axes never lock into one repeating figure.
        self._ph = [random.uniform(0.0, 2.0 * math.pi) for _ in range(5)]
        # Idle "glance" gesture: every few seconds pick a nearby point and look at it
        # briefly, like someone waiting. (t_next, t_end, yaw_deg, pitch_deg)
        self._glance_next = self._t0 + random.uniform(4.0, 9.0)
        self._glance_end = 0.0
        self._glance_yaw = 0.0
        self._glance_pitch = 0.0
        # Listening backchannel: an occasional small nod while the user talks.
        self._nod_next = 0.0
        self._nod_end = 0.0
        # Long-idle life signs: after ~3 min with no conversation, occasionally play a
        # subtle emotion locally (the official app runs the same LLM-free idle policy).
        self._last_active = self._t0
        self._idle_emo_next: Optional[float] = None
        # effective-rate telemetry
        self._rate_t0 = self._t0
        self._rate_n = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if not self.cfg.enable_motion:
            log.info("motion disabled")
            return
        try:
            self.mini.enable_motors()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, name="motion", daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread:
            self._thread.join(timeout=1.0)

    # -- intents ------------------------------------------------------------
    def _drain_intents(self) -> None:
        while True:
            try:
                intent: MotionIntent = self.bus.motion_intents.get_nowait()
            except Exception:
                return
            if intent.kind == "cancel":
                self._active_move = None
            elif intent.kind == "emotion" and intent.emotion:
                self._begin_emotion(intent.emotion, intent.intensity)
            elif intent.kind == "doa" and intent.angle is not None:
                y = self._doa.update(intent.angle)
                if y is not None:
                    self._doa_yaw = y

    def _begin_emotion(self, name: str, intensity: float) -> None:
        move = self.emotions.get(name)
        dur = float(getattr(move, "duration", 0.0) or 0.0) if move is not None else _PROC_DUR.get(name, 0.9)
        self._active_move = (move, name, time.monotonic(), max(0.3, dur), intensity)
        self.bus.emit("emotion", {"name": name})

    # -- main loop ----------------------------------------------------------
    def _loop(self) -> None:
        dt = 1.0 / max(20, self.cfg.control_hz)
        while not self.bus.stop_event.is_set():
            tick = time.monotonic()
            try:
                self._drain_intents()
                self._poll_doa(tick)
                head, antennas, body_yaw = self._compute(tick)
                head = self._smooth_head(safety.clamp_head_pose(head))
                antennas = self._smooth_antennas(safety.clamp_antennas(antennas))
                if body_yaw is not None:
                    body_yaw = safety.clamp_body_yaw(body_yaw, safety.head_yaw_deg(head) * math.pi / 180.0)
                self.mini.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
                self._tick_rate(tick)
            except Exception as e:
                log.debug("motion tick error: %s", e)
            elapsed = time.monotonic() - tick
            time.sleep(max(0.0, dt - elapsed))

    # -- output smoothing ---------------------------------------------------
    def _smooth_antennas(self, antennas) -> list:
        a = self._ant_alpha
        for i in range(min(2, len(antennas))):
            self._ant_cmd[i] += a * (float(antennas[i]) - self._ant_cmd[i])
        return list(self._ant_cmd)

    def _smooth_head(self, head: np.ndarray) -> np.ndarray:
        """EMA the head in rpy space (matrix EMA isn't a valid rotation)."""
        roll, pitch, yaw = safety.matrix_to_rpy(head[:3, :3], degrees=True)
        if self._head_rpy_cmd is None:
            self._head_rpy_cmd = [roll, pitch, yaw]
        else:
            a = self._head_alpha
            cur = self._head_rpy_cmd
            cur[0] += a * (roll - cur[0])
            cur[1] += a * (pitch - cur[1])
            # yaw wraps: smooth the shortest angular delta
            dyaw = (yaw - cur[2] + 180.0) % 360.0 - 180.0
            cur[2] += a * dyaw
        out = np.array(head, dtype=np.float64, copy=True)
        out[:3, :3] = safety.rpy_to_matrix(*self._head_rpy_cmd, degrees=True)
        return out

    def _tick_rate(self, tick: float) -> None:
        self._rate_n += 1
        if tick - self._rate_t0 >= 5.0:
            hz = self._rate_n / (tick - self._rate_t0)
            if hz < 0.8 * max(20, self.cfg.control_hz):
                log.info("motion: effective control rate %.0f Hz (target %d Hz)", hz, self.cfg.control_hz)
            else:
                log.debug("motion: effective control rate %.0f Hz", hz)
            self._rate_t0 = tick
            self._rate_n = 0

    def _poll_doa(self, now: float) -> None:
        # The audio capture loop owns the get_DoA() USB reads (~20 Hz) and caches
        # the voiced angle on the bus — consume that here instead of hitting the
        # USB device from a second thread.
        if not (self.cfg.enable_doa and self.bus.user_speaking.is_set()):
            return
        if now - self._doa_last_poll < 0.1:
            return
        self._doa_last_poll = now
        angle = self.bus.doa_angle
        if angle is not None:
            y = self._doa.update(angle)
            if y is not None:
                self._doa_yaw = y

    # -- pose synthesis -----------------------------------------------------
    def _compute(self, now: float) -> Tuple[np.ndarray, list, Optional[float]]:
        # active emotion move overrides mood motion (still clamped downstream)
        if self._active_move is not None:
            move, name, t0, dur, intensity = self._active_move
            t = now - t0
            if t >= dur:
                self._active_move = None
            else:
                return self._emotion_pose(move, name, t, dur, intensity)

        # Track the played voice envelope: consume samples whose play time has arrived
        # (the playback thread stamps them with their due time — see AudioEngine), then
        # smooth with fast attack / slow release, like the official wobbler DSP.
        env = self.bus.speech_env
        while env:
            try:
                due, lvl = env[0]
            except IndexError:  # raced the playback thread's clear()
                break
            if due > now:
                break
            env.popleft()
            self.bus.speech_level = lvl
        target_lvl = self.bus.speech_level if self.bus.robot_speaking.is_set() else 0.0
        a = 0.5 if target_lvl > self._speech_lvl else 0.12
        self._speech_lvl += a * (target_lvl - self._speech_lvl)

        state = self.bus.state
        t = now - self._t0
        if state == ConvState.LISTENING:
            head, ant = self._listening(t, now)
        elif state == ConvState.THINKING:
            head, ant = self._thinking(t)
        elif state in (ConvState.SPEAKING, ConvState.INTERRUPTED) or self.bus.robot_speaking.is_set():
            head, ant = self._speaking(t)
        else:
            head, ant = self._idle(t, now)

        self._maybe_idle_emotion(state, now)

        # Blend DOA yaw into the head. Hold the orientation while EITHER side is talking
        # — the official app anchors the head toward the person for the whole reply;
        # decaying mid-reply made the robot slowly turn away from the user it was
        # answering. Only drift back to center once the conversation goes quiet.
        yaw = math.degrees(self._doa_yaw)
        if not (self.bus.user_speaking.is_set() or self.bus.robot_speaking.is_set()):
            self._doa_yaw *= 0.96
        roll, pitch, base_yaw, z = head
        head_pose = _head_pose(roll, pitch, base_yaw + yaw, z)
        body_yaw = self._doa_yaw * 0.5 if abs(yaw) > 45 else None
        return head_pose, ant, body_yaw

    def _maybe_idle_emotion(self, state: ConvState, now: float) -> None:
        """After ~3 min of no conversation, occasionally play a subtle gesture."""
        busy = (
            state != ConvState.IDLE
            or self.bus.robot_speaking.is_set()
            or self.bus.user_speaking.is_set()
            or self._active_move is not None
        )
        if busy:
            self._last_active = now
            self._idle_emo_next = None
            return
        if now - self._last_active < 180.0:
            return
        if self._idle_emo_next is None:
            self._idle_emo_next = now + random.uniform(0.0, 30.0)
            return
        if now >= self._idle_emo_next:
            self._idle_emo_next = now + random.uniform(120.0, 240.0)
            self._begin_emotion(random.choice(_IDLE_EMOTIONS), 0.7)

    # mood generators return ((roll,pitch,yaw)deg + z(m), [ant_r,ant_l]rad)
    def _idle(self, t: float, now: float):
        pitch = 2.0 * math.sin(2 * math.pi * 0.15 * t)
        roll = 1.0 * math.sin(2 * math.pi * 0.11 * t + 1.0)
        yaw = 4.0 * math.sin(2 * math.pi * 0.05 * t)
        z = 0.005 * math.sin(2 * math.pi * 0.1 * t)             # official breathing: 5 mm @ 0.1 Hz
        # Occasional glance at a nearby point — a robot that never shifts its gaze
        # reads as frozen. Eased in/out; the EMA smoothing rounds the edges further.
        if now >= self._glance_next:
            self._glance_end = now + random.uniform(1.2, 2.6)
            self._glance_next = self._glance_end + random.uniform(4.0, 10.0)
            self._glance_yaw = random.uniform(-28.0, 28.0)
            self._glance_pitch = random.uniform(-10.0, 6.0)
        if now < self._glance_end:
            yaw += self._glance_yaw
            pitch += self._glance_pitch
        ant = [math.radians(10 + 8 * math.sin(2 * math.pi * 0.2 * t)),
               math.radians(10 + 8 * math.sin(2 * math.pi * 0.2 * t + math.pi))]
        return (roll, pitch, yaw, z), ant

    def _listening(self, t: float, now: float):
        pitch = -6.0 + 1.5 * math.sin(2 * math.pi * 0.4 * t)    # slight forward/attentive
        roll = 2.0 * math.sin(2 * math.pi * 0.3 * t)
        # Backchannel: a small nod every few seconds while the user talks ("I'm
        # with you") — listeners that hold perfectly still look switched off.
        if now >= self._nod_next:
            self._nod_end = now + 0.7
            self._nod_next = self._nod_end + random.uniform(2.5, 5.0)
        if now < self._nod_end:
            p = 1.0 - (self._nod_end - now) / 0.7
            pitch += 6.0 * math.sin(math.pi * p) * math.sin(2 * math.pi * 2.0 * p)
        ant = [math.radians(25), math.radians(25)]              # antennas perked up
        return (roll, pitch, 0.0, 0.0), ant

    def _thinking(self, t: float):
        pitch = -10.0                                           # look up, pondering
        roll = 8.0 * math.sin(2 * math.pi * 0.5 * t)
        yaw = 8.0
        ant = [math.radians(-15), math.radians(15)]
        return (roll, pitch, yaw, 0.0), ant

    def _speaking(self, t: float):
        # Voice-synced wobble, matching the official head-wobbler oscillator bank:
        # slow independent sinusoids per axis (random per-session phases), amplitude
        # scaled by the played speech envelope — the head moves WITH the words and
        # stills in the silence between sentences. A small base keeps a pulse of life.
        lvl = self._speech_lvl
        w = 2 * math.pi
        pitch = (0.6 + 4.5 * lvl) * math.sin(w * 2.2 * t + self._ph[0])
        yaw = (0.8 + 7.5 * lvl) * math.sin(w * 0.6 * t + self._ph[1])
        roll = (0.4 + 2.25 * lvl) * math.sin(w * 1.3 * t + self._ph[2])
        z = 0.00225 * lvl * math.sin(w * 0.25 * t + self._ph[3])
        wig = math.radians((2.0 + 10.0 * lvl) * math.sin(w * 2.0 * t + self._ph[4]))
        ant = [wig, -wig]
        return (roll, pitch, yaw, z), ant

    def _emotion_pose(self, move, name: str, t: float, dur: float, intensity: float):
        if move is not None:
            try:
                head, antennas, body_yaw = move.evaluate(t)
                if head is not None:
                    head = safety.clamp_head_pose(np.asarray(head))
                    ant = list(antennas) if antennas is not None else [0.0, 0.0]
                    return head, ant, (float(body_yaw) if body_yaw is not None else None)
            except Exception:
                pass
        # procedural fallback
        (roll, pitch, yaw), ant = _procedural(name, t, dur, intensity)
        return _head_pose(roll, pitch, yaw), ant, None


# Subtle gestures suitable for unprompted long-idle "signs of life".
_IDLE_EMOTIONS = ["curious1", "thoughtful1", "attentive1", "calming1"]

# ---- procedural gestures (used when the emotion library isn't available) ----
_PROC_DUR = {
    "yes1": 1.0, "no1": 1.1, "curious1": 1.2, "confused1": 1.2, "surprised1": 0.9,
    "cheerful1": 1.2, "laughing1": 1.4, "welcoming1": 1.3, "thoughtful1": 1.4,
}


def _procedural(name: str, t: float, dur: float, k: float):
    p = t / dur  # 0..1
    env = math.sin(math.pi * min(1.0, p))  # ease in/out
    if name == "yes1":  # nod
        return (0.0, 22 * env * math.sin(2 * math.pi * 2 * t) * k, 0.0), [math.radians(30 * env), math.radians(30 * env)]
    if name == "no1":   # shake
        return (0.0, 0.0, 30 * env * math.sin(2 * math.pi * 2 * t) * k), [math.radians(-20 * env), math.radians(20 * env)]
    if name in ("curious1", "confused1", "thoughtful1", "uncertain1"):  # tilt
        return (25 * env * k, -6 * env, 10 * env * math.sin(2 * math.pi * 0.8 * t)), [math.radians(-15 * env), math.radians(20 * env)]
    if name in ("surprised1",):  # quick pull-back up
        return (0.0, -25 * env * k, 0.0), [math.radians(35 * env), math.radians(35 * env)]
    if name in ("laughing1", "cheerful1", "enthusiastic1"):  # bouncy
        return (6 * env * math.sin(2 * math.pi * 3 * t) * k, 10 * env * math.sin(2 * math.pi * 3 * t) * k, 0.0), \
               [math.radians(30 * env * math.sin(2 * math.pi * 4 * t)), math.radians(-30 * env * math.sin(2 * math.pi * 4 * t))]
    if name in ("sad1", "sad2"):  # droop
        return (0.0, 18 * env * k, 8 * env), [math.radians(-25 * env), math.radians(-25 * env)]
    if name in ("welcoming1", "welcoming2", "grateful1"):  # warm sway
        return (10 * env * math.sin(2 * math.pi * 1.2 * t) * k, -4 * env, 12 * env * math.sin(2 * math.pi * 0.8 * t)), \
               [math.radians(25 * env), math.radians(25 * env)]
    # default: gentle acknowledging nod
    return (0.0, 12 * env * math.sin(2 * math.pi * 1.5 * t) * k, 0.0), [math.radians(15 * env), math.radians(15 * env)]
