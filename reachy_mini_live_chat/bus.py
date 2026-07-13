"""Coordination hub: conversation state machine + thread-safe queues + events.

Every stage runs in its own thread and communicates only through this ``Bus`` —
no stage touches another stage's internals. This keeps the full-duplex flow
(and especially barge-in / cancellation) easy to reason about.
"""
from __future__ import annotations

import enum
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

import numpy as np


class ConvState(enum.Enum):
    IDLE = "idle"            # nobody talking, robot at gentle rest
    LISTENING = "listening"  # user is speaking; we stream ASR
    THINKING = "thinking"    # endpoint fired; waiting on LLM first token
    SPEAKING = "speaking"    # robot is producing audio
    INTERRUPTED = "interrupted"  # barge-in: tearing down current reply


@dataclass
class Utterance:
    """A finalized user turn from ASR."""
    text: str
    lang: str                       # 'zh' | 'en'
    t_end_speech: float             # monotonic time the user stopped talking
    turn_id: int = 0


@dataclass
class MotionIntent:
    """A request for the motion controller. It owns set_target; we only ask."""
    kind: str                       # 'mood' | 'emotion' | 'doa' | 'cancel' | 'gesture'
    mood: Optional[str] = None      # for kind='mood': idle|listening|thinking|speaking
    emotion: Optional[str] = None   # for kind='emotion': emotions-library move name
    angle: Optional[float] = None   # for kind='doa': radians (SDK convention)
    intensity: float = 1.0
    meta: dict = field(default_factory=dict)


class Bus:
    def __init__(self) -> None:
        # --- queues ---------------------------------------------------------
        self.utterances: "queue.Queue[Utterance]" = queue.Queue()
        self.tts_in: "queue.Queue[Optional[str]]" = queue.Queue()      # clauses; None = flush
        self.tts_audio: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self.motion_intents: "queue.Queue[MotionIntent]" = queue.Queue()

        # --- lifecycle events ----------------------------------------------
        self.stop_event = threading.Event()      # global shutdown
        self.interrupt_event = threading.Event()  # abort current reply (barge-in)
        self.user_speaking = threading.Event()    # VAD says a human is talking now
        self.robot_speaking = threading.Event()   # TTS/audio-out is active now

        # --- shared state ---------------------------------------------------
        self._lock = threading.RLock()
        self._state = ConvState.IDLE
        self._turn_id = 0
        self.doa_angle: Optional[float] = None     # last DOA reading (radians)
        self.latest_frame_jpeg: Optional[bytes] = None  # for web UI preview

        # --- latency instrumentation ---------------------------------------
        self._marks: dict[str, float] = {}
        self.last_latency_ms: Optional[float] = None
        # end-of-speech time of the turn currently being answered; the audio-out
        # thread reads this the moment it plays the first chunk to compute e2e latency.
        self.pending_t_end: Optional[float] = None
        self.pending_measured: bool = True

        # --- UI event fan-out (SSE / logging) ------------------------------
        self._subscribers: list[Callable[[dict], None]] = []
        self.transcript: Deque[dict] = deque(maxlen=200)

    # -- state machine ------------------------------------------------------
    @property
    def state(self) -> ConvState:
        with self._lock:
            return self._state

    def set_state(self, state: ConvState) -> None:
        with self._lock:
            if self._state is state:
                return
            self._state = state
        self.emit("state", {"state": state.value})

    @property
    def turn_id(self) -> int:
        with self._lock:
            return self._turn_id

    def next_turn(self) -> int:
        """Bump the turn id. Audio/moves tagged with an older id are stale and dropped."""
        with self._lock:
            self._turn_id += 1
            return self._turn_id

    # -- barge-in -----------------------------------------------------------
    def request_interrupt(self) -> None:
        """Signal every producer to abandon the current reply immediately."""
        self.interrupt_event.set()
        self.set_state(ConvState.INTERRUPTED)
        # drop anything queued for the mouth
        _drain(self.tts_in)
        _drain(self.tts_audio)
        self.motion_intents.put(MotionIntent(kind="cancel"))
        self.emit("interrupt", {})

    def clear_interrupt(self) -> None:
        self.interrupt_event.clear()

    # -- latency ------------------------------------------------------------
    def mark(self, name: str) -> None:
        self._marks[name] = time.monotonic()

    def measure_e2e(self, t_end_speech: float) -> float:
        """Record end-to-end latency: end-of-speech -> first audio, in ms."""
        dt = (time.monotonic() - t_end_speech) * 1000.0
        self.last_latency_ms = dt
        self.emit("latency", {"e2e_ms": round(dt, 1)})
        return dt

    # -- UI fan-out ---------------------------------------------------------
    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._subscribers.append(fn)

    def emit(self, kind: str, data: dict) -> None:
        evt = {"kind": kind, "t": time.time(), **data}
        if kind in ("user", "assistant", "system"):
            self.transcript.append(evt)
        for fn in list(self._subscribers):
            try:
                fn(evt)
            except Exception:
                pass


def _drain(q: "queue.Queue[Any]") -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
