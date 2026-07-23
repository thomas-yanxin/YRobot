"""Shared runtime state between the audio, network and motion threads."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now() -> float:
    return time.monotonic()


@dataclass
class Shared:
    """Plain attributes only — single-writer per field, atomic under the GIL."""

    ready: bool = False  # a realtime session is up
    voice_active: bool = False  # user is speaking (energy gate)
    last_voice_onset: float = field(default_factory=lambda: now() - 3600)
    last_voice_end: float = field(default_factory=lambda: now() - 3600)
    play_head: float = 0.0  # monotonic time until which speaker audio is queued
    body_yaw: float = 0.0  # current commanded body yaw (written by motion)
    yaw_target: float = 0.0  # where the body should point (written by DOA logic)

    def robot_speaking(self) -> bool:
        return now() < self.play_head
