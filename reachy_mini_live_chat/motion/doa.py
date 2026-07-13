"""Direction-of-arrival helpers: turn the mic-array angle into a head orientation.

SDK DOA convention (from the Python SDK docs): ``0 = left, π/2 = front, π = right``.
World frame is x-forward, y-left, z-up. So a source at DOA angle ``a`` sits roughly at
``(sin a, cos a, 0)`` in world coordinates, and the head yaw that faces it is ``π/2 - a``
(front → 0, left → +90°, right → −90°). We cap DOA-driven yaw to a natural ±70° and smooth
the (noisy) readings so the head doesn't jitter.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

DOA_FRONT = math.pi / 2
MAX_DOA_YAW = math.radians(70.0)


def doa_to_head_yaw(angle: float) -> float:
    yaw = DOA_FRONT - angle
    return max(-MAX_DOA_YAW, min(MAX_DOA_YAW, yaw))


def doa_to_world_point(angle: float, dist: float = 1.0) -> Tuple[float, float, float]:
    return (dist * math.sin(angle), dist * math.cos(angle), 0.0)


class DoaTracker:
    """EMA-smoothed DOA with a change gate to avoid twitchy head turns."""

    def __init__(self, alpha: float = 0.4, min_change_deg: float = 12.0) -> None:
        self.alpha = alpha
        self.min_change = math.radians(min_change_deg)
        self._yaw: Optional[float] = None
        self._last_emitted: Optional[float] = None

    def update(self, angle: Optional[float]) -> Optional[float]:
        """Feed a raw DOA angle (rad). Returns a target head yaw (rad) worth acting on, else None."""
        if angle is None:
            return None
        target = doa_to_head_yaw(angle)
        self._yaw = target if self._yaw is None else (1 - self.alpha) * self._yaw + self.alpha * target
        if self._last_emitted is None or abs(self._yaw - self._last_emitted) >= self.min_change:
            self._last_emitted = self._yaw
            return self._yaw
        return None

    @property
    def yaw(self) -> Optional[float]:
        return self._yaw
