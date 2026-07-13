"""Safety clamps — Reachy Mini's documented joint limits, enforced by us too.

| axis | range |
|------|-------|
| head pitch / roll | [-40, +40]° |
| head yaw          | [-180, +180]° |
| body yaw          | [-160, +160]° |
| head-body yaw delta | <= 65° |

The SDK already clamps, but every pose *we* generate (idle motion, DOA look-at) is routed
through here so our own math can never command past the safe envelope. Antenna range is not
published; we sanitize to a conservative ±150°.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

HEAD_PITCH_ROLL_DEG = 40.0
HEAD_YAW_DEG = 180.0
BODY_YAW_DEG = 160.0
YAW_DELTA_DEG = 65.0
ANTENNA_DEG = 150.0  # heuristic (undocumented)

_D2R = np.pi / 180.0
_R2D = 180.0 / np.pi


def _clip(v: float, lim: float) -> float:
    return float(max(-lim, min(lim, v)))


def clamp_rpy_deg(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float]:
    return (
        _clip(roll, HEAD_PITCH_ROLL_DEG),
        _clip(pitch, HEAD_PITCH_ROLL_DEG),
        _clip(yaw, HEAD_YAW_DEG),
    )


def clamp_head_pose(pose: np.ndarray) -> np.ndarray:
    """Clamp the rotation of a 4x4 head pose (translation preserved)."""
    from scipy.spatial.transform import Rotation as R

    pose = np.array(pose, dtype=np.float64, copy=True)
    roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    roll, pitch, yaw = clamp_rpy_deg(roll, pitch, yaw)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    return pose


def head_yaw_deg(pose: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation as R

    return float(R.from_matrix(np.asarray(pose)[:3, :3]).as_euler("xyz", degrees=True)[2])


def clamp_body_yaw(body_yaw_rad: float, head_yaw_rad: float = 0.0) -> float:
    """Clamp body yaw to range AND to <= 65° from the head yaw."""
    body = _clip(body_yaw_rad * _R2D, BODY_YAW_DEG)
    head = head_yaw_rad * _R2D
    delta = body - head
    if delta > YAW_DELTA_DEG:
        body = head + YAW_DELTA_DEG
    elif delta < -YAW_DELTA_DEG:
        body = head - YAW_DELTA_DEG
    body = _clip(body, BODY_YAW_DEG)
    return body * _D2R


def clamp_antennas(antennas: Sequence[float]) -> List[float]:
    lim = ANTENNA_DEG * _D2R
    return [float(max(-lim, min(lim, a))) for a in antennas]
