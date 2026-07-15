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


# --- rpy <-> matrix (extrinsic "xyz": R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) -----
# Matches scipy Rotation.from_euler("xyz")/as_euler("xyz") for our clamped range
# (|pitch| <= 40°, so no gimbal lock), letting us drop the scipy dependency: an
# on-robot client that offloads all inference shouldn't pull in scipy just for this.
def rpy_to_matrix(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    if degrees:
        roll, pitch, yaw = roll * _D2R, pitch * _D2R, yaw * _D2R
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_rpy(m: np.ndarray, degrees: bool = True) -> Tuple[float, float, float]:
    m = np.asarray(m, dtype=np.float64)
    pitch = np.arctan2(-m[2, 0], np.hypot(m[0, 0], m[1, 0]))
    roll = np.arctan2(m[2, 1], m[2, 2])
    yaw = np.arctan2(m[1, 0], m[0, 0])
    if degrees:
        return (float(roll * _R2D), float(pitch * _R2D), float(yaw * _R2D))
    return (float(roll), float(pitch), float(yaw))


def clamp_rpy_deg(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float]:
    return (
        _clip(roll, HEAD_PITCH_ROLL_DEG),
        _clip(pitch, HEAD_PITCH_ROLL_DEG),
        _clip(yaw, HEAD_YAW_DEG),
    )


def clamp_head_pose(pose: np.ndarray) -> np.ndarray:
    """Clamp the rotation of a 4x4 head pose (translation preserved)."""
    pose = np.array(pose, dtype=np.float64, copy=True)
    roll, pitch, yaw = matrix_to_rpy(pose[:3, :3], degrees=True)
    roll, pitch, yaw = clamp_rpy_deg(roll, pitch, yaw)
    pose[:3, :3] = rpy_to_matrix(roll, pitch, yaw, degrees=True)
    return pose


def head_yaw_deg(pose: np.ndarray) -> float:
    return matrix_to_rpy(np.asarray(pose)[:3, :3], degrees=True)[2]


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
