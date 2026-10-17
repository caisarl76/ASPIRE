"""Pure analysis for the on-robot tracking acceptance (spec section 9)."""

from __future__ import annotations

import math

import numpy as np

from cap_harness.contracts import Pose

HOLD_MAX_ERROR_RAD = 0.05
SEGMENT_MAX_POSITION_M = 0.015
SEGMENT_MAX_ORIENTATION_RAD = math.radians(5.0)


def hold_error(targets: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Max abs joint error per arm over ``(N, 14)`` arrays (left 7 then right 7)."""
    err = np.abs(np.asarray(measured, dtype=float) - np.asarray(targets, dtype=float))
    return {"left": float(err[:, :7].max()), "right": float(err[:, 7:].max())}


def segment_error(target_pose: Pose, measured_pose: Pose) -> tuple[float, float]:
    """Position error (m) and sign-invariant quaternion angle (rad) between two poses."""
    pos = float(np.linalg.norm(measured_pose.position - target_pose.position))
    dot = abs(float(np.dot(target_pose.quaternion_wxyz, measured_pose.quaternion_wxyz)))
    dot /= float(
        np.linalg.norm(target_pose.quaternion_wxyz) * np.linalg.norm(measured_pose.quaternion_wxyz)
    )
    return pos, 2.0 * math.acos(min(1.0, dot))
