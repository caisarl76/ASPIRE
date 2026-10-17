"""Level-1 controller: waypoints resampled against the wall clock into SONIC's planner stream.

The cascade differs from YAM's at the bottom. Level 1 (here) turns sparse
waypoints into a 50 Hz stream of 17-dof upper-body targets plus two 7-dof hand
targets; SONIC's planner mode consumes them directly and closes the loop on the
whole body, balance included. There is no PD layer of ours underneath, so this
is also where the per-tick joint step is bounded: SONIC tracks whatever it is
sent, and a target that jumps is a joint that jumps.

Every publish goes through ``env._send_tick`` so the stale-feedback refusal and
the publish lock live in one place, shared with the keep-alive.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from cap_harness.g1_real.codec import HAND_DOF, UPPER_BODY_DOF, upper_body_from_arms
from cap_harness.yam_real.control.controller import _raise_if_stopped, _sample

# One (N, 31) row per waypoint: q17, then the left and right hand, so a single
# ``_sample`` call interpolates everything a tick needs.
_Q17 = slice(0, UPPER_BODY_DOF)
_LEFT_HAND = slice(UPPER_BODY_DOF, UPPER_BODY_DOF + HAND_DOF)
_RIGHT_HAND = slice(UPPER_BODY_DOF + HAND_DOF, UPPER_BODY_DOF + 2 * HAND_DOF)


def _rows(
    waist: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    q17 = np.stack(
        [upper_body_from_arms(waist, lhs, rhs) for lhs, rhs in zip(left, right, strict=True)]
    )
    return np.hstack([q17, left_hand, right_hand])


def _step_toward(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move at most ``max_step`` per joint toward ``target``; land exactly on it when in reach."""
    delta = target - prev
    if np.all(np.abs(delta) <= max_step):
        return target.copy()
    return prev + np.clip(delta, -max_step, max_step)


def execute_upper_body_trajectory(
    env: Any,
    timestamps: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    *,
    command_hz: float,
    start_interp_s: float = 0.0,
    settle_s: float = 0.2,
    playback_speed: float = 1.0,
) -> dict[str, Any]:
    """Stream both arms and hands at ``command_hz``; waist targets come from the profile.

    Three phases after the optional start interpolation: the timestamped
    trajectory, a catch-up phase that keeps stepping (clamped) until the emitted
    target equals the final waypoint, then a settle hold. Returns a result dict;
    a stale-feedback refusal from ``env._send_tick`` is returned as-is, a stop
    raises ``KeyboardInterrupt`` like the YAM controller.
    """
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 1:
        return {"success": False, "reason": "empty timestamps"}
    n = int(ts.size)
    ts = ts - float(ts[0])
    keep = np.ones(n, dtype=bool)
    keep[1:] = np.diff(ts) > 1e-9
    ts = ts[keep]
    rows = _rows(
        np.asarray(env.config.waist_targets, dtype=np.float64),
        np.asarray(left, dtype=np.float64).reshape(n, -1)[keep],
        np.asarray(right, dtype=np.float64).reshape(n, -1)[keep],
        np.asarray(left_hand, dtype=np.float64).reshape(n, HAND_DOF)[keep],
        np.asarray(right_hand, dtype=np.float64).reshape(n, HAND_DOF)[keep],
    )

    speed = max(0.05, float(playback_speed))
    ts = ts / speed
    duration_s = float(ts[-1])
    hz = max(1.0, float(command_hz))
    dt = 1.0 / hz
    max_step = float(env.config.controller.max_joint_step_rad)

    measured_q17, measured_lh, measured_rh = env.measured_command()
    # Start interpolation is one more waypoint: where the robot is, at t=0.
    interp_s = max(0.0, float(start_interp_s))
    if interp_s > 1e-9:
        ts = np.concatenate([[0.0], ts + interp_s])
        rows = np.vstack([np.concatenate([measured_q17, measured_lh, measured_rh]), rows])
    stream_s = float(ts[-1])

    prev = measured_q17.copy()
    final = rows[-1]

    def tick(row: np.ndarray, *, hold: bool = False) -> dict[str, Any] | None:
        nonlocal prev
        q17 = prev if hold else _step_toward(prev, row[_Q17], max_step)
        refused = env._send_tick(q17, (q17 - prev) * hz, row[_LEFT_HAND], row[_RIGHT_HAND])
        prev = q17
        return refused

    t0 = time.monotonic()
    command_count = 0
    while True:
        _raise_if_stopped("during trajectory execution", dt)
        t_now = time.monotonic() - t0
        if (refused := tick(_sample(ts, rows, t_now))) is not None:
            return refused
        command_count += 1
        if t_now >= stream_s:
            break
        time.sleep(dt)

    # Catch-up: the step clamp can leave the emitted target behind the final
    # waypoint when the trajectory was faster than max_joint_step_rad allows.
    catch_up_ticks = 0
    bound = math.ceil(float(np.max(np.abs(final[_Q17] - prev))) / max_step) + 1
    while not np.array_equal(prev, final[_Q17]) and catch_up_ticks < bound:
        _raise_if_stopped("during catch-up", dt)
        time.sleep(dt)
        if (refused := tick(final)) is not None:
            return refused
        catch_up_ticks += 1

    for _ in range(max(0, round(max(0.0, float(settle_s)) / dt))):
        _raise_if_stopped("during settle", dt)
        time.sleep(dt)
        if (refused := tick(final, hold=True)) is not None:
            return refused

    return {
        "success": True,
        "reason": "ok",
        "waypoints": int(np.count_nonzero(keep)),
        "duration_s": round(duration_s, 4),
        "command_hz": hz,
        "command_count": int(command_count),
        "catch_up_ticks": int(catch_up_ticks),
        "start_interp_s": interp_s,
        "settle_s": float(settle_s),
        "playback_speed": speed,
    }


__all__ = ["execute_upper_body_trajectory"]
