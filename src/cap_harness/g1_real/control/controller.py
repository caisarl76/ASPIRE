"""Level-1 controller: waypoints resampled against the wall clock into SONIC's planner stream.

The cascade differs from YAM's at the bottom. Level 1 (here) turns sparse
waypoints into a 50 Hz stream of 17-dof upper-body targets plus two 7-dof hand
targets; SONIC's planner mode consumes them directly and closes the loop on the
whole body, balance included. There is no PD layer of ours underneath, so this
is also where the per-tick joint step is bounded: SONIC tracks whatever it is
sent, and a target that jumps is a joint that jumps.

Every publish goes through ``env._send_tick``, which owns the emitted-target
chain: it steps the last emitted target toward the goal it is handed under the
joint and hand step limits, so the stale-feedback refusal, the publish lock and
the clamp live in one place, shared with the keep-alive and the engage hold.
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
_ARMS = slice(3, UPPER_BODY_DOF)  # SONIC-17 minus the three waist joints
# Settle-until-still: two consecutive 0.25 s windows whose means differ by less
# than 0.01 rad on every arm joint (about 3-4 mm at the hand).
_STILL_WINDOW_S = 0.25
_STILL_RAD = 0.01


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


def step_toward(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
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
    hands = env.config.hands
    # Per-column step limits over one (31,) row; the env clamps with the same values.
    max_step = np.concatenate(
        [
            np.full(UPPER_BODY_DOF, float(env.config.controller.max_joint_step_rad)),
            np.full(HAND_DOF, float(hands["left"].max_step_rad)),
            np.full(HAND_DOF, float(hands["right"].max_step_rad)),
        ]
    )

    def emitted() -> np.ndarray:
        return np.concatenate(env.last_command())

    # Start interpolation is one more waypoint at t=0: the chain target, i.e.
    # where the robot is commanded to be, so the first tick never steps back
    # toward a lagging measurement.
    interp_s = max(0.0, float(start_interp_s))
    if interp_s > 1e-9:
        ts = np.concatenate([[0.0], ts + interp_s])
        rows = np.vstack([emitted(), rows])
    stream_s = float(ts[-1])
    final = rows[-1]

    # A tracking controller lags the chain by its bias, and a plan starts at the
    # measurement, so the first tick would otherwise walk the command back by
    # that bias before moving. The arm columns start at the chain instead and
    # the offset fades out over the stream, so the final waypoint is unchanged.
    # (Waist is held, hands carry their own targets, and an uncommanded arm's
    # rows already equal the chain, so those columns get a zero offset.)
    start_offset = np.zeros(rows.shape[1])
    if not getattr(env, "_resume_from_measured", False):
        # After a feedback outage the first publish re-seeds the chain from the
        # measurement (SONIC idled the arms meanwhile), so there is no bias to
        # carry and the pre-outage chain must not be re-sent.
        start_offset[_ARMS] = (emitted() - rows[0])[_ARMS]

    def tick(row: np.ndarray) -> dict[str, Any] | None:
        return env._send_tick(row[_Q17], row[_LEFT_HAND], row[_RIGHT_HAND], hz=hz)

    def command_at(t_now: float) -> np.ndarray:
        if stream_s <= 1e-9:
            return rows[-1]
        # Cosine ease: zero velocity at both ends of the stream. SONIC takes the
        # streamed velocity as an input, so a linear ramp's velocity steps at
        # the start and the end are felt as jerks.
        phase = min(max(t_now / stream_s, 0.0), 1.0)
        tau = stream_s * (1.0 - math.cos(math.pi * phase)) / 2.0
        # The offset fades with the eased progress, not wall time, so the
        # command never dips back toward the measurement while easing in.
        return _sample(ts, rows, tau) + start_offset * (1.0 - tau / stream_s)

    t0 = time.monotonic()
    command_count = 0
    while True:
        _raise_if_stopped("during trajectory execution", dt)
        t_now = time.monotonic() - t0
        if (refused := tick(command_at(t_now))) is not None:
            return refused
        command_count += 1
        if t_now >= stream_s:
            break
        time.sleep(dt)

    # Catch-up: the step clamp can leave the emitted target behind the final
    # waypoint when the trajectory was faster than the step limits allow.
    catch_up_ticks = 0
    bound = math.ceil(float(np.max(np.abs(final - emitted()) / max_step))) + 1
    while not np.array_equal(emitted(), final) and catch_up_ticks < bound:
        _raise_if_stopped("during catch-up", dt)
        time.sleep(dt)
        if (refused := tick(final)) is not None:
            return refused
        catch_up_ticks += 1

    settle_ticks = max(0, round(max(0.0, float(settle_s)) / dt))
    extra_s = float(env.config.controller.settle_until_still_max_s) if settle_ticks else 0.0
    max_ticks = settle_ticks + round(extra_s / dt)
    window = max(1, round(_STILL_WINDOW_S / dt))
    arms: list[np.ndarray] = []
    settled = extra_s <= 0.0
    ticks_done = 0
    while ticks_done < max_ticks:
        _raise_if_stopped("during settle", dt)
        time.sleep(dt)
        if (refused := tick(emitted())) is not None:
            return refused
        ticks_done += 1
        if extra_s <= 0.0:
            continue
        obs = env.body.get_observations()
        if obs is not None:
            arms.append(np.concatenate([obs["left_joint_pos"], obs["right_joint_pos"]]))
        # Still: the mean of the last window moved less than _STILL_RAD from the
        # window before it on every joint. Window means, not raw samples, so
        # sensor noise does not read as motion.
        if ticks_done >= settle_ticks and len(arms) >= 2 * window:
            before = np.mean(arms[-2 * window : -window], axis=0)
            after = np.mean(arms[-window:], axis=0)
            if float(np.max(np.abs(after - before))) < _STILL_RAD:
                settled = True
                break

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
        "settle_ticks": int(ticks_done),
        "settled": bool(settled),
        "playback_speed": speed,
    }


__all__ = ["execute_upper_body_trajectory", "step_toward"]
