"""Transport seam between the G1 embodiment and GEAR-SONIC.

SONIC owns balance and the low-level loop; ASPIRE only streams upper-body targets
on the ZMQ ``planner`` topic and reads the ``g1_debug`` feedback. Everything the
embodiment needs from that link is the :class:`G1BodyClient` protocol, so the
controller above it can run against :class:`FakeBodyClient` (a first-order lag
toward the last command) without a robot or a SONIC process.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import math
import threading
import time
from typing import Protocol

import numpy as np
import zmq

from cap_harness.g1_real.codec import (
    HAND_DOF,
    MUJOCO_ARM_SLICES,
    MUJOCO_UPPER_BODY_INDICES,
    MUJOCO_WAIST_SLICE,
    UPPER_BODY_DOF,
    heading_yaw_from_quat_wxyz,
)
from cap_harness.g1_real.config import G1SonicConfig
from cap_harness.g1_real.sonic.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)
from cap_harness.g1_real.sonic.zmq_state_subscriber import ZMQStateSubscriber

#: Default per-message hand joint step; the station profile's ``hands.*.max_step_rad`` overrides it.
HAND_STEP_RAD = 0.25
_POLL_PERIOD_S = 0.01
_SIDES = ("left", "right")

_log = logging.getLogger(__name__)


class G1BodyClient(Protocol):
    def get_observations(self) -> dict[str, np.ndarray | float] | None: ...

    def command_upper_body(
        self,
        q17: np.ndarray,
        dq17: np.ndarray,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
    ) -> None: ...

    def send_start(self) -> None: ...

    def seed_hands(self, left: np.ndarray, right: np.ndarray) -> None: ...

    def feedback_age_s(self) -> float: ...

    def close(self) -> None: ...


def _observation(
    body_q: np.ndarray,
    body_dq: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    base_quat: np.ndarray,
    heading_yaw: float,
    age_s: float,
    timestamp_s: float,
) -> dict[str, np.ndarray | float]:
    q = np.asarray(body_q, dtype=np.float64)
    dq = np.asarray(body_dq, dtype=np.float64)
    return {
        "left_joint_pos": q[MUJOCO_ARM_SLICES["left"]].copy(),
        "right_joint_pos": q[MUJOCO_ARM_SLICES["right"]].copy(),
        "waist": q[MUJOCO_WAIST_SLICE].copy(),
        "left_joint_vel": dq[MUJOCO_ARM_SLICES["left"]].copy(),
        "right_joint_vel": dq[MUJOCO_ARM_SLICES["right"]].copy(),
        "left_hand_q": np.array(left_hand_q, dtype=np.float64),
        "right_hand_q": np.array(right_hand_q, dtype=np.float64),
        "base_quat": np.array(base_quat, dtype=np.float64),
        "heading_yaw": float(heading_yaw),
        "age_s": float(age_s),
        "timestamp_s": float(timestamp_s),
    }


class SonicZmqBodyClient:
    """PUB planner commands to SONIC and poll its ``g1_debug`` feedback at 100 Hz."""

    def __init__(
        self,
        config: G1SonicConfig,
        *,
        hand_max_step_rad: float | Mapping[str, float] = HAND_STEP_RAD,
    ) -> None:
        self._config = config
        self._hand_step = {
            side: float(
                hand_max_step_rad[side]
                if isinstance(hand_max_step_rad, Mapping)
                else hand_max_step_rad
            )
            for side in _SIDES
        }
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(config.command_bind)
        self._sub = ZMQStateSubscriber(
            config.feedback_host, config.feedback_port, config.feedback_topic
        )
        self._last_hand = {side: np.zeros(HAND_DOF) for side in _SIDES}
        self._lock = threading.Lock()
        self._msg: dict | None = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="sonic-feedback", daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop.wait(_POLL_PERIOD_S):
            # A dead poll thread would read as permanently stale feedback, so a
            # bad message (or a socket hiccup) is logged and polling continues.
            try:
                msg = self._sub.get_msg(clear=False)
            except Exception as exc:
                _log.warning("feedback poll failed: %s: %s", type(exc).__name__, exc)
                continue
            with self._lock:
                # clear=False re-returns the last message; only a new object is new feedback,
                # so the stamp (and hence the age) is not refreshed while SONIC is silent.
                if msg is not None and msg is not self._msg:
                    self._msg, self._stamp = msg, time.monotonic()

    def _snapshot(self) -> tuple[dict | None, float]:
        with self._lock:
            return self._msg, self._stamp

    @staticmethod
    def _heading(msg: dict) -> float:
        """Yaw that holds the current heading, in the frame SONIC reads ``facing`` in.

        The planner's frame has yaw zero at the heading the policy initialised
        with: the fork's deploy publishes that correction as
        ``reference_heading_quat`` (valid while ``planner_reference_active``),
        older builds only ``init_base_quat`` plus ``delta_heading``. The raw IMU
        yaw is in a frame the planner never sees; sent as ``facing`` it commands
        a turn by the heading the robot had at init (a 90 deg turn on engage on
        the real robot, invisible in sim where that heading is zero). Without
        any heading telemetry the only safe answer is the planner's own zero.
        """

        def yaw(key: str) -> float:
            return heading_yaw_from_quat_wxyz(np.asarray(msg[key]).reshape(-1))

        if "reference_heading_quat" in msg:
            active = float(np.asarray(msg.get("planner_reference_active", 0)).reshape(-1)[0])
            if active != 1:
                return 0.0  # planner initialising at reference yaw zero
            reference = yaw("reference_heading_quat")
        elif "init_base_quat" in msg:
            reference = yaw("init_base_quat") + float(
                np.asarray(msg.get("delta_heading", 0.0)).reshape(-1)[0]
            )
        else:
            return 0.0
        delta = yaw("base_quat") - reference
        return math.atan2(math.sin(delta), math.cos(delta))

    def get_observations(self) -> dict[str, np.ndarray | float] | None:
        msg, stamp = self._snapshot()
        if msg is None:
            return None
        return _observation(
            msg["body_q"],
            msg["body_dq"],
            msg["left_hand_q"],
            msg["right_hand_q"],
            msg["base_quat"],
            self._heading(msg),
            time.monotonic() - stamp,
            stamp,
        )

    def feedback_age_s(self) -> float:
        msg, stamp = self._snapshot()
        return float("inf") if msg is None else time.monotonic() - stamp

    def _clamped_hand(self, side: str, target: np.ndarray | None) -> np.ndarray | None:
        if target is None:
            return None
        last = self._last_hand[side]
        limit = self._hand_step[side]
        step = np.clip(np.asarray(target, dtype=np.float64) - last, -limit, limit)
        self._last_hand[side] = last + step
        return self._last_hand[side]

    def command_upper_body(
        self,
        q17: np.ndarray,
        dq17: np.ndarray,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
    ) -> None:
        msg, _ = self._snapshot()
        yaw = 0.0 if msg is None else self._heading(msg)
        self._pub.send(
            build_planner_message(
                mode=self._config.mode,
                movement=(0.0, 0.0, 0.0),
                facing=(math.cos(yaw), math.sin(yaw), 0.0),
                speed=self._config.speed,
                height=self._config.height,
                upper_body_position=np.asarray(q17, dtype=np.float64).reshape(UPPER_BODY_DOF),
                upper_body_velocity=np.asarray(dq17, dtype=np.float64).reshape(UPPER_BODY_DOF),
                left_hand_position=self._clamped_hand("left", left_hand),
                right_hand_position=self._clamped_hand("right", right_hand),
            )
        )

    def send_start(self) -> None:
        self._pub.send(build_command_message(start=True, stop=False, planner=True))

    def seed_hands(self, left: np.ndarray, right: np.ndarray) -> None:
        """Start the hand step limiter from the measured hands, not from zeros."""
        for side, hand in zip(_SIDES, (left, right), strict=True):
            self._last_hand[side] = np.asarray(hand, dtype=np.float64).reshape(HAND_DOF).copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sub.close()
        self._pub.close()


class FakeBodyClient:
    """First-order lag toward the last command; ``tau_s`` may be changed mid-run."""

    def __init__(self, tau_s: float = 0.05, initial_q17: np.ndarray | None = None) -> None:
        self.tau_s = tau_s
        q17 = np.zeros(UPPER_BODY_DOF) if initial_q17 is None else np.asarray(initial_q17, float)
        self._q17 = q17.copy()
        self._target = q17.copy()
        self._hands = {side: np.zeros(HAND_DOF) for side in _SIDES}
        self._hand_targets = {side: np.zeros(HAND_DOF) for side in _SIDES}
        self._t = time.monotonic()
        self.commands: list[
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
        ] = []
        self.start_count = 0
        self.seeded_hands: tuple[np.ndarray, np.ndarray] | None = None
        self.stale = False

    def _advance(self) -> None:
        now = time.monotonic()
        alpha = 1.0 - math.exp(-(now - self._t) / self.tau_s)
        self._t = now
        self._q17 += (self._target - self._q17) * alpha
        for side in _SIDES:
            self._hands[side] += (self._hand_targets[side] - self._hands[side]) * alpha

    def command_upper_body(
        self,
        q17: np.ndarray,
        dq17: np.ndarray,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
    ) -> None:
        self._advance()  # settle toward the old target before it changes
        q = np.array(q17, dtype=np.float64)
        lh = None if left_hand is None else np.array(left_hand, dtype=np.float64)
        rh = None if right_hand is None else np.array(right_hand, dtype=np.float64)
        self.commands.append((q, np.array(dq17, dtype=np.float64), lh, rh))
        self._target = q.copy()
        for side, hand in zip(_SIDES, (lh, rh), strict=True):
            if hand is not None:
                self._hand_targets[side] = hand.copy()

    def get_observations(self) -> dict[str, np.ndarray | float]:
        self._advance()
        body_q = np.zeros(29)
        body_q[list(MUJOCO_UPPER_BODY_INDICES)] = self._q17
        return _observation(
            body_q,
            np.zeros(29),
            self._hands["left"],
            self._hands["right"],
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.0,
            self.feedback_age_s(),
            self._t,
        )

    def send_start(self) -> None:
        self.start_count += 1

    def seed_hands(self, left: np.ndarray, right: np.ndarray) -> None:
        self.seeded_hands = (np.array(left, dtype=np.float64), np.array(right, dtype=np.float64))

    def feedback_age_s(self) -> float:
        return float("inf") if self.stale else 0.0

    def close(self) -> None:
        pass
