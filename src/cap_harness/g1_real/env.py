"""The G1 plant: SONIC body link, camera, kinematics, and the one path that actuates.

``G1Env`` is the G1 twin of :class:`~cap_harness.yam_real.env.RealYamEnv` and
keeps its contract: :meth:`execute_action_batch` is the **only** way anything
commands the robot -- ``go_home``, ``set_gripper`` and ``set_hand_joints`` are
built on it -- and the episode recorder reads ``_last_action_input`` /
``_action_input_seq`` exactly as on YAM.

What is new is the stream. SONIC's planner mode drops back to its idle pose
when the ``planner`` topic goes quiet for a second, so between batches a
keep-alive thread re-sends the last emitted target every 20 ms. One re-entrant
lock serialises every publish -- batch tick, keep-alive, engage hold -- and
every publish refuses when SONIC's feedback is older than
``sonic.feedback_stale_s``: a target sent blind is a target nothing checks.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from cap_harness.g1_real import codec
from cap_harness.g1_real.body_client import G1BodyClient
from cap_harness.g1_real.codec import ARM_DOF, HAND_DOF, UPPER_BODY_DOF
from cap_harness.g1_real.config import G1StationConfig
from cap_harness.g1_real.control.controller import execute_upper_body_trajectory
from cap_harness.g1_real.kinematics import G1Kinematics
from cap_harness.yam_real.action import YamActionBatch, resolve_action_batch

ARMS = ("left", "right")
CONTROL_CONTRACT_VERSION = "g1-sonic-v1-planner-stream"
KEEP_ALIVE_PERIOD_S = 0.02
#: Seconds a full open-to-closed hand transition is given when no duration is passed.
HAND_FULL_TRAVEL_S = 1.0

_log = logging.getLogger(__name__)


class G1Env:
    """Unitree G1 upper body over GEAR-SONIC, driven through a single action chokepoint."""

    dof = ARM_DOF

    def __init__(
        self,
        config: G1StationConfig,
        body: G1BodyClient,
        *,
        camera: Any | None = None,
        kinematics: G1Kinematics | None = None,
    ) -> None:
        self.config = config
        self.body = body
        self._camera = camera
        self.kin = kinematics if kinematics is not None else G1Kinematics(config.model_xml)
        # Both observation and planning write the mink configuration; one lock
        # held across write-then-read, as on YAM.
        self._kin_lock = threading.Lock()
        # Serialises every publish. Re-entrant so a batch tick can call
        # _send_tick while the keep-alive loop holds it for its read-then-send.
        self._lock = threading.RLock()
        self._last_cmd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._streaming = False
        self._stale_logged = False
        self._stop = threading.Event()
        self._keep_alive: threading.Thread | None = None
        self._engaged: dict[str, Any] | None = None
        self._close_hooks: list[Any] = []
        self._closed = False
        # Recorder contract, see RealYamEnv: last commanded (joints, gripper scalar) per arm.
        self._last_action_input: dict[str, tuple[np.ndarray, float]] = {}
        self._action_input_seq: dict[str, int] = dict.fromkeys(ARMS, 0)

    # -- observation -------------------------------------------------------

    def _raw(self) -> dict[str, Any]:
        raw = self.body.get_observations()
        if raw is None:
            raise RuntimeError("no SONIC feedback received yet")
        return raw

    def measured_command(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The measured state as a ``(q17, left_hand, right_hand)`` command triple."""
        raw = self._raw()
        q17 = codec.upper_body_from_arms(
            raw["waist"], raw["left_joint_pos"], raw["right_joint_pos"]
        )
        return (
            q17,
            np.asarray(raw["left_hand_q"], dtype=np.float64),
            np.asarray(raw["right_hand_q"], dtype=np.float64),
        )

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        """One arm's state: joints, hand, gripper scalar, and the torso-frame grasp pose."""
        if side not in ARMS:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        raw = self._raw()
        with self._kin_lock:
            left_pos, left_quat, right_pos, right_quat = self.kin.forward_kinematics(
                raw["left_joint_pos"], raw["right_joint_pos"], raw["waist"]
            )
        poses = {"left": (left_pos, left_quat), "right": (right_pos, right_quat)}
        hand_q = np.asarray(raw[f"{side}_hand_q"], dtype=np.float64).reshape(HAND_DOF)
        hand = self.config.hands[side]
        return {
            "joint_pos": np.asarray(raw[f"{side}_joint_pos"], dtype=np.float64).reshape(ARM_DOF),
            "joint_vel": np.asarray(raw[f"{side}_joint_vel"], dtype=np.float64).reshape(ARM_DOF),
            "gripper_pos": np.array([codec.hand_scalar(hand_q, hand.open, hand.closed)]),
            "hand_q": hand_q,
            "ee_pos": poses[side][0],
            "ee_quat": poses[side][1],
        }

    def read_camera(self) -> Any | None:
        return None if self._camera is None else self._camera.read()

    def read_aux_camera(self, role: str) -> Any | None:
        del role  # no image-only cameras on this station
        return None

    @property
    def aux_camera_roles(self) -> tuple[str, ...]:
        return ()

    @property
    def camera_aliases(self) -> tuple[str, ...]:
        return () if self._camera is None else (str(self.config.camera.role),)

    def camera_frame(self, alias: str) -> Any | None:
        if self._camera is not None and alias == str(self.config.camera.role):
            return self._camera.read()
        return None

    def render_rgb(self, alias: str) -> np.ndarray | None:
        frame = self.camera_frame(alias)
        return None if frame is None else np.asarray(frame.rgb, dtype=np.uint8)

    def add_close_hook(self, hook: Any) -> None:
        """Call ``hook`` on close, before the devices go down; failures are logged."""
        self._close_hooks.append(hook)

    # -- actuation ---------------------------------------------------------

    def _send_tick(
        self,
        q17: np.ndarray,
        dq17: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> dict[str, Any] | None:
        """Publish one planner message, or return a refusal when feedback is stale."""
        with self._lock:
            age = self.body.feedback_age_s()
            limit = self.config.sonic.feedback_stale_s
            if age > limit:
                return {
                    "success": False,
                    "reason": f"feedback stale ({age:.3f} s > {limit:.3f} s)",
                }
            q = np.asarray(q17, dtype=np.float64).reshape(UPPER_BODY_DOF)
            lh = np.asarray(left_hand, dtype=np.float64).reshape(HAND_DOF)
            rh = np.asarray(right_hand, dtype=np.float64).reshape(HAND_DOF)
            self.body.command_upper_body(q, dq17, lh, rh)
            self._last_cmd = (q, lh, rh)
        return None

    def _keep_alive_loop(self) -> None:
        while not self._stop.wait(KEEP_ALIVE_PERIOD_S):
            if self._streaming:
                continue
            with self._lock:
                if self._streaming or self._last_cmd is None:
                    continue
                q17, lh, rh = self._last_cmd
                refused = self._send_tick(q17, np.zeros(UPPER_BODY_DOF), lh, rh)
            if refused is None:
                self._stale_logged = False
            elif not self._stale_logged:
                self._stale_logged = True
                _log.warning("keep-alive paused: %s", refused["reason"])

    @property
    def keep_alive_running(self) -> bool:
        return self._keep_alive is not None and self._keep_alive.is_alive()

    def engage(self) -> dict[str, Any]:
        """Hold the measured pose on the stream, then hand SONIC its ``start``.

        Idempotent: a second call reports ``already_engaged`` and sends nothing.
        Refuses before any publish when the feedback is stale, since the hold
        command is seeded from it.
        """
        if self._engaged is not None:
            return {**self._engaged, "already_engaged": True}
        age = self.body.feedback_age_s()
        limit = self.config.sonic.feedback_stale_s
        if age > limit:
            raise RuntimeError(f"feedback stale ({age:.3f} s > {limit:.3f} s); not engaging")
        q17, lh, rh = self.measured_command()
        dt = 1.0 / float(self.config.controller.command_stream_hz)
        ticks = max(1, round(float(self.config.sonic.engage_hold_s) / dt))
        for _ in range(ticks):
            refused = self._send_tick(q17, np.zeros(UPPER_BODY_DOF), lh, rh)
            if refused is not None:
                raise RuntimeError(refused["reason"])
            time.sleep(dt)
        self.body.send_start()
        self._keep_alive = threading.Thread(
            target=self._keep_alive_loop, name="g1-keep-alive", daemon=True
        )
        self._keep_alive.start()
        self._engaged = {"success": True, "hold_ticks": ticks, "already_engaged": False}
        return dict(self._engaged)

    def execute_action_batch(
        self,
        batch: YamActionBatch | dict[str, Any],
        *,
        command_hz: float | None = None,
        start_interp_s: float = 0.0,
        settle_s: float = 0.2,
        playback_speed: float = 1.0,
    ) -> dict[str, Any]:
        """Execute one typed action batch. The only path to the robot.

        Hands follow the batch's gripper scalar through ``codec.hand_pose``
        unless ``batch.meta["hand_joints"]`` carries explicit ``{side: (N, 7)}``
        rows for a side. The waist holds the profile's target on every tick.
        """
        resolved = resolve_action_batch(self, batch)
        self._note_action_input(resolved)
        n = int(resolved.timestamps.size)
        overrides = resolved.meta.get("hand_joints") or {}
        scalars = {
            "left": resolved.left_gripper_positions,
            "right": resolved.right_gripper_positions,
        }
        hands: dict[str, np.ndarray] = {}
        for side in ARMS:
            if side in overrides:
                hands[side] = np.asarray(overrides[side], dtype=np.float64).reshape(n, HAND_DOF)
            else:
                hand = self.config.hands[side]
                column = np.asarray(scalars[side], dtype=np.float64).reshape(-1)
                hands[side] = np.stack([codec.hand_pose(p, hand.open, hand.closed) for p in column])
        self._streaming = True
        try:
            result = execute_upper_body_trajectory(
                self,
                resolved.timestamps,
                resolved.left_joint_positions,
                resolved.right_joint_positions,
                hands["left"],
                hands["right"],
                command_hz=float(command_hz or self.config.controller.command_stream_hz),
                start_interp_s=start_interp_s,
                settle_s=settle_s,
                playback_speed=playback_speed,
            )
        finally:
            self._streaming = False
        result.setdefault("source", resolved.source)
        result.setdefault("input_space", resolved.input_space)
        return result

    def _note_action_input(self, resolved: Any) -> None:
        """Publish the batch's final target for the episode recorder (see RealYamEnv)."""
        for side, rows, grip in (
            ("left", resolved.left_joint_positions, resolved.left_gripper_positions),
            ("right", resolved.right_joint_positions, resolved.right_gripper_positions),
        ):
            self._last_action_input[side] = (
                np.asarray(rows, dtype=np.float64)[-1].copy(),
                float(np.asarray(grip, dtype=np.float64).reshape(-1)[-1]),
            )
            self._action_input_seq[side] += 1

    def _held_arms(self) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
        """Measured observations and the measured joints clamped into the limits."""
        observed = {side: self.get_observations(side) for side in ARMS}
        joints = {
            side: np.clip(observed[side]["joint_pos"], arm.lower, arm.upper)
            for side, arm in self.config.arms.items()
        }
        return observed, joints

    def _hand_transition(
        self,
        side: str,
        observed: dict[str, dict[str, np.ndarray]],
        joints: dict[str, np.ndarray],
        goal_scalar: float,
        travel: float,
        duration_s: float | None,
        *,
        source: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Two waypoints: arms held, one hand moving; the window scales with the travel."""
        start = {name: float(observed[name]["gripper_pos"][0]) for name in ARMS}
        goal = {**start, side: goal_scalar}
        duration = (
            max(0.05, travel * HAND_FULL_TRAVEL_S)
            if duration_s is None
            else max(0.02, float(duration_s))
        )
        batch = YamActionBatch.joint_abs(
            [0.0, duration],
            [joints["left"], joints["left"]],
            [joints["right"], joints["right"]],
            [[start["left"]], [goal["left"]]],
            [[start["right"]], [goal["right"]]],
            source=source,
            meta={"side": side, **meta},
        )
        return self.execute_action_batch(batch, settle_s=0.0)

    def set_gripper(
        self, side: str, position: float, *, duration_s: float | None = None
    ) -> dict[str, Any]:
        """Move one hand along its open/closed line while both arms hold."""
        if side not in ARMS:
            return {"success": False, "reason": f"invalid side: {side!r}"}
        observed, joints = self._held_arms()
        goal = float(np.clip(position, 0.0, 1.0))
        travel = abs(goal - float(observed[side]["gripper_pos"][0]))
        result = self._hand_transition(
            side, observed, joints, goal, travel, duration_s, source="cap_gripper", meta={}
        )
        result.update({"side": side, "gripper": goal})
        return result

    def set_hand_joints(
        self, side: str, joints7: Any, *, duration_s: float | None = None
    ) -> dict[str, Any]:
        """Drive one Dex3 hand to explicit finger joints while both arms hold."""
        if side not in ARMS:
            return {"success": False, "reason": f"invalid side: {side!r}"}
        hand = self.config.hands[side]
        target = np.asarray(joints7, dtype=np.float64).reshape(-1)
        if target.shape != (HAND_DOF,) or not np.all(np.isfinite(target)):
            return {"success": False, "reason": "hand joints must be 7 finite values"}
        if np.any(target < hand.lower) or np.any(target > hand.upper):
            return {"success": False, "reason": f"{side} hand joints outside the profile limits"}
        observed, joints = self._held_arms()
        start = np.asarray(observed[side]["hand_q"], dtype=np.float64)
        travel = float(np.max(np.abs(target - start)) / np.max(np.abs(hand.closed - hand.open)))
        result = self._hand_transition(
            side,
            observed,
            joints,
            codec.hand_scalar(target, hand.open, hand.closed),
            travel,
            duration_s,
            source="cap_hand_joints",
            meta={"hand_joints": {side: np.stack([start, target])}},
        )
        result.update({"side": side, "hand_joints": target.tolist()})
        return result

    def go_home(self, *, duration_s: float = 3.0, keep_grippers: bool = True) -> dict[str, Any]:
        """Drive both arms to the profile's home pose with a start interpolation."""
        observed = {side: self.get_observations(side) for side in ARMS}
        grippers = {
            side: (float(observed[side]["gripper_pos"][0]) if keep_grippers else 1.0)
            for side in ARMS
        }
        batch = YamActionBatch.joint_abs(
            [0.0],
            [self.config.arms["left"].home_joints],
            [self.config.arms["right"].home_joints],
            [[grippers["left"]]],
            [[grippers["right"]]],
            source="cap_home",
        )
        return self.execute_action_batch(
            batch, start_interp_s=max(0.0, float(duration_s)), settle_s=0.2
        )

    # -- lifecycle ---------------------------------------------------------

    def control_contract(self) -> dict[str, Any]:
        """The versioned plant description recorded actions were executed under."""
        controller = self.config.controller
        return {
            "version": CONTROL_CONTRACT_VERSION,
            "action_semantics": (
                "joint7+hand7 absolute targets per arm, SONIC-17 upper body stream "
                "at command_stream_hz"
            ),
            "action_space": "joint_abs",
            "command_stream_hz": float(controller.command_stream_hz),
            "max_joint_step_rad": float(controller.max_joint_step_rad),
            "interpolation": "linear, wall-clock time-sampled, per-tick step clamp",
            "tracker": "GEAR-SONIC planner mode over the ZMQ planner topic",
            "owner": "cap_harness.g1_real.control.controller (level 1)",
            "gripper_action": "continuous position target in [0, 1]; 0.0=closed, 1.0=open",
            "station": self.config.station,
            "config_sha256": self.config.config_sha256,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._keep_alive is not None:
            self._keep_alive.join(timeout=1.0)
        for hook in reversed(self._close_hooks):
            try:
                hook()
            except Exception as exc:  # a hook's failure is not the station's
                _log.warning("close hook failed: %s: %s", type(exc).__name__, exc)
        errors: list[str] = []
        for name, device in (("body", self.body), ("camera", self._camera)):
            if device is None:
                continue
            try:
                device.close()
            except Exception as exc:  # every device gets its close attempt
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("G1 station cleanup failed: " + "; ".join(errors))


__all__ = ["ARMS", "CONTROL_CONTRACT_VERSION", "G1Env"]
