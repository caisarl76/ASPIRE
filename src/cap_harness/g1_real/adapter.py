"""``EnvironmentAdapter`` for the real G1: the YAM adapter with the G1 codec injected.

Everything that makes the YAM adapter safe to run model-generated code against --
the motion interlock, the recorder contract, the trajectory residual check --
is inherited unchanged. What is G1's own: a 7-DOF planning context on the torso
frame, per-arm joint limits, and the Dex3 hands, which the shared gripper tools
drive along an open/closed line and the two extension tools here address joint
by joint.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ExecutionResult,
    RobotPlanningContext,
    RobotState,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.g1_real import codec
from cap_harness.g1_real.registry import G1RealTaskRegistry
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.registry import YamRealTaskRegistry

# Contact-aware Dex3 close (close_gripper). Closure c runs 0 (open) to 1 (the
# profile's full-close pose); it advances in steps, each given time to settle,
# and stops when the fingers lag their command, i.e. are blocked by an object.
GRIP_STEP = 0.05
GRIP_SETTLE_S = 0.15
GRIP_CONTACT_LAG_RAD = 0.15  # per moving joint
GRIP_CONTACT_JOINTS = 2  # blocked joints that count as contact
GRIP_SQUEEZE = (
    0.15  # closure past the contact point; in sim 0.08 let a 0.5 kg bottle slip, 0.15 lifts it
)


class G1RealAdapter(YamRealAdapter):
    """Protocol-facing wrapper over a G1 station."""

    # An open Dex3 hand reads ~0.94 when its long finger joints sit ~0.1 rad off
    # zero (the left hand on PC2 .222, 2026-10-20); that is open for approaching.
    home_gripper_open_min = 0.9

    def __init__(
        self,
        env: Any,
        *,
        registry: YamRealTaskRegistry | None = None,
        task_context: TaskContext | None = None,
        allow_physical_motion: bool = False,
        run_observer: object | None = None,
    ) -> None:
        super().__init__(
            env,
            registry=registry or G1RealTaskRegistry(),
            task_context=task_context,
            allow_physical_motion=allow_physical_motion,
            run_observer=run_observer,
            codec=codec,
        )
        # Read by CapApi.move_to_pose: SONIC tracks a streamed target with a
        # steady bias, so the goal is re-commanded from the measured residual.
        self.pose_correction_rounds = env.config.controller.pose_correction_rounds

    def _provenance(self) -> dict[str, Any]:
        # No calibration bundle on this station: the profile hash attributes an episode.
        config = self._env.config
        return {"station": config.station, "config_sha256": config.config_sha256}

    def _held_joints(self, state: RobotState, side: str) -> np.ndarray:
        """The chain target for an uncommanded arm, clamped into its limits.

        Under gravity load the measured joints sit a tracking error behind the
        command; holding the measurement would re-command that sag on every
        single-arm batch and ratchet the chain downward. Before engage there is
        no chain, and the measured hold lets the plant refuse the batch itself.
        """
        try:
            q17, _, _ = self._env.last_command()
        except RuntimeError:
            return super()._held_joints(state, side)
        _, left, right = codec.arms_from_upper_body(q17)
        lower, upper = self._limits(side)
        return np.clip({"left": left, "right": right}[side], lower, upper)

    def _held_gripper(self, state: RobotState, side: str) -> float:
        """The chain hand's scalar, which the plant holds verbatim.

        A measured scalar is racy while the hand still moves: the next feedback
        message changes it, the plant no longer recognises the column as a
        hold, and a joint-level grasp gets projected onto the open/closed line.
        """
        try:
            _, left, right = self._env.last_command()
        except RuntimeError:
            return super()._held_gripper(state, side)
        hand = self._env.config.hands[side]
        return codec.hand_scalar({"left": left, "right": right}[side], hand.open, hand.closed)

    # -- hands -------------------------------------------------------------

    def set_hand_joints(self, arm: str, joints: list[float]) -> ExecutionResult:
        """Drive one Dex3 hand to explicit finger joints while both arms hold."""
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        side = self.resolve_arm(arm)
        # Same guard as ``step``: an unknown arm must fail as a typed refusal
        # before anything is counted or recorded, not as a ValueError later.
        if side not in self.arms:
            return self._denied_execution(
                ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"no arm {arm!r} on this {self.embodiment} station",
                    details={"known_arms": list(self.arms)},
                )
            )
        self._before_command()
        try:
            result = self._env.set_hand_joints(side, joints)
        except Exception as exc:
            denied = self._denied_execution(self._plant_failure(exc, "set_hand_joints"))
            self._record(denied)
            return denied
        self._command_rpc_count += 1
        execution = ExecutionResult(
            ok=bool(result.get("success", True)),
            error=self._plant_refusal(result, "set_hand_joints"),
            steps_executed=1,
            final_observation=self.get_observation(),
            diagnostics={
                "reason": str(result.get("reason", "")),
                "command_rpc_count": self._command_rpc_count,
            },
        )
        # The recorder's action space is the gripper scalar, so the COMMANDED
        # hand pose is recorded as its projection onto the open/closed line --
        # never the measured one, which still reads open while the hand closes.
        # A refused command commanded nothing, so it falls back to measured.
        gripper = None
        if execution.ok:
            hand = self._env.config.hands[side]
            gripper = {side: codec.hand_scalar(joints, hand.open, hand.closed)}
        self._record(execution, gripper=gripper)
        return execution

    def close_gripper(self, *, arm: str = "primary") -> ExecutionResult:
        """Close a Dex3 hand until its fingers stop on an object, then squeeze lightly.

        Steps the hand from where it is toward the profile's full-close pose
        and watches the measured fingers: when at least two of them lag their
        command by more than 0.15 rad they are blocked, and the hand is
        commanded a little past where they stopped instead of to full close,
        which would keep pushing into the object. Reports ``contact`` (False:
        it closed on nothing), the final ``closure`` and the resulting
        ``gripper_position`` scalar, which later arm moves hold.
        """
        side = self.resolve_arm(arm)
        if side not in self.arms:
            return self.set_hand_joints(arm, [])  # the same typed refusal
        hand = self._env.config.hands[side]
        opened = np.asarray(hand.open, dtype=np.float64)
        span = np.asarray(hand.closed, dtype=np.float64) - opened
        moving = np.abs(span) > 0.1

        def pose(closure: float) -> list[float]:
            return (opened + closure * span).tolist()

        def measured() -> np.ndarray:
            return np.asarray(self._env.get_observations(side)["hand_q"], dtype=np.float64)

        _, *held = self._env.last_command()
        closure = 1.0 - codec.hand_scalar(held[self.arms.index(side)], opened, hand.closed)
        contact, steps, lag = False, 0, np.zeros(int(moving.sum()))
        while closure < 1.0:
            closure = min(1.0, closure + GRIP_STEP)
            result = self.set_hand_joints(side, pose(closure))
            steps += 1
            if not result.ok:
                return result
            time.sleep(GRIP_SETTLE_S)
            lag = np.abs(np.asarray(pose(closure)) - measured())[moving]
            if int(np.sum(lag > GRIP_CONTACT_LAG_RAD)) >= GRIP_CONTACT_JOINTS:
                contact = True
                break
        if contact:
            reached = 1.0 - codec.hand_scalar(measured(), opened, hand.closed)
            closure = min(1.0, reached + GRIP_SQUEEZE)
            result = self.set_hand_joints(side, pose(closure))
            steps += 1
            if not result.ok:
                return result
            time.sleep(GRIP_SETTLE_S)
        _, *held = self._env.last_command()
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=self.get_observation(),
            diagnostics={
                "completion": "contact" if contact else "closed_empty",
                "contact": contact,
                "closure": round(float(closure), 4),
                "gripper_position": codec.hand_scalar(
                    held[self.arms.index(side)], opened, hand.closed
                ),
                "hand_q": measured().round(4).tolist(),
                "max_finger_lag_rad": round(float(lag.max(initial=0.0)), 4),
            },
        )

    # -- posture and leg clearance -------------------------------------------------

    def _posture(self):
        if getattr(self, "_posture_model", None) is None:
            from cap_harness.g1_real.posture import G1Posture

            hands = self._env.config.hands
            self._posture_model = G1Posture(
                {s: hands[s].joint_names for s in self.arms},
                {s: hands[s].closed for s in self.arms},
            )
            self._posture_arms = self._posture_model.arm_qpos(
                {s: self._env.config.arms[s].joint_names for s in self.arms}
            )
        return self._posture_model

    def _posture_from_feedback(self) -> bool:
        raw = self._env.body.get_observations()
        if raw is None or "body_q" not in raw:
            return False
        self._posture().set_state(raw["body_q"], {s: raw[f"{s}_hand_q"] for s in self.arms})
        return True

    def get_posture(self) -> dict[str, Any]:
        """Whole-body posture from SONIC's feedback, torso frame: shoulder, elbow, wrist,
        palm center and normal per arm, hips, knees and ankles, and each forearm's
        clearance to the legs. A read, not a command.
        """
        if not self._posture_from_feedback():
            return {"ok": False, "error": "no SONIC feedback"}
        return {"ok": True, **self._posture().summary()}

    def _leg_guard(self, trajectory: Trajectory | SynchronizedTrajectory) -> ExecutionResult | None:
        """Refuse a trajectory that brings a forearm or hand closer than the margin to the
        legs, unless it moves away from them (so a tight start can still be left).
        """
        margin = float(self._env.config.controller.leg_clearance_m)
        if margin <= 0.0 or not self._posture_from_feedback():
            return None
        if isinstance(trajectory, Trajectory):
            rows = {self.resolve_arm(trajectory.arm): np.asarray(trajectory.joint_positions)}
        else:
            rows = {
                self.resolve_arm(a): np.asarray(q) for a, q in trajectory.joint_positions.items()
            }
        posture = self._posture()
        start = {side: posture.leg_clearance(side)[0] for side in rows}
        n = len(next(iter(rows.values())))
        for i in np.unique(np.linspace(0, n - 1, min(n, 40)).astype(int)):
            posture.set_arms({side: q[i] for side, q in rows.items()}, self._posture_arms)
            for side in rows:
                clearance, arm_geom, leg_geom = posture.leg_clearance(side)
                if clearance < margin and clearance < start[side] - 0.005:
                    return ExecutionResult(
                        ok=False,
                        steps_executed=0,
                        error=ApiError(
                            code=ErrorCode.PLANNING_FAILED,
                            message=(
                                f"trajectory brings the {side} {arm_geom.removesuffix('_fit')} "
                                f"within {clearance * 100:.1f} cm of the robot's "
                                f"{leg_geom.removesuffix('_fit')} (minimum {margin * 100:.0f} cm); "
                                "nothing moved"
                            ),
                            details={
                                "arm": side,
                                "leg_clearance_m": round(clearance, 4),
                                "margin_m": margin,
                                "waypoint": int(i),
                            },
                        ),
                    )
        return None

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        refused = self._leg_guard(trajectory)
        return refused if refused is not None else super().execute_trajectory(trajectory)

    def get_hand_state(self) -> dict[str, list[float]]:
        """Measured Dex3 finger joints per hand; a read, not a command."""
        return {
            side: np.asarray(self._env.get_observations(side)["hand_q"], dtype=np.float64)
            .reshape(-1)
            .tolist()
            for side in self.arms
        }

    # -- planning and metadata --------------------------------------------

    def get_planning_context(self) -> RobotPlanningContext:
        config = self._env.config
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="g1_real",
            joint_names={side: config.arms[side].joint_names for side in self.arms},
            # Identity: the planner's own model places both arms on the torso.
            base_transforms={side: np.eye(4, dtype=np.float64) for side in self.arms},
            end_effector_links={side: f"{side}_grasp" for side in self.arms},
        )

    def get_task_metadata(self) -> dict[str, Any]:
        context = self._task_context
        return {
            "embodiment": self.embodiment,
            "station": self._env.config.station,
            "arms": list(self.arms),
            "dof": self.dof,
            "cameras": [self._env.config.camera.role],
            "language": "" if context is None else context.language,
            "hands": list(self.arms),
            "base_frame": codec.BASE_FRAME,
        }


__all__ = ["G1RealAdapter"]
