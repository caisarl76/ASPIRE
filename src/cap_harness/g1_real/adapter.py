"""``EnvironmentAdapter`` for the real G1: the YAM adapter with the G1 codec injected.

Everything that makes the YAM adapter safe to run model-generated code against --
the motion interlock, the recorder contract, the trajectory residual check --
is inherited unchanged. What is G1's own: a 7-DOF planning context on the torso
frame, per-arm joint limits, and the Dex3 hands, which the shared gripper tools
drive along an open/closed line and the two extension tools here address joint
by joint.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cap_harness.contracts import ExecutionResult, RobotPlanningContext, RobotState, TaskContext
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.g1_real import codec
from cap_harness.g1_real.registry import G1RealTaskRegistry
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.registry import YamRealTaskRegistry


class G1RealAdapter(YamRealAdapter):
    """Protocol-facing wrapper over a G1 station."""

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
