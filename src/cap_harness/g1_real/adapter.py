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

from cap_harness.contracts import ExecutionResult, RobotPlanningContext, TaskContext
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

    def _provenance(self) -> dict[str, Any]:
        # No calibration bundle on this station: the profile hash attributes an episode.
        config = self._env.config
        return {"station": config.station, "config_sha256": config.config_sha256}

    # -- hands -------------------------------------------------------------

    def set_hand_joints(self, arm: str, joints: list[float]) -> ExecutionResult:
        """Drive one Dex3 hand to explicit finger joints while both arms hold."""
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        side = self.resolve_arm(arm)
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
        # The recorder's action space is the gripper scalar, so the commanded
        # hand pose is recorded as its projection onto the open/closed line.
        gripper = float(np.asarray(self._env.get_observations(side)["gripper_pos"]).reshape(-1)[0])
        self._record(execution, gripper={side: gripper})
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
