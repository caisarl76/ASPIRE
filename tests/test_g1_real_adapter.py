"""``G1RealAdapter`` on the fake SONIC body: identity, interlock, hands, residuals."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.g1_real import codec
from cap_harness.g1_real.adapter import G1RealAdapter
from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.yam_real.action import YamActionBatch
from cap_harness.yam_real.cameras import SyntheticCamera


def _home_q17(cfg):
    return codec.upper_body_from_arms(
        cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )


def _engaged_adapter(cfg, q17):
    env = G1Env(cfg, FakeBodyClient(tau_s=0.01, initial_q17=q17), camera=SyntheticCamera())
    env.engage()
    return G1RealAdapter(env, allow_physical_motion=True)


@pytest.fixture
def adapter():
    cfg = load_g1_station_config("g1-lab")
    a = _engaged_adapter(cfg, _home_q17(cfg))
    yield a
    a.close()


def _right_trajectory(adapter, delta0: float, steps: int = 25) -> Trajectory:
    state = adapter.get_robot_state()
    start = state.joint_positions["right"]
    target = start + np.array([delta0, 0, 0, 0, 0, 0, 0])
    return Trajectory(
        joint_positions=np.linspace(start, target, steps),
        dt_s=0.02,
        joint_names=list(adapter.native_env.config.arms["right"].joint_names),
        planner="test",
        collision_aware=False,
        expected_start=state,
        arm="right",
        embodiment="g1_real",
    )


def test_identity(adapter):
    assert adapter.embodiment == "g1_real"
    assert adapter.dof == 7
    assert adapter.registry_prefix == "g1_real"
    assert adapter.arms == ("left", "right")
    ctx = adapter.get_planning_context()
    assert ctx.embodiment == "g1_real" and ctx.model == "g1_real"
    assert ctx.end_effector_links["right"] == "right_grasp"
    assert len(ctx.joint_names["left"]) == 7
    np.testing.assert_array_equal(ctx.base_transforms["left"], np.eye(4))


def test_unauthorized_adapter_refuses_hands_and_motion():
    cfg = load_g1_station_config("g1-lab")
    env = G1Env(cfg, FakeBodyClient(), camera=SyntheticCamera())
    a = G1RealAdapter(env)
    assert a.set_hand_joints("left", [0.0] * 7).error.code == ErrorCode.SAFETY_INTERLOCK
    assert a.go_home().error.code == ErrorCode.SAFETY_INTERLOCK
    assert env.body.commands == []
    assert a.command_rpc_count == 0
    # Reads are not motion.
    assert set(a.get_hand_state()) == {"left", "right"}


def test_go_home_reports_home_verified(adapter):
    result = adapter.go_home()
    assert result.ok
    assert result.diagnostics["home_verified"]
    assert result.diagnostics["grippers_open"] == {"left": True, "right": True}


def test_trajectory_residual_fails_when_the_body_lags(adapter):
    adapter.native_env.body.tau_s = 5.0  # arm barely moves
    result = adapter.execute_trajectory(_right_trajectory(adapter, 0.4))
    assert not result.ok
    assert result.error.code == ErrorCode.EXECUTION_FAILED
    assert "did not track" in result.error.message


def test_trajectory_holds_an_uncommanded_arm_inside_its_limits():
    """R13: a hold row built from a measured pose past a limit is clamped, not refused."""
    cfg = load_g1_station_config("g1-lab")
    left = cfg.arms["left"].home_joints.copy()
    left[0] = cfg.arms["left"].upper[0] + 0.01
    q17 = codec.upper_body_from_arms(cfg.waist_targets, left, cfg.arms["right"].home_joints)
    adapter = _engaged_adapter(cfg, q17)
    try:
        result = adapter.execute_trajectory(_right_trajectory(adapter, 0.05))
    finally:
        adapter.close()
    assert result.ok, result.error


def test_hand_tools_round_trip(adapter):
    q = [0.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    result = adapter.set_hand_joints("left", q)
    assert result.ok, result.error
    assert result.steps_executed == 1
    assert adapter.command_rpc_count == 1
    time.sleep(0.1)
    np.testing.assert_allclose(adapter.get_hand_state()["left"], q, atol=0.05)
    assert len(adapter.get_hand_state()["right"]) == 7


def test_hand_tool_maps_a_plant_refusal_to_an_error(adapter):
    result = adapter.set_hand_joints("right", [9.0] * 7)
    assert not result.ok
    assert result.error.code == ErrorCode.EXECUTION_FAILED
    assert "outside the profile limits" in result.error.message


def test_hand_tool_refuses_an_unknown_arm_before_any_rpc(adapter):
    published = len(adapter.native_env.body.commands)
    result = adapter.set_hand_joints("wrist", [0.0] * 7)
    assert not result.ok
    assert result.error.code == ErrorCode.INVALID_REQUEST
    assert adapter.command_rpc_count == 0
    assert len(adapter.native_env.body.commands) == published


class CapturingObserver:
    def __init__(self) -> None:
        self.actions: list[RobotAction] = []

    def before_step(self, action: RobotAction) -> None:
        pass

    def after_step(self, action: RobotAction, result: object) -> None:
        self.actions.append(action)

    def on_reset(self, observation: object, metadata: object) -> None:
        pass


def test_hand_tool_records_the_commanded_openness_not_the_measured_one():
    """A closed-hand command must be recorded as closed, even while the hand is still open."""
    cfg = load_g1_station_config("g1-lab")
    observer = CapturingObserver()
    env = G1Env(
        cfg, FakeBodyClient(tau_s=0.01, initial_q17=_home_q17(cfg)), camera=SyntheticCamera()
    )
    env.engage()
    adapter = G1RealAdapter(env, allow_physical_motion=True, run_observer=observer)
    try:
        q = [0.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        assert adapter.set_hand_joints("left", q).ok
    finally:
        adapter.close()
    hand = cfg.hands["left"]
    expected = codec.hand_scalar(q, hand.open, hand.closed)
    assert observer.actions
    recorded = observer.actions[-1].arms["left"].gripper_position
    assert recorded == pytest.approx(expected)
    assert recorded < 0.5, f"recorded the measured (open) hand, {recorded}, not the command"


def test_metadata(adapter):
    task = adapter.get_task_metadata()
    assert task["embodiment"] == "g1_real"
    assert task["hands"] == ["left", "right"]
    assert task["base_frame"] == "torso"
    assert task["cameras"] == ["ego_view"]
    assert task["dof"] == 7
    controller = adapter.get_controller_metadata()
    assert controller["control_contract_version"] == "g1-sonic-v1-planner-stream"
    assert len(controller["action_lower_bounds"]) == 14
    assert len(controller["action_upper_bounds"]) == 14
    assert controller["arm_names"] == ["left", "right"]
    assert controller["physical_motion_authorized"] is True


def test_reset_homes_and_registers_a_g1_task(adapter):
    observation = adapter.reset("observe_station")
    assert observation.robot_state.embodiment == "g1_real"
    context = adapter.get_task_context()
    assert context.family == "g1_real"
    assert context.metadata["station"] == "g1-lab"


def test_hand_tools_are_registered_as_g1_extensions(adapter):
    from cap_harness.api import CapApi
    from cap_harness.registry import G1_REAL_PUBLIC_TOOL_NAMES, ToolRegistry

    registry = CapApi(adapter, default_camera="ego_view").register_tools(
        ToolRegistry(public_extension_allowlist=G1_REAL_PUBLIC_TOOL_NAMES)
    )
    names = {spec.name for spec in registry.all_specs(public_only=True)}
    assert {
        "g1_real.set_hand_joints",
        "g1_real.get_hand_state",
        "g1_real.get_controller_metadata",
        "g1_real.get_task_metadata",
    } <= names


class _SlowHands(FakeBodyClient):
    """Hands that close a tenth of the remaining gap per observation read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slow = {side: np.zeros(7) for side in ("left", "right")}

    def get_observations(self):
        raw = super().get_observations()
        for side in ("left", "right"):
            self._slow[side] += 0.1 * (self._hand_targets[side] - self._slow[side])
            raw[f"{side}_hand_q"] = self._slow[side].copy()
        return raw


def test_step_holds_a_hand_still_converging_on_its_command():
    """P2 on the adapter path: the hold column is the chain scalar, not a stale read."""
    cfg = load_g1_station_config("g1-lab")
    env = G1Env(cfg, _SlowHands(tau_s=0.01, initial_q17=_home_q17(cfg)), camera=SyntheticCamera())
    env.engage()
    adapter = G1RealAdapter(env, allow_physical_motion=True)
    try:
        q = [0.1, 0.2, 0.3, -0.4, -0.5, -0.6, -0.7]
        assert adapter.set_hand_joints("left", q).ok
        state = adapter.get_robot_state()
        hands = cfg.hands["left"]
        assert (
            abs(state.gripper_positions["left"] - codec.hand_scalar(q, hands.open, hands.closed))
            > 0.01
        )
        target = state.joint_positions["right"] + np.array([0.02, 0, 0, 0, 0, 0, 0])
        action = RobotAction({"right": ArmCommand("joint_position", target, embodiment="g1_real")})
        assert adapter.step(action).ok
        np.testing.assert_allclose(env.body.commands[-1][2], q, atol=1e-9)
    finally:
        adapter.close()


def test_single_arm_trajectory_holds_the_idle_arm_at_its_chain_target():
    """P4: the idle arm holds the chain, not its lagging measurement."""
    cfg = load_g1_station_config("g1-lab")
    adapter = _engaged_adapter(cfg, _home_q17(cfg))
    env = adapter.native_env
    try:
        env.body.tau_s = 5.0  # the measured pose stays behind the chain
        left = cfg.arms["left"].home_joints.copy()
        left[0] += 0.05
        result = env.execute_action_batch(
            YamActionBatch.joint_abs([0.0], [left], [cfg.arms["right"].home_joints], source="test"),
            settle_s=0.0,
        )
        assert result["success"] is True
        np.testing.assert_allclose(codec.arms_from_upper_body(env.last_command()[0])[1], left)
        assert abs(env.get_observations("left")["joint_pos"][0] - left[0]) > 0.03
        adapter.execute_trajectory(_right_trajectory(adapter, 0.05))
        np.testing.assert_array_equal(codec.arms_from_upper_body(env.last_command()[0])[1], left)
    finally:
        adapter.close()


def test_metadata_caps_planning_steps_and_batches_blend_from_the_chain(adapter, monkeypatch):
    # 0.5 rad/s over a 20 ms control period
    assert adapter.get_controller_metadata()["max_planning_step_rad"] == pytest.approx(0.01)
    seen = {}
    env = adapter.native_env
    real = env.execute_action_batch

    def capture(batch, **kwargs):
        seen.update(kwargs)
        return real(batch, **kwargs)

    monkeypatch.setattr(env, "execute_action_batch", capture)
    adapter.execute_trajectory(_right_trajectory(adapter, 0.05))
    # The profile keeps the blend at 0: the stream starts at the chain itself.
    assert seen["start_interp_s"] == pytest.approx(0.0)
    assert adapter.trajectory_start_interp_s == 0.0
