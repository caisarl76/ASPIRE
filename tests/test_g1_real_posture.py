import numpy as np
import pytest

from cap_harness.contracts import Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.g1_real import codec
from cap_harness.g1_real.adapter import G1RealAdapter
from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.g1_real.posture import G1Posture
from cap_harness.yam_real.cameras import SyntheticCamera

CFG = load_g1_station_config("g1-222")


def _posture() -> G1Posture:
    return G1Posture(
        {s: CFG.hands[s].joint_names for s in ("left", "right")},
        {s: CFG.hands[s].closed for s in ("left", "right")},
    )


def _body_q(left, right) -> np.ndarray:
    q = np.zeros(29)
    q[codec.MUJOCO_ARM_SLICES["left"]] = left
    q[codec.MUJOCO_ARM_SLICES["right"]] = right
    return q


def test_palms_face_the_body_midline_in_the_handshake_pose():
    # zero arm joints: forearms forward, thumbs up (the palm-touch handshake pose)
    posture = _posture()
    posture.set_state(np.zeros(29))
    summary = posture.summary()
    assert summary["frame"] == "torso"
    assert summary["right"]["palm_normal"][1] > 0.9  # right palm faces +y
    assert summary["left"]["palm_normal"][1] < -0.9  # left palm faces -y
    assert summary["right"]["finger_direction"][0] > 0.9
    # palm center sits between the wrist and the grasp site
    wrist, palm = np.array(summary["right"]["wrist"]), np.array(summary["right"]["palm_center"])
    assert 0.09 < palm[0] - wrist[0] < 0.13
    for leg in ("left_leg", "right_leg"):
        assert summary[leg]["knee"][2] < summary[leg]["hip"][2] < 0.0


def test_home_pose_keeps_the_hands_off_the_legs():
    posture = _posture()
    posture.set_state(_body_q(CFG.arms["left"].home_joints, CFG.arms["right"].home_joints))
    for side in ("left", "right"):
        clearance, *_ = posture.leg_clearance(side)
        assert clearance > 0.03, side


def _toward_the_thigh() -> np.ndarray:
    """A right-arm configuration whose hand presses into the right thigh (searched)."""
    posture = _posture()
    left = CFG.arms["left"].home_joints
    best, best_q = np.inf, None
    for pitch in np.linspace(-0.3, 1.2, 16):
        for elbow in np.linspace(-1.0, 1.0, 11):
            q = np.array([pitch, -0.2, 0.0, elbow, 0.0, 0.0, 0.0])
            posture.set_state(_body_q(left, q))
            clearance, *_ = posture.leg_clearance("right")
            if clearance < best:
                best, best_q = clearance, q
    assert best < 0.01
    return best_q


def _adapter(right_start) -> G1RealAdapter:
    q17 = codec.upper_body_from_arms(CFG.waist_targets, CFG.arms["left"].home_joints, right_start)
    env = G1Env(CFG, FakeBodyClient(tau_s=0.01, initial_q17=q17), camera=SyntheticCamera())
    env.engage()
    return G1RealAdapter(env, allow_physical_motion=True)


def _trajectory(adapter: G1RealAdapter, start, goal) -> Trajectory:
    rows = np.linspace(np.asarray(start), np.asarray(goal), 20)
    state = adapter.get_robot_state()
    return Trajectory(
        joint_positions=rows,
        dt_s=0.05,
        joint_names=state.joint_names["right"],
        planner="test",
        collision_aware=False,
        expected_start=state,
        arm="right",
        embodiment="g1_real",
    )


def test_leg_guard_refuses_a_trajectory_into_the_thigh():
    home = np.asarray(CFG.arms["right"].home_joints)
    adapter = _adapter(home)
    try:
        result = adapter.execute_trajectory(_trajectory(adapter, home, _toward_the_thigh()))
        assert not result.ok and result.error.code == ErrorCode.PLANNING_FAILED
        assert "nothing moved" in result.error.message
        assert result.error.details["leg_clearance_m"] < 0.03
        np.testing.assert_allclose(
            adapter.native_env.get_observations("right")["joint_pos"], home, atol=1e-3
        )
    finally:
        adapter.close()


def test_leg_guard_lets_a_tight_start_move_away():
    tight = _toward_the_thigh()
    home = np.asarray(CFG.arms["right"].home_joints)
    adapter = _adapter(tight)
    try:
        result = adapter.execute_trajectory(_trajectory(adapter, tight, home))
        assert result.error is None or result.error.code != ErrorCode.PLANNING_FAILED
    finally:
        adapter.close()


def test_posture_tool_reads_the_feedback():
    adapter = _adapter(np.asarray(CFG.arms["right"].home_joints))
    try:
        posture = adapter.get_posture()
        assert posture["ok"] and posture["right"]["leg_clearance_m"] > 0.03
        assert set(posture) >= {"left", "right", "left_leg", "right_leg", "pelvis"}
    finally:
        adapter.close()


@pytest.mark.parametrize("station", ["g1-lab", "g1-222", "g1-sim"])
def test_profiles_turn_the_leg_guard_on(station):
    assert load_g1_station_config(station).controller.leg_clearance_m == pytest.approx(0.03)
