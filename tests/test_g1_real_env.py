import time

import numpy as np
import pytest

from cap_harness.g1_real import codec
from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.yam_real.action import YamActionBatch
from cap_harness.yam_real.cameras import SyntheticCamera


@pytest.fixture
def env():
    cfg = load_g1_station_config("g1-lab")
    home = codec.upper_body_from_arms(
        cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )
    fake = FakeBodyClient(tau_s=0.01, initial_q17=home)
    e = G1Env(cfg, fake, camera=SyntheticCamera())
    e.engage()
    yield e
    e.close()


def test_step_clamp_stretches_a_jump(env):
    start = env.get_observations("right")["joint_pos"].copy()
    target = start.copy()
    target[0] += 0.5
    env.execute_action_batch(
        YamActionBatch.joint_abs(
            [0.0], [env.get_observations("left")["joint_pos"]], [target], source="test"
        ),
        settle_s=0.0,
    )
    # right shoulder_pitch is SONIC-17 index 4
    steps = np.diff([c[0][4] for c in env.body.commands])
    assert np.max(np.abs(steps)) <= 0.03 + 1e-9
    assert len(env.body.commands) >= 16


def test_keep_alive_resends_last_target_between_batches(env):
    n = len(env.body.commands)
    time.sleep(0.25)
    assert len(env.body.commands) - n >= 8
    assert env.keep_alive_running


def test_stale_feedback_aborts_a_batch(env):
    env.body.stale = True
    n = len(env.body.commands)
    result = env.execute_action_batch(
        YamActionBatch.joint_abs([0.0, 0.5], np.zeros((2, 7)), np.zeros((2, 7)), source="test")
    )
    assert result["success"] is False
    assert "stale" in result["reason"]
    assert len(env.body.commands) == n


def test_batch_before_engage_is_refused():
    cfg = load_g1_station_config("g1-lab")
    fake = FakeBodyClient()
    e = G1Env(cfg, fake, camera=SyntheticCamera())
    result = e.execute_action_batch(
        YamActionBatch.joint_abs([0.0], np.zeros(7), np.zeros(7), source="test")
    )
    assert result["success"] is False
    assert "not engaged" in result["reason"]
    assert fake.commands == []
    e.close()


class _RaisesOnce(FakeBodyClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raise_next = False

    def command_upper_body(self, *args, **kwargs):
        if self.raise_next:
            self.raise_next = False
            raise OSError("socket closed")
        super().command_upper_body(*args, **kwargs)


def test_keep_alive_survives_a_body_client_error():
    cfg = load_g1_station_config("g1-lab")
    fake = _RaisesOnce()
    e = G1Env(cfg, fake, camera=SyntheticCamera())
    e.engage()
    fake.raise_next = True
    time.sleep(0.1)
    n = len(fake.commands)
    time.sleep(0.1)
    assert fake.raise_next is False
    assert e.keep_alive_running
    assert len(fake.commands) > n
    e.close()


def test_out_of_range_arm_target_is_refused(env):
    left = env.get_observations("left")["joint_pos"]
    right = env.get_observations("right")["joint_pos"].copy()
    right[1] = env.config.arms["right"].upper[1] + 0.5
    before = env.body.commands[-1][0]
    n = len(env.body.commands)
    result = env.execute_action_batch(
        YamActionBatch.joint_abs([0.0], [left], [right], source="test"), settle_s=0.0
    )
    assert result["success"] is False
    assert "outside the configured limits" in result["reason"]
    assert "right joint 1" in result["reason"]
    # only keep-alive re-sends of the previous target may have gone out
    assert all(np.array_equal(c[0], before) for c in env.body.commands[n:])
    assert env._action_input_seq["right"] == 0


def test_go_home_interpolates_from_measured_to_home(env):
    cfg = env.config
    away = cfg.arms["right"].home_joints.copy()
    away[0] += 0.1
    env.execute_action_batch(
        YamActionBatch.joint_abs([0.0], [cfg.arms["left"].home_joints], [away], source="test"),
        settle_s=0.0,
    )
    time.sleep(0.1)
    measured = env.measured_command()[0]
    home = codec.upper_body_from_arms(
        cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )
    assert not np.allclose(measured, home, atol=1e-3)
    n = len(env.body.commands)
    result = env.go_home(duration_s=0.1)
    assert result["success"] is True
    np.testing.assert_allclose(env.body.commands[n][0], measured, atol=1e-3)
    np.testing.assert_array_equal(env.body.commands[-1][0], home)


def test_gripper_scalar_round_trips_through_the_hand(env):
    env.set_gripper("right", 0.3)
    time.sleep(0.1)
    assert abs(env.get_observations("right")["gripper_pos"][0] - 0.3) < 0.05
    sent = env.body.commands[-1][3]
    hands = env.config.hands["right"]
    np.testing.assert_allclose(sent, codec.hand_pose(0.3, hands.open, hands.closed), atol=1e-6)


def test_joint_only_batch_holds_the_measured_hand(env):
    env.set_gripper("right", 0.2)
    time.sleep(0.1)
    env.execute_action_batch(
        YamActionBatch.joint_abs(
            [0.0],
            [env.get_observations("left")["joint_pos"]],
            [env.get_observations("right")["joint_pos"]],
            source="test",
        ),
        settle_s=0.0,
    )
    hands = env.config.hands["right"]
    sent = codec.hand_scalar(env.body.commands[-1][3], hands.open, hands.closed)
    assert abs(sent - 0.2) < 0.05


def test_set_hand_joints_overrides_scalar_mapping(env):
    q = np.array([0.1, 0.2, 0.3, -0.4, -0.5, -0.6, -0.7])
    env.set_hand_joints("left", q)
    np.testing.assert_allclose(env.body.commands[-1][2], q, atol=1e-6)


def test_engage_is_idempotent_and_refuses_stale_feedback():
    cfg = load_g1_station_config("g1-lab")
    fake = FakeBodyClient()
    e = G1Env(cfg, fake, camera=SyntheticCamera())
    fake.stale = True
    with pytest.raises(RuntimeError, match="stale"):
        e.engage()
    assert fake.start_count == 0
    assert not e.keep_alive_running
    fake.stale = False
    first = e.engage()
    second = e.engage()
    assert fake.start_count == 1
    assert first["already_engaged"] is False
    assert second["already_engaged"] is True
    e.close()
    assert not e.keep_alive_running


def test_robot_state_is_seven_joint_torso_frame(env):
    state = codec.robot_state(env)
    assert state.embodiment == "g1_real"
    assert state.base_frame == "torso"
    assert state.joint_positions["left"].shape == (7,)
    assert state.joint_names["right"] == env.config.arms["right"].joint_names
    obs = codec.observation(env)
    assert obs.robot_state.embodiment == "g1_real"
    assert set(obs.cameras) == {env.config.camera.role}


def test_control_contract_names_the_planner_stream(env):
    contract = env.control_contract()
    assert contract["version"] == "g1-sonic-v1-planner-stream"
    assert contract["command_stream_hz"] == 50.0
    assert contract["station"] == "g1-lab"
