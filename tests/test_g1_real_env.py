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


def _lagging_env(tau_s=5.0):
    """An engaged env over a fake so slow the measured pose stays behind the chain."""
    cfg = load_g1_station_config("g1-lab")
    home = codec.upper_body_from_arms(
        cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )
    e = G1Env(cfg, FakeBodyClient(tau_s=tau_s, initial_q17=home), camera=SyntheticCamera())
    e.engage()
    return e


def _move_right_shoulder(e, delta):
    left = e.config.arms["left"].home_joints
    right = e.config.arms["right"].home_joints.copy()
    right[0] += delta
    result = e.execute_action_batch(
        YamActionBatch.joint_abs([0.0], [left], [right], source="test"), settle_s=0.0
    )
    assert result["success"] is True
    return codec.upper_body_from_arms(e.config.waist_targets, left, right)


def test_joint_only_batch_keeps_an_off_line_hand_verbatim(env):
    q = np.array([0.1, 0.2, 0.3, -0.4, -0.5, -0.6, -0.7])
    hands = env.config.hands["left"]
    projected = codec.hand_pose(
        codec.hand_scalar(q, hands.open, hands.closed), hands.open, hands.closed
    )
    assert not np.allclose(projected, q, atol=1e-3)
    assert env.set_hand_joints("left", q)["success"] is True
    time.sleep(0.1)
    right = env.get_observations("right")["joint_pos"].copy()
    right[0] += 0.05
    env.execute_action_batch(
        YamActionBatch.joint_abs(
            [0.0], [env.get_observations("left")["joint_pos"]], [right], source="test"
        ),
        settle_s=0.0,
    )
    np.testing.assert_allclose(env.body.commands[-1][2], q, atol=1e-9)
    np.testing.assert_allclose(env.last_command()[1], q, atol=1e-9)


def test_keep_alive_resumes_from_measured_after_an_outage():
    e = _lagging_env()
    try:
        target = _move_right_shoulder(e, 0.1)
        np.testing.assert_array_equal(e.last_command()[0], target)
        e.body.stale = True
        time.sleep(0.1)
        n = len(e.body.commands)
        time.sleep(0.05)
        assert len(e.body.commands) == n  # paused, nothing sent blind
        e.body.stale = False
        time.sleep(0.1)
        resumed = e.body.commands[n][0]
        measured = e.measured_command()[0]
        assert abs(resumed[4] - target[4]) > 0.05  # not the pre-outage target
        np.testing.assert_allclose(resumed, measured, atol=0.01)
    finally:
        e.close()


def test_batch_starting_at_the_chain_target_does_not_step_back():
    e = _lagging_env()
    try:
        _move_right_shoulder(e, 0.1)
        target = e.last_command()[0]  # the chain (float32-exact, as resolved rows are)
        assert abs(e.measured_command()[0][4] - target[4]) > 0.05  # measurement lags
        n = len(e.body.commands)
        _, left, right = codec.arms_from_upper_body(target)
        ahead = right.copy()
        ahead[0] += 0.01
        e.execute_action_batch(
            YamActionBatch.joint_abs([0.0, 0.1], [left, left], [right, ahead], source="test"),
            settle_s=0.0,
        )
        sent = [c[0] for c in e.body.commands[n:]]
        # The first tick samples at t = a few microseconds along the forward
        # segment, so it sits on the chain target, never below it.
        np.testing.assert_allclose(sent[0], target, atol=1e-6)
        assert all(q[4] >= target[4] for q in sent)
    finally:
        e.close()


def test_engage_runs_again_after_feedback_loss(env):
    assert env.body.start_count == 1
    assert env.body.seeded_hands is not None
    env.body.stale = True
    time.sleep(0.1)
    env.body.stale = False
    again = env.engage()
    assert again["already_engaged"] is False and env.body.start_count == 2
    assert env.engage()["already_engaged"] is True and env.body.start_count == 2
    assert env.keep_alive_running


def test_last_command_returns_copies(env):
    q17, lh, rh = env.last_command()
    q17[0] += 1.0
    assert env.last_command()[0][0] != q17[0]


def test_batch_publishing_first_after_an_outage_resumes_from_measured():
    """P1: the re-seed lives in the chokepoint, so a batch that beats the keep-alive
    to the first fresh publish starts from the measured pose, and the keep-alive
    afterwards carries the batch's final target instead of stepping back.
    """
    e = _lagging_env()
    try:
        target = _move_right_shoulder(e, 0.1)
        e.body.stale = True
        time.sleep(0.1)
        # Feedback returns while a batch is about to publish, before the
        # keep-alive's next wake: holding the stream as a batch does keeps it out.
        e._streaming = True
        e.body.stale = False
        n = len(e.body.commands)
        measured = e.measured_command()[0]
        _, left, right = codec.arms_from_upper_body(measured)
        ahead = right.copy()
        ahead[0] += 0.02
        result = e.execute_action_batch(
            YamActionBatch.joint_abs([0.0, 0.1], [left, left], [right, ahead], source="test"),
            settle_s=0.0,
        )
        assert result["success"] is True
        sent = np.stack([c[0] for c in e.body.commands[n:]])
        assert abs(sent[0][4] - target[4]) > 0.05  # not the pre-outage target
        np.testing.assert_allclose(sent[0], measured, atol=1e-3)
        assert np.max(np.abs(np.diff(sent, axis=0))) <= 0.03 + 1e-9
        last = e.body.commands[-1][0]
        m = len(e.body.commands)
        time.sleep(0.1)
        np.testing.assert_array_equal(e.body.commands[m][0], last)
    finally:
        e.close()


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


def test_joint_only_batch_holds_a_hand_still_converging_on_its_command():
    """P2: consecutive scalar reads of a moving hand differ by more than the hold
    tolerance; the batch must still hold the chain hand, not project a stale read.
    """
    cfg = load_g1_station_config("g1-lab")
    home = codec.upper_body_from_arms(
        cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )
    e = G1Env(cfg, _SlowHands(tau_s=0.01, initial_q17=home), camera=SyntheticCamera())
    e.engage()
    try:
        q = np.array([0.1, 0.2, 0.3, -0.4, -0.5, -0.6, -0.7])
        assert e.set_hand_joints("left", q)["success"] is True
        hands = cfg.hands["left"]
        measured = e.get_observations("left")["gripper_pos"][0]
        assert abs(measured - codec.hand_scalar(q, hands.open, hands.closed)) > 0.01
        e.execute_action_batch(
            YamActionBatch.joint_abs(
                [0.0],
                [e.get_observations("left")["joint_pos"]],
                [e.get_observations("right")["joint_pos"]],
                source="test",
            ),
            settle_s=0.0,
        )
        np.testing.assert_allclose(e.body.commands[-1][2], q, atol=1e-9)
    finally:
        e.close()


def test_go_home_opens_every_hand_joint(env):
    """P3: thumb_0 is zero in both open and closed, so the scalar cannot see it."""
    thumb = np.zeros(7)
    thumb[0] = 0.3
    assert env.set_hand_joints("left", thumb)["success"] is True
    hands = env.config.hands["left"]
    assert codec.hand_scalar(thumb, hands.open, hands.closed) == 1.0
    assert env.go_home(duration_s=0.1, keep_grippers=False)["success"] is True
    np.testing.assert_array_equal(env.body.commands[-1][2], np.zeros(7))
    np.testing.assert_array_equal(env.body.commands[-1][3], np.zeros(7))
