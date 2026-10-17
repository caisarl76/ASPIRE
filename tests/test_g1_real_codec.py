import numpy as np
from cap_harness.g1_real import codec
from g1_real_wire import decode_packed
from cap_harness.g1_real.sonic.zmq_planner_sender import build_planner_message

# Copied from the fork: gear_sonic/envs/manager_env/robots/g1.py
G1_ISAACLAB_TO_MUJOCO_DOF = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]
# Upstream C++: upper_body_joint_isaaclab_order_in_isaaclab_index
UPPER_BODY_ISAACLAB = [2,5,8,11,12,15,16,19,20,21,22,23,24,25,26,27,28]


def test_mujoco_upper_body_indices_match_the_isaaclab_table():
    # the table is indexed by mujoco joint and yields its isaaclab index, so invert it
    expected = [G1_ISAACLAB_TO_MUJOCO_DOF.index(i) for i in UPPER_BODY_ISAACLAB]
    assert list(codec.MUJOCO_UPPER_BODY_INDICES) == expected


def test_arm_split_round_trips():
    body_q = np.arange(29.0)
    q17 = codec.upper_body_from_body_q(body_q)
    waist, left, right = codec.arms_from_upper_body(q17)
    np.testing.assert_array_equal(left, body_q[15:22])
    np.testing.assert_array_equal(right, body_q[22:29])
    np.testing.assert_array_equal(waist, body_q[12:15])
    np.testing.assert_array_equal(codec.upper_body_from_arms(waist, left, right), q17)


def test_hand_scalar_is_inverse_of_hand_pose():
    opened = np.zeros(7); closed = np.array([0,0,1.75,-1.57,-1.75,-1.57,-1.75])
    for p in (0.0, 0.25, 1.0):
        assert abs(codec.hand_scalar(codec.hand_pose(p, opened, closed), opened, closed) - p) < 1e-9
    assert codec.hand_scalar(closed * 1.3, opened, closed) == 0.0  # clamped


def test_heading_yaw_of_identity_is_zero():
    assert codec.heading_yaw_from_quat_wxyz([1, 0, 0, 0]) == 0.0
    assert abs(codec.heading_yaw_from_quat_wxyz([np.cos(0.4), 0, 0, np.sin(0.4)]) - 0.8) < 1e-9


def test_wire_decoder_reads_a_planner_message():
    msg = build_planner_message(0, [0,0,0], [1,0,0], upper_body_position=list(range(17)),
                                upper_body_velocity=[0.0]*17, left_hand_position=[1.0]*7,
                                right_hand_position=[2.0]*7)
    fields = decode_packed(msg, b"planner")
    assert fields["mode"].tolist() == [0]
    np.testing.assert_allclose(fields["upper_body_position"], np.arange(17, dtype=np.float32))
    # the wire field is named *_hand_joints even though the builder kwarg is *_hand_position
    assert fields["right_hand_joints"].shape == (7,)
