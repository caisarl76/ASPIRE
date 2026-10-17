import numpy as np

from cap_harness.contracts import JOINT_DIMENSIONS, Pose, RobotState
from cap_harness.registry import G1_REAL_PUBLIC_TOOL_NAMES, ToolRegistry


def test_g1_real_joint_dimension_is_seven():
    assert JOINT_DIMENSIONS["g1_real"] == 7


def test_robot_state_accepts_seven_joint_g1_arms():
    pose = Pose(position=np.zeros(3), quaternion_wxyz=np.array([1.0, 0, 0, 0]), frame="torso")
    state = RobotState(
        joint_positions={"left": np.zeros(7), "right": np.zeros(7)},
        joint_velocities={"left": np.zeros(7), "right": np.zeros(7)},
        end_effector_poses={"left": pose, "right": pose},
        gripper_positions={"left": 1.0, "right": 1.0},
        base_frame="torso",
        embodiment="g1_real",
    )
    assert state.joint_positions["left"].shape == (7,)


def test_g1_real_extension_names_are_approved():
    assert G1_REAL_PUBLIC_TOOL_NAMES == frozenset(
        {
            "g1_real.get_task_metadata",
            "g1_real.get_controller_metadata",
            "g1_real.set_hand_joints",
            "g1_real.get_hand_state",
        }
    )
    ToolRegistry(public_extension_allowlist=G1_REAL_PUBLIC_TOOL_NAMES)
