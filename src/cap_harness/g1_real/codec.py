"""Joint-order and unit conversions between the G1 / GEAR-SONIC side and the harness contracts.

SONIC's planner stream wants a 17-dof upper-body vector (waist yaw/roll/pitch, then
left/right arm joints interleaved per joint) in IsaacLab order, while the robot state
arrives as a 29-dof MuJoCo-order body vector. Both layouts are resolved here so the
rest of the embodiment only sees (waist, left, right) arrays.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from cap_harness.contracts import ArmCommand, Observation, Pose, RobotState, TaskContext
from cap_harness.yam_real.codec import observation as yam_observation
from cap_harness.yam_real.codec import quat_xyzw_to_wxyz

EMBODIMENT = "g1_real"
ARMS = ("left", "right")
ARM_DOF = 7
BASE_FRAME = "torso"
UPPER_BODY_DOF = 17
HAND_DOF = 7
JOINT_NAMES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)

# Indices into the 29-dof MuJoCo body vector, in the planner's upper-body order.
MUJOCO_UPPER_BODY_INDICES = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
MUJOCO_ARM_SLICES = {"left": slice(15, 22), "right": slice(22, 29)}
MUJOCO_WAIST_SLICE = slice(12, 15)

# Positions of each part inside the 17-vector (waist first, then arms interleaved).
_Q17_WAIST = slice(0, 3)
_Q17_LEFT = (3, 5, 7, 9, 11, 13, 15)
_Q17_RIGHT = (4, 6, 8, 10, 12, 14, 16)


def upper_body_from_arms(waist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    q17 = np.empty(UPPER_BODY_DOF, dtype=np.float64)
    q17[_Q17_WAIST] = waist
    q17[list(_Q17_LEFT)] = left
    q17[list(_Q17_RIGHT)] = right
    return q17


def arms_from_upper_body(q17: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q17, dtype=np.float64).reshape(UPPER_BODY_DOF)
    return q[_Q17_WAIST].copy(), q[list(_Q17_LEFT)], q[list(_Q17_RIGHT)]


def upper_body_from_body_q(body_q29: Any) -> np.ndarray:
    q = np.asarray(body_q29, dtype=np.float64).reshape(29)
    return q[list(MUJOCO_UPPER_BODY_INDICES)]


def hand_scalar(hand_q: Any, hand_open: Any, hand_closed: Any) -> float:
    """Project a 7-dof hand pose onto the open-to-closed line; 1.0 is open, 0.0 closed."""
    opened = np.asarray(hand_open, dtype=np.float64)
    span = np.asarray(hand_closed, dtype=np.float64) - opened
    t = float(np.dot(np.asarray(hand_q, dtype=np.float64) - opened, span) / np.dot(span, span))
    return 1.0 - min(max(t, 0.0), 1.0)


def hand_pose(p: float, hand_open: Any, hand_closed: Any) -> np.ndarray:
    """Inverse of ``hand_scalar``: interpolate the 7-dof hand pose for openness ``p``."""
    opened = np.asarray(hand_open, dtype=np.float64)
    return opened + (1.0 - p) * (np.asarray(hand_closed, dtype=np.float64) - opened)


def heading_yaw_from_quat_wxyz(q: Any) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def robot_state(env: Any, *, timestamp_s: float | None = None) -> RobotState:
    """Build a shared ``RobotState`` from both arms of a G1 plant.

    The env already projects each Dex3 hand onto its open/closed line
    (``hand_scalar``), so ``gripper_positions`` is that scalar per side.
    """
    positions: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    poses: dict[str, Pose] = {}
    grippers: dict[str, float] = {}
    names: dict[str, tuple[str, ...]] = {}
    base_frame = env.config.base_frame
    for side in ARMS:
        obs = env.get_observations(side)
        positions[side] = np.asarray(obs["joint_pos"], dtype=np.float64).reshape(ARM_DOF)
        velocities[side] = np.asarray(obs["joint_vel"], dtype=np.float64).reshape(ARM_DOF)
        poses[side] = Pose(
            position=np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3),
            quaternion_wxyz=quat_xyzw_to_wxyz(obs["ee_quat"]),
            frame=base_frame,
        )
        grippers[side] = float(np.asarray(obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        names[side] = tuple(env.config.arms[side].joint_names)
    return RobotState(
        joint_positions=positions,
        joint_velocities=velocities,
        end_effector_poses=poses,
        gripper_positions=grippers,
        base_frame=base_frame,
        joint_names=names,
        timestamp_s=timestamp_s,
        embodiment=EMBODIMENT,
    )


def observation(env: Any, *, task_context: TaskContext | None = None) -> Observation:
    """Assemble a full harness observation; the camera half is shared with YAM."""
    return yam_observation(env, task_context=task_context, robot_state_fn=robot_state)


def arm_command_targets(action: Any) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    """Split a ``RobotAction`` into per-arm 7-joint targets and gripper scalars."""
    joints: dict[str, np.ndarray] = {}
    grippers: dict[str, float | None] = {}
    for side, command in action.arms.items():
        if not isinstance(command, ArmCommand):
            raise ValueError(f"arms[{side!r}] must be an ArmCommand")
        joints[side] = np.asarray(command.target, dtype=np.float64).reshape(-1)[:ARM_DOF]
        grippers[side] = (
            None if command.gripper_position is None else float(command.gripper_position)
        )
    return joints, grippers
