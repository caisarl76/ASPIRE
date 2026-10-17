"""Torso-rooted G1 model and mink kinematics.

The generated MuJoCo model has no legs and no meshes; every public pose from
``G1Kinematics`` is in the ``torso_link`` frame, and IK moves only the fourteen
arm joints while the waist and finger joints hold their seeded values.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.kinematics import G1Kinematics

MODEL = Path("src/cap_harness/g1_real/description/g1_upper_body.xml")


def test_model_has_no_legs_and_both_grasp_sites():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert not any("hip" in n or "knee" in n or "ankle" in n for n in names)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site") >= 0


def test_fk_at_home_is_in_front_of_the_torso_and_mirrored():
    cfg = load_g1_station_config("g1-lab")
    kin = G1Kinematics(MODEL)
    lp, _, rp, _ = kin.forward_kinematics(
        cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )
    assert lp[0] > 0.15 and rp[0] > 0.15  # forward of torso_link
    # Unitree's model places the shoulders at y=+0.10022 / -0.10021, so the
    # mirror is exact only to 1e-5; the sign convention is what matters here.
    assert abs(lp[1] + rp[1]) < 1e-4 and lp[1] > 0  # left arm on +Y


def test_ik_reaches_a_displaced_home_pose():
    cfg = load_g1_station_config("g1-lab")
    kin = G1Kinematics(MODEL)
    home_l, home_r = cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    lp, lq, rp, rq = kin.forward_kinematics(home_l, home_r)
    kin.seed(home_l, home_r)
    l7, r7 = kin.inverse_kinematics(lp, lq, rp + np.array([0.05, 0, 0.05]), rq, seeded=True)
    _, _, rp2, _ = kin.forward_kinematics(l7, r7)
    assert np.linalg.norm(rp2 - (rp + [0.05, 0, 0.05])) < 0.01
    assert np.allclose(l7, home_l, atol=0.02)  # untouched arm holds


def test_ik_provider_accepts_seven_joint_state():
    from cap_harness.contracts import Pose, RobotState
    from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider

    cfg = load_g1_station_config("g1-lab")
    kin = G1Kinematics(MODEL)
    home = {s: cfg.arms[s].home_joints for s in ("left", "right")}
    lp, lq, rp, rq = kin.forward_kinematics(home["left"], home["right"])

    def pose(p, q):
        return Pose(position=p, quaternion_wxyz=np.array([q[3], q[0], q[1], q[2]]), frame="torso")

    state = RobotState(
        joint_positions=home,
        joint_velocities={s: np.zeros(7) for s in home},
        end_effector_poses={"left": pose(lp, lq), "right": pose(rp, rq)},
        gripper_positions={"left": 1.0, "right": 1.0},
        base_frame="torso",
        embodiment="g1_real",
    )
    result = YamKinematicsIKProvider(kin, dof=7).solve_ik(
        pose(rp + [0.03, 0, 0], rq), state, arm="right"
    )
    assert result.ok and result.joint_positions.shape == (7,)
