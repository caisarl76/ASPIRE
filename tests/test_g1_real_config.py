from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from cap_harness.g1_real.config import load_g1_station_config

FORK_URDF = Path.home() / "work/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_dual_arm.urdf"


def test_profile_loads_with_seven_joint_arms_and_torso_base():
    cfg = load_g1_station_config("g1-lab")
    assert cfg.base_frame == "torso" and cfg.arms["left"].home_joints.shape == (7,)
    assert cfg.arm_aliases["primary"] == "right"
    assert cfg.hands["right"].closed[2] == -1.75 and cfg.sonic.feedback_port == 5557
    assert cfg.model_xml.is_absolute() and cfg.model_urdf.name == "g1_dual_arm.urdf"
    assert set(cfg.joint_limits_lower) == {"left", "right"}
    assert cfg.home_joint_positions["left"].shape == (7,)


def test_camera_transform_is_optical_frame_z_forward():
    # Optical Z points mostly forward (+X torso) and tilts down; optical X is right (-Y torso).
    base_from_camera = load_g1_station_config("g1-lab").camera.base_from_camera
    assert base_from_camera[0, 2] > 0.5 and base_from_camera[2, 2] < 0.0
    assert base_from_camera[1, 0] == pytest.approx(-1.0)


def test_profile_limits_match_the_fork_urdf():
    if not FORK_URDF.is_file():
        pytest.skip("fork URDF not present")
    cfg = load_g1_station_config("g1-lab")
    limits = {
        j.get("name"): j.find("limit")
        for j in ET.parse(FORK_URDF).getroot().iter("joint")
        if j.find("limit") is not None
    }
    for side in ("left", "right"):
        for i, name in enumerate(cfg.arms[side].joint_names):
            assert float(limits[name].get("lower")) == cfg.arms[side].lower[i]
            assert float(limits[name].get("upper")) == cfg.arms[side].upper[i]


def test_unknown_station_raises():
    with pytest.raises(FileNotFoundError):
        load_g1_station_config("nope")


def test_profile_declares_sonic_tracking_knobs():
    controller = load_g1_station_config("g1-lab").controller
    assert controller.pose_correction_rounds == 3
    assert controller.plan_max_joint_velocity_rad_s == pytest.approx(0.5)
    assert controller.trajectory_start_interp_s == 0.0  # the stream starts at the chain itself
