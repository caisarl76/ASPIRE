from pathlib import Path

import numpy as np
import yaml

from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.providers.curobo import service


def _kinematics() -> tuple[Path, dict]:
    path = Path(service._g1_real_robot_config())
    return path, yaml.safe_load(path.read_text())["robot_cfg"]["kinematics"]


def test_g1_real_config_is_v2_torso_rooted_with_both_grasp_frames():
    path, kin = _kinematics()
    assert kin["format_version"] == 2.0 and kin["base_link"] == "torso_link"
    assert kin["tool_frames"] == ["left_grasp", "right_grasp"]
    assert len(kin["cspace"]["joint_names"]) == 14
    assert Path(kin["urdf_path"]).is_file()
    assert service._g1_real_robot_config() == str(path)  # cached


def test_g1_real_config_matches_the_station_profile_and_urdf():
    _, kin = _kinematics()
    cfg = load_g1_station_config("g1-lab")
    left, right = cfg.arms["left"], cfg.arms["right"]
    assert kin["cspace"]["joint_names"] == [*left.joint_names, *right.joint_names]
    np.testing.assert_allclose(
        kin["cspace"]["default_joint_position"], [*left.home_joints, *right.home_joints]
    )
    urdf = Path(kin["urdf_path"]).read_text()
    assert all(f'name="{n}"' in urdf for n in kin["cspace"]["joint_names"])
    spheres = kin["collision_spheres"]
    assert sum(len(v) for v in spheres.values()) < 60
    assert set(kin["self_collision_buffer"]) == set(spheres)
    assert all(f'<link name="{link}"' in urdf for link in spheres)


def test_robot_config_dispatches_g1_real():
    request = service.BaseRequest.model_construct(model="g1_real")
    assert service._robot_config(request) == service._g1_real_robot_config()
