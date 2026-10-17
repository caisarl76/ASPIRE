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


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis, angle):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def _link_poses_in_torso(urdf_path: Path, home: dict[str, float]) -> dict[str, np.ndarray]:
    """Link frames (4x4) in torso_link at the given joint angles, walking URDF origins."""
    import xml.etree.ElementTree as ET

    children = {}
    for joint in ET.parse(urdf_path).getroot().iter("joint"):
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = _rot(rpy), xyz
        if joint.get("type") == "revolute":
            axis = [float(v) for v in joint.find("axis").get("xyz").split()]
            m = np.eye(4)
            m[:3, :3] = _axis_angle(axis, home[joint.get("name")])
            t = t @ m
        children.setdefault(joint.find("parent").get("link"), []).append(
            (joint.find("child").get("link"), t)
        )
    poses = {"torso_link": np.eye(4)}
    stack = ["torso_link"]
    while stack:
        parent = stack.pop()
        for child, t in children.get(parent, []):
            poses[child] = poses[parent] @ t
            stack.append(child)
    return poses


def test_g1_real_home_pose_has_no_unignored_sphere_overlap():
    _, kin = _kinematics()
    cspace = kin["cspace"]
    poses = _link_poses_in_torso(
        Path(kin["urdf_path"]),
        dict(zip(cspace["joint_names"], cspace["default_joint_position"], strict=True)),
    )
    ignored = {frozenset((a, b)) for a, bs in kin["self_collision_ignore"].items() for b in bs}
    spheres = [
        (link, (poses[link] @ [*s["center"], 1.0])[:3], s["radius"])
        for link, items in kin["collision_spheres"].items()
        for s in items
    ]
    buffers = kin["self_collision_buffer"]
    clashes = [
        f"{a}/{b}: {np.linalg.norm(ca - cb):.3f} < {ra + rb + max(buffers[a], buffers[b]):.3f}"
        for i, (a, ca, ra) in enumerate(spheres)
        for b, cb, rb in spheres[i + 1 :]
        if a != b
        and frozenset((a, b)) not in ignored
        and np.linalg.norm(ca - cb) < ra + rb + max(buffers[a], buffers[b])
    ]
    assert not clashes
