"""SONIC MuJoCo sim for ASPIRE: the fork's G1 loop plus a case, a bottle and an RGB-D head camera.

Run with the GR00T-WholeBodyControl fork's sim venv, from the fork root (scripts/g1_sim.sh
does this):

    MUJOCO_GL=egl <fork>/.venv_sim/bin/python <aspire>/scripts/g1_sim/sim_rgbd.py

The fork is not modified: this script builds the fork's simulator, swaps the scene model
at load time and replaces its render hook. Headless (no viewer): the robot starts on the
elastic band, and SIGUSR1 toggles the band exactly like the viewer's `9` key
(``kill -USR1 $(cat <pid-file>)``); SIGUSR2 re-places the case and the bottle in front of
the robot as it stands now (it steps a little when dropped), the bottle upright.

The head camera is moved to the real D435 mounting and field of view (the transform comes
from the g1-lab station profile), so the g1-sim profile shares the real camera transform.
Frames go out on tcp://*:<port> in the composed-camera schema ASPIRE's G1 camera client
reads: ``ego_view`` RGB, ``ego_view_depth`` uint16 millimetres, ``intrinsics``, plus an
image-only ``third_view`` from a fixed camera ahead and to the right of the robot. Each
message also carries ``sim_truth`` for checking runs; ASPIRE ignores it: object positions
in the torso frame, and per object the robot links that touched it since the previous
message as ``[peak normal force, x, y, z]`` with the contact point in the link's frame
(contacts are collected on every physics step).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
import types

import msgpack
import msgpack_numpy
import mujoco
import numpy as np
import yaml
import zmq

ASPIRE_ROOT = Path(__file__).resolve().parents[2]
PROFILE = ASPIRE_ROOT / "src/cap_harness/configs/g1_real/g1-lab/station.yaml"
WIDTH, HEIGHT = 640, 480
REAL_FY_PX = 606.0  # the D435s on both PC2s report fy 604-607 at 640x480
MAX_DEPTH_M = 4.0  # beyond this the RealSense reports no return
# Scene, relative to the torso origin at load (the robot faces +x): a 0.84 m case whose
# top is at torso height with its near face 0.30 m ahead, and a 500 ml bottle on it.
CASE_NEAR_X, CASE_DEPTH, CASE_Y = 0.30, 0.40, (-0.25, 0.20)
BOTTLE_XY = (0.36, -0.06)
# A fixed third-person camera ahead and to the right, aimed at the bottle area
# (published as image-only `third_view`): eye and target, torso frame.
THIRD_EYE, THIRD_TARGET, THIRD_FOVY = (0.95, -0.75, 0.40), (0.40, -0.08, 0.02), 60.0


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera orientation (looks down -z, +y up) from eye to target, world up +z."""
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    return quat


def _third_view_pose(ahead, torso_z: float) -> tuple[np.ndarray, np.ndarray]:
    eye = np.array([*ahead(*THIRD_EYE[:2]), torso_z + THIRD_EYE[2]])
    target = np.array([*ahead(*THIRD_TARGET[:2]), torso_z + THIRD_TARGET[2]])
    return eye, _look_at(eye, target)


def _camera_pose() -> tuple[np.ndarray, np.ndarray]:
    """The real head camera in the torso frame, as a MuJoCo camera (looks down -z, +y up)."""
    rows = yaml.safe_load(PROFILE.read_text())["camera"]["base_from_camera"]
    t = np.asarray(rows, dtype=np.float64)
    # optical frame (z forward, x right, y down) -> MuJoCo camera frame: flip y and z
    rotation = t[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return t[:3, 3], quat


def build_model(path: str, *, objects: bool, yaw_deg: float = 0.0) -> mujoco.MjModel:
    """The fork's scene with the real camera (and the objects), loaded from an XML file.

    The fork's simulator calls ``mj_saveLastXML``, which needs a model loaded from XML,
    so the edited spec is written to a temporary file (asset dirs made absolute) and
    loaded from there.
    """
    spec = mujoco.MjSpec.from_file(str(Path(path).resolve()))
    camera = spec.camera("head_camera")
    camera.pos, camera.quat = _camera_pose()
    camera.fovy = math.degrees(2.0 * math.atan(HEIGHT / 2.0 / REAL_FY_PX))
    if yaw_deg:  # start facing elsewhere (the real robot's heading at init is not zero)
        half = math.radians(yaw_deg) / 2.0
        spec.body("pelvis").quat = [math.cos(half), 0.0, 0.0, math.sin(half)]
    if objects:
        _add_objects(spec)
    base = Path(path).resolve().parent
    for attr in ("meshdir", "texturedir"):
        value = getattr(spec, attr)
        if value and not os.path.isabs(value):
            setattr(spec, attr, str(base / value))
    with tempfile.NamedTemporaryFile(
        "w", suffix=".xml", prefix="aspire_g1_sim_", delete=False
    ) as out:
        out.write(spec.to_xml())
    return mujoco.MjModel.from_xml_path(out.name)


def _add_objects(spec: mujoco.MjSpec) -> None:
    probe = spec.compile()
    data = mujoco.MjData(probe)
    mujoco.mj_forward(probe, data)
    torso = probe.body("torso_link").id
    tx, ty, tz = data.xpos[torso]
    fx, fy = data.xmat[torso].reshape(3, 3)[:2, 0]
    yaw = math.atan2(fy, fx)

    def ahead(x: float, y: float) -> list[float]:
        return [
            tx + math.cos(yaw) * x - math.sin(yaw) * y,
            ty + math.sin(yaw) * x + math.cos(yaw) * y,
        ]

    upright = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    world = spec.worldbody
    # A mocap body: immovable by contact, but re-placeable at runtime (reset).
    case = world.add_body(
        name="aspire_case",
        mocap=True,
        pos=[*ahead(CASE_NEAR_X + CASE_DEPTH / 2, sum(CASE_Y) / 2), tz / 2],
        quat=upright,
    )
    case.add_geom(
        name="aspire_case_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CASE_DEPTH / 2, (CASE_Y[1] - CASE_Y[0]) / 2, tz / 2],
        rgba=[0.06, 0.06, 0.07, 1.0],
    )
    eye, quat = _third_view_pose(ahead, tz)
    rig = world.add_body(
        name="aspire_third_view_rig", mocap=True, pos=eye.tolist(), quat=quat.tolist()
    )
    rig.add_camera(name="aspire_third_view", fovy=THIRD_FOVY)
    bottle = world.add_body(
        name="aspire_bottle", pos=[*ahead(*BOTTLE_XY), tz + 0.075], quat=upright
    )
    bottle.add_freejoint(name="aspire_bottle_free")
    clear = [0.78, 0.88, 0.96, 1.0]
    for name, radius, half, z, rgba, mass in (
        ("body", 0.033, 0.075, 0.0, clear, 0.5),
        ("shoulder", 0.025, 0.012, 0.087, clear, 0.01),
        ("neck", 0.014, 0.012, 0.111, clear, 0.005),
        ("cap", 0.016, 0.009, 0.132, [0.05, 0.62, 0.55, 1.0], 0.005),
    ):
        bottle.add_geom(
            name=f"aspire_bottle_{name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, half, 0.0],
            pos=[0.0, 0.0, z],
            rgba=rgba,
            mass=mass,
        )


class ContactLog:
    """Robot links touching each object, collected on every physics step."""

    def __init__(self) -> None:
        self._objects: dict[int, str] | None = None
        self._force = np.zeros(6)
        self.touches: dict[str, dict[str, float]] = {}

    def collect(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._objects is None:
            self._objects = {
                g: name
                for name in ("aspire_case", "aspire_bottle")
                if (b := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == b
            }
        for i in range(data.ncon):
            contact = data.contact[i]
            pair = (contact.geom1, contact.geom2)
            hit = [self._objects.get(g) for g in pair]
            if hit[0] is None and hit[1] is None:
                continue
            obj, other = (hit[0], pair[1]) if hit[0] is not None else (hit[1], pair[0])
            body = model.geom_bodyid[other]
            if body == 0 or other in self._objects:  # floor / the other object
                continue
            mujoco.mj_contactForce(model, data, i, self._force)
            link = model.body(body).name
            force = round(float(abs(self._force[0])), 2)
            peak = self.touches.setdefault(obj, {})
            if force >= peak.get(link, (0.0,))[0]:
                # where on the link: the contact point in the link's own frame
                local = data.xmat[body].reshape(3, 3).T @ (contact.pos - data.xpos[body])
                peak[link] = (force, *np.round(local, 3).tolist())

    def drain(self) -> dict[str, dict[str, float]]:
        touches, self.touches = self.touches, {}
        return touches


class RgbdPublisher:
    """Renders the head camera (RGB and depth) and publishes ASPIRE's camera schema."""

    def __init__(self, port: int, contacts: ContactLog, layout_file: Path | None) -> None:
        self._contacts = contacts
        self._layout_file = layout_file
        self._sock = zmq.Context.instance().socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 3)
        self._sock.bind(f"tcp://*:{port}")
        self._rgb = self._depth = None
        focal = REAL_FY_PX  # the field of view above is derived from it
        self._intrinsics = {
            "fx": focal, "fy": focal, "ppx": WIDTH / 2.0, "ppy": HEIGHT / 2.0,
            "width": WIDTH, "height": HEIGHT, "depth_scale": 0.001,
        }  # fmt: skip
        third_focal = HEIGHT / 2.0 / math.tan(math.radians(THIRD_FOVY) / 2.0)
        self._third_intrinsics = {**self._intrinsics, "fx": third_focal, "fy": third_focal}

    reset_requested = False
    band_on = staticmethod(lambda: None)  # set once the simulator exists

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.reset_requested:  # applied here, in the simulation thread
            self.reset_requested = False
            bottle_xy = BOTTLE_XY
            if self._layout_file is not None and self._layout_file.exists():
                bottle_xy = tuple(
                    json.loads(self._layout_file.read_text()).get("bottle_xy", BOTTLE_XY)
                )
            _place_objects(model, data, bottle_xy)
        if self._rgb is None:  # GL contexts belong to the rendering thread
            self._rgb = mujoco.Renderer(model, HEIGHT, WIDTH)
            self._depth = mujoco.Renderer(model, HEIGHT, WIDTH)
            self._depth.enable_depth_rendering()
        self._rgb.update_scene(data, camera="head_camera")
        self._depth.update_scene(data, camera="head_camera")
        rgb = self._rgb.render()
        depth = self._depth.render()
        third = None
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "aspire_third_view") >= 0:
            self._rgb.update_scene(data, camera="aspire_third_view")
            third = self._rgb.render()
        depth_mm = np.where(depth < MAX_DEPTH_M, np.round(depth * 1000.0), 0).astype(np.uint16)
        message = {
            "timestamps": {"ego_view": time.time()},
            "images": {
                "ego_view": rgb,
                "ego_view_depth": depth_mm,
                **({"third_view": third} if third is not None else {}),
            },
            "intrinsics": {"ego_view": self._intrinsics, "third_view": self._third_intrinsics},
            "sim_truth": {
                **_truth(model, data),
                "contacts": self._contacts.drain(),
                "band": self.band_on(),
            },
        }
        self._sock.send(msgpack.packb(message, default=msgpack_numpy.encode, use_bin_type=True))


def _place_objects(
    model: mujoco.MjModel, data: mujoco.MjData, bottle_xy: tuple[float, float] = BOTTLE_XY
) -> None:
    """The case and the upright bottle at their layout relative to the torso's current pose."""
    case = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "aspire_case")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "aspire_bottle_free")
    if case < 0 or joint < 0:
        return
    torso = model.body("torso_link").id
    heading = data.xmat[torso].reshape(3, 3)[:, 0]
    yaw = math.atan2(heading[1], heading[0])
    c, s = math.cos(yaw), math.sin(yaw)
    origin = data.xpos[torso]

    def ahead(x: float, y: float) -> np.ndarray:
        return origin[:2] + np.array([c * x - s * y, s * x + c * y])

    upright = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    mocap = model.body_mocapid[case]
    half_height = float(model.geom_size[model.body_geomadr[case]][2])
    data.mocap_pos[mocap] = [*ahead(CASE_NEAR_X + CASE_DEPTH / 2, sum(CASE_Y) / 2), half_height]
    data.mocap_quat[mocap] = upright
    rig = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "aspire_third_view_rig")
    if rig >= 0:
        eye, quat = _third_view_pose(ahead, float(origin[2]))
        data.mocap_pos[model.body_mocapid[rig]] = eye
        data.mocap_quat[model.body_mocapid[rig]] = quat
    qpos, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[qpos : qpos + 3] = [*ahead(*bottle_xy), 2 * half_height + 0.076]
    data.qpos[qpos + 3 : qpos + 7] = upright
    data.qvel[dof : dof + 6] = 0.0
    print(
        f"[g1-sim] case and bottle placed in front of the robot (yaw {math.degrees(yaw):+.1f} deg, "
        f"bottle at {bottle_xy})",
        flush=True,
    )


def _truth(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[float]]:
    """Object positions in the torso frame (what ASPIRE's localization should find)."""
    torso = model.body("torso_link").id
    rotation, origin = data.xmat[torso].reshape(3, 3), data.xpos[torso]
    truth = {}
    for name in ("aspire_bottle", "aspire_case"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body >= 0:
            truth[name] = (rotation.T @ (data.xpos[body] - origin)).round(4).tolist()
            if name == "aspire_bottle":  # 1.0 upright, 0.0 lying on its side
                truth["aspire_bottle_upright"] = round(float(data.xmat[body][8]), 3)
    # the torso in the world: how far SONIC moves the frame everything is expressed in
    pitch = math.degrees(math.asin(-float(np.clip(rotation[2, 0], -1, 1))))
    roll = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    truth["torso_world"] = [*origin.round(4).tolist(), round(pitch, 2), round(roll, 2)]
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--no-objects", action="store_true", help="robot only, no case or bottle")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="initial heading of the robot")
    parser.add_argument(
        "--layout-file", type=Path, help='JSON {"bottle_xy": [x, y]} (torso frame) read on reset'
    )
    args = parser.parse_args()

    from gear_sonic.utils.mujoco_sim import base_sim
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

    # Swap the scene: base_sim loads it with mujoco.MjModel.from_xml_path.
    loader = types.SimpleNamespace(
        from_xml_path=lambda path: build_model(
            path, objects=not args.no_objects, yaw_deg=args.yaw_deg
        )
    )
    proxy = types.ModuleType("mujoco")
    proxy.__dict__.update(mujoco.__dict__)
    proxy.MjModel = loader
    base_sim.mujoco = proxy

    if args.yaw_deg:
        # The elastic band springs the torso's orientation toward identity, which would
        # turn the hanging robot back to yaw 0 before engage; aim it at the start heading.
        from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as bridge
        from scipy.spatial.transform import Rotation

        to_start = Rotation.from_euler("z", args.yaw_deg, degrees=True).inv()
        advance = bridge.ElasticBand.Advance

        def advance_toward_start(band: object, pose: np.ndarray) -> np.ndarray:
            pose = np.array(pose, dtype=float)
            w, x, y, z = pose[3:7]
            # error in the world frame (the band's torque is a world-frame torque)
            qx, qy, qz, qw = (Rotation.from_quat([x, y, z, w]) * to_start).as_quat()
            pose[3:7] = [qw, qx, qy, qz]
            return advance(band, pose)

        bridge.ElasticBand.Advance = advance_toward_start

    contacts = ContactLog()
    step = base_sim.DefaultEnv.sim_step

    def sim_step(env: object) -> None:
        step(env)
        contacts.collect(env.mj_model, env.mj_data)

    base_sim.DefaultEnv.sim_step = sim_step
    publisher = RgbdPublisher(args.port, contacts, args.layout_file)
    base_sim.DefaultEnv.update_render_caches = lambda env: publisher.publish(
        env.mj_model, env.mj_data
    )

    config = SimLoopConfig().load_wbc_yaml()
    config.update(ENV_NAME="default", IMAGE_DT=1.0 / args.fps)
    init_channel(config=config)
    sim = SimulatorFactory.create_simulator(
        config=config,
        env_name="default",
        onscreen=False,
        offscreen=False,
        enable_image_publish=False,
    )

    publisher.band_on = lambda: bool(
        getattr(getattr(sim.sim_env, "elastic_band", None), "enable", False)
    )

    def toggle_band(*_: object) -> None:
        band = getattr(sim.sim_env, "elastic_band", None)
        if band is not None:
            band.enable = not band.enable
            print(f"[g1-sim] elastic band {'on' if band.enable else 'off'}", flush=True)

    def reset_bottle(*_: object) -> None:
        publisher.reset_requested = True

    signal.signal(signal.SIGUSR1, toggle_band)
    signal.signal(signal.SIGUSR2, reset_bottle)
    if args.pid_file:
        args.pid_file.write_text(str(os.getpid()))
    print(
        f"[g1-sim] running; RGB-D on tcp://*:{args.port}; SIGUSR1 toggles the band, "
        "SIGUSR2 resets the bottle",
        flush=True,
    )
    SimulatorFactory.start_simulator(sim, as_thread=False, enable_image_publish=False)


if __name__ == "__main__":
    main()
