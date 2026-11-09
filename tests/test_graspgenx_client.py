import threading

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from cap_harness.contracts import PointCloud
from cap_harness.providers.graspgenx import GraspGenXProvider


def _serve_once(socket: zmq.Socket, grasps: np.ndarray, scores: np.ndarray, seen: list) -> None:
    seen.append(msgpack.unpackb(socket.recv(), object_hook=msgpack_numpy.decode, raw=False))
    reply = {"grasps": grasps.astype(np.float32), "confidences": scores.astype(np.float32)}
    socket.send(msgpack.packb(reply, default=msgpack_numpy.encode, use_bin_type=True))


def test_level_side_grasp_maps_to_identity_site_pose() -> None:
    # G approaching along +y with x closing is the bottle skill's level hand (quat 1 0 0 0);
    # the site sits 9.7 cm past G along x and 3 cm ahead of it along the approach (+y).
    g = np.eye(4)
    g[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    g[:3, 3] = [0.3, -0.2, 0.1]
    upward = np.eye(4)  # approaches along +z: outside the right-side cone despite its higher score
    thumb_out = g @ np.diag([-1.0, -1.0, 1.0, 1.0])  # the same grasp half-turned about the approach
    socket = zmq.Context.instance().socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    seen: list = []
    server = threading.Thread(
        target=_serve_once, args=(socket, np.stack([upward, thumb_out]), np.array([0.95, 0.9]), seen)
    )
    server.start()
    provider = GraspGenXProvider(f"tcp://127.0.0.1:{port}", timeout_s=5.0)
    cloud = PointCloud(points=np.random.default_rng(0).normal(size=(50, 3)), frame="torso")
    result = provider.generate_grasps(cloud, max_candidates=1)
    server.join(5.0)
    socket.close()

    assert seen[0]["action"] == "infer_object"
    assert seen[0]["point_cloud"].shape == (50, 3)
    assert seen[0]["sweep_volume_params"]["gripper_type"] == 2
    assert result.ok and len(result.grasps) == 1 and result.diagnostics["in_approach_cone"] == 1
    best = result.grasps[0]
    assert best.score == np.float32(0.9) and best.pose.frame == "torso"
    np.testing.assert_allclose(np.abs(best.pose.quaternion_wxyz), [1, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(best.pose.position, [0.397, -0.17, 0.1], atol=1e-9)


def test_unreachable_server_is_a_recoverable_timeout() -> None:
    provider = GraspGenXProvider("tcp://127.0.0.1:1", timeout_s=0.2)
    cloud = PointCloud(points=np.zeros((5, 3)), frame="torso")
    result = provider.generate_grasps(cloud)
    assert not result.ok and result.error.code.value in {"timeout", "provider_unavailable"}


def test_large_cloud_is_capped_before_sending() -> None:
    # A broad SAM3 mask (345k points) once made the server's outlier removal ask for 477 GB.
    socket = zmq.Context.instance().socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    seen: list = []
    server = threading.Thread(
        target=_serve_once, args=(socket, np.eye(4)[None], np.array([0.5]), seen)
    )
    server.start()
    points = np.random.default_rng(1).normal(size=(20_000, 3))
    provider = GraspGenXProvider(f"tcp://127.0.0.1:{port}", timeout_s=5.0, max_points=1000)
    provider.generate_grasps(PointCloud(points=points, frame="torso"))
    server.join(5.0)
    socket.close()

    sent = seen[0]["point_cloud"]
    assert sent.shape == (1000, 3)
    assert len(np.unique(sent, axis=0)) == 1000  # no repeats
    original = {tuple(p) for p in points.astype(np.float32)}
    assert all(tuple(p) in original for p in sent)  # a subset, not new points
