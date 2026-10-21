import io
import time

import imageio.v3 as iio
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

from cap_harness.g1_real.cameras import ComposedCameraRgbd, drop_depth_edges

INTR = {
    "fx": 600.0,
    "fy": 600.0,
    "ppx": 2.0,
    "ppy": 2.0,
    "width": 4,
    "height": 4,
    "depth_scale": 0.001,
}


def _publish(port, payload):
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    time.sleep(0.3)
    pub.send(msgpack.packb(payload, default=m.encode, use_bin_type=True))
    return pub


def _read(cam):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        frame = cam.read()
        if frame is not None:
            return frame
        time.sleep(0.05)
    return None


def test_client_builds_metric_frame():
    cam = ComposedCameraRgbd("127.0.0.1", 5730, role="ego_view")
    pub = _publish(
        5730,
        {
            "timestamps": {"ego_view": 1.0},
            "images": {
                "ego_view": np.zeros((4, 4, 3), np.uint8),
                "ego_view_depth": np.full((4, 4), 1500, np.uint16),
            },
            "intrinsics": {"ego_view": INTR},
        },
    )
    frame = _read(cam)
    assert frame.depth_m[0, 0] == 1.5 and frame.intrinsics[0, 0] == 600.0
    assert frame.intrinsics[0, 2] == 2.0 and not cam.depth_missing
    cam.close()
    pub.close()


def test_missing_depth_is_flagged_not_fatal():
    cam = ComposedCameraRgbd("127.0.0.1", 5731)
    pub = _publish(
        5731,
        {"timestamps": {"ego_view": 1.0}, "images": {"ego_view": np.zeros((4, 4, 3), np.uint8)}},
    )
    frame = _read(cam)
    assert frame is not None and not np.any(frame.depth_m) and cam.depth_missing
    assert np.all(np.isfinite(frame.depth_m))  # the CameraObservation contract rejects NaN
    assert cam.intrinsics_missing
    np.testing.assert_array_equal(frame.intrinsics, [[1, 0, 2], [0, 1, 2], [0, 0, 1]])
    cam.close()
    pub.close()


def test_jpeg_bytes_image_is_decoded():
    buf = io.BytesIO()
    iio.imwrite(buf, np.full((8, 6, 3), 200, np.uint8), extension=".jpg")
    cam = ComposedCameraRgbd("127.0.0.1", 5732)
    pub = _publish(5732, {"timestamps": {}, "images": {"ego_view": buf.getvalue()}})
    frame = _read(cam)
    assert frame.rgb.shape == (8, 6, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth_m.shape == (8, 6)
    cam.close()
    pub.close()


def test_read_is_none_before_first_message():
    cam = ComposedCameraRgbd("127.0.0.1", 5733)
    assert cam.read() is None
    cam.close()


def test_undecodable_messages_are_dropped_and_keep_last_good_frame():
    cam = ComposedCameraRgbd("127.0.0.1", 5734)
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5734")
    time.sleep(0.3)

    def send(payload):
        pub.send(msgpack.packb(payload, default=m.encode, use_bin_type=True))
        time.sleep(0.3)

    send({"images": {"other": np.zeros((4, 4, 3), np.uint8)}})
    send({"timestamps": {}})
    send({"images": {"ego_view": b"not a jpeg"}})
    assert cam.read() is None
    send({"images": {"ego_view": np.zeros((4, 4, 3), np.uint8)}})
    good = cam.read()
    assert good is not None
    send({"images": {"ego_view": b"not a jpeg"}})
    assert cam.read() is good
    cam.close()
    pub.close()


def test_depth_without_intrinsics_is_dropped_to_zero():
    cam = ComposedCameraRgbd("127.0.0.1", 5735)
    pub = _publish(
        5735,
        {
            "images": {
                "ego_view": np.zeros((4, 4, 3), np.uint8),
                "ego_view_depth": np.full((4, 4), 1500, np.uint16),
            }
        },
    )
    frame = _read(cam)
    assert frame is not None and not np.any(frame.depth_m) and np.all(np.isfinite(frame.depth_m))
    assert cam.depth_missing and cam.intrinsics_missing
    cam.close()
    pub.close()


def test_depth_shape_mismatch_is_dropped_to_zero():
    cam = ComposedCameraRgbd("127.0.0.1", 5736)
    pub = _publish(
        5736,
        {
            "images": {
                "ego_view": np.zeros((4, 4, 3), np.uint8),
                "ego_view_depth": np.full((2, 2), 1500, np.uint16),
            },
            "intrinsics": {"ego_view": INTR},
        },
    )
    frame = _read(cam)
    assert frame is not None and frame.depth_m.shape == (4, 4) and not np.any(frame.depth_m)
    assert cam.depth_missing and not cam.intrinsics_missing
    assert frame.intrinsics[0, 0] == 600.0
    cam.close()
    pub.close()


def test_depth_edges_are_dropped_and_surfaces_kept():
    depth = np.full((40, 40), 1.0)  # background plane
    depth[10:30, 10:30] = 0.5  # object in front
    depth[10, 20] = 0.8  # flying pixel on the object's edge
    depth[0, 0] = 0.0  # no return stays no return
    out = drop_depth_edges(depth)
    assert out[10, 20] == 0.0  # the mixed depth is gone
    assert out[20, 20] == 0.5 and out[35, 35] == 1.0  # interiors untouched
    assert np.all(out[10:30, 10:30][4:-4, 4:-4] == 0.5)
    assert out[0, 0] == 0.0 and np.all(out[(out > 0)] > 0.4)


def test_slanted_surface_is_not_an_edge():
    # a plane receding 3 mm per pixel (a table seen at 45 deg) has no jump
    depth = 0.6 + 0.003 * np.arange(60)[:, None] * np.ones((60, 60))
    assert np.all(drop_depth_edges(depth) == depth)


def _render_depth(base_from_camera, k, floor_z, tilt_x, h=48, w=64):
    """Depth of a torso-frame scene: a tilted floor and a level case top at z=0."""
    from cap_harness.g1_real.cameras import drop_floor  # noqa: F401  (import check)

    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones((h, w))], -1)
    r = rays @ base_from_camera[:3, :3].T
    o = base_from_camera[:3, 3]
    # floor: z = floor_z + tilt_x * x
    t_floor = (floor_z + tilt_x * o[0] - o[2]) / (r[..., 2] - tilt_x * r[..., 0])
    t_case = (0.0 - o[2]) / r[..., 2]
    hit = o + t_case[..., None] * r
    on_case = (t_case > 0) & (np.abs(hit[..., 0] - 0.45) < 0.15) & (np.abs(hit[..., 1]) < 0.12)
    depth = np.where(on_case, t_case, np.where(t_floor > 0, t_floor, 0.0))
    return depth, on_case


def test_floor_is_dropped_and_the_case_top_kept():
    from cap_harness.g1_real.cameras import drop_floor
    from cap_harness.g1_real.config import load_g1_station_config

    cfg = load_g1_station_config("g1-lab")
    k = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    T = np.asarray(cfg.camera.base_from_camera)
    depth, on_case = _render_depth(T, k, floor_z=-0.84, tilt_x=0.03)  # SONIC's stance tilt
    out = drop_floor(depth, k, T, below_z_m=-0.5)
    floor = (depth > 0) & ~on_case
    assert floor.sum() > 100 and on_case.sum() > 100
    assert np.all(out[floor] == 0.0)
    assert np.all(out[on_case] == depth[on_case])


def test_no_floor_in_view_drops_nothing():
    from cap_harness.g1_real.cameras import drop_floor
    from cap_harness.g1_real.config import load_g1_station_config

    T = np.asarray(load_g1_station_config("g1-lab").camera.base_from_camera)
    k = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    depth = np.full((48, 64), 0.5)  # everything near the camera, nothing low
    assert np.array_equal(drop_floor(depth, k, T, below_z_m=-0.5), depth)


def test_image_only_extra_camera_is_read_from_the_same_message():
    cam = ComposedCameraRgbd("127.0.0.1", 5737, aux_roles=("third_view",))
    pub = _publish(
        5737,
        {
            "images": {
                "ego_view": np.zeros((4, 4, 3), np.uint8),
                "third_view": np.full((6, 8, 3), 90, np.uint8),
            },
            "intrinsics": {"ego_view": INTR},
        },
    )
    assert _read(cam) is not None
    aux = cam.read_aux("third_view")
    assert aux is not None and aux.rgb.shape == (6, 8, 3) and np.all(aux.rgb == 90)
    assert not np.any(aux.depth_m)  # image only
    assert cam.read_aux("missing") is None
    cam.close()
    pub.close()
