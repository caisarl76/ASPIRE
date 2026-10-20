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
