import io
import time

import imageio.v3 as iio
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

from cap_harness.g1_real.cameras import ComposedCameraRgbd

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
    assert frame is not None and np.isnan(frame.depth_m).all() and cam.depth_missing
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
