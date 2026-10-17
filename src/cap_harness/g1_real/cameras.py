"""RGB-D client for the G1's composed camera server (msgpack frames over ZMQ)."""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any

import imageio.v3 as iio
import msgpack
import msgpack_numpy
import numpy as np
import zmq

from cap_harness.yam_real.cameras import CameraFrame

_POLL_MS = 100


def _decode_rgb(value: Any) -> np.ndarray:
    if isinstance(value, str):  # legacy base64 JPEG
        value = base64.b64decode(value)
    if isinstance(value, bytes | bytearray):
        return np.asarray(iio.imread(io.BytesIO(bytes(value)), extension=".jpg"), np.uint8)
    return np.asarray(value, np.uint8)


class ComposedCameraRgbd:
    """Keeps the latest server message; ``read()`` turns it into a ``CameraFrame``."""

    def __init__(
        self, host: str, port: int, *, role: str = "ego_view", depth_scale_m: float = 0.001
    ):
        self._role = role
        self._depth_scale_m = depth_scale_m
        self.depth_missing = False
        self.intrinsics_missing = False
        self._latest: tuple[dict[str, Any], float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 3)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.connect(f"tcp://{host}:{port}")
        self._thread = threading.Thread(target=self._run, daemon=True, name="g1-camera-rgbd")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._sock.poll(_POLL_MS):
                continue
            try:
                msg = msgpack.unpackb(
                    self._sock.recv(), object_hook=msgpack_numpy.decode, raw=False
                )
            except Exception:  # noqa: S112
                continue  # malformed message must not kill the reader
            with self._lock:
                self._latest = (msg, time.monotonic())

    def read(self) -> CameraFrame | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        msg, received_s = latest
        images = msg.get("images", {})
        rgb = _decode_rgb(images[self._role])
        h, w = rgb.shape[:2]
        intr = msg.get("intrinsics", {}).get(self._role)
        self.intrinsics_missing = intr is None
        if intr is None:
            k = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], np.float64)
        else:
            k = np.array(
                [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], np.float64
            )
        depth = images.get(f"{self._role}_depth")
        self.depth_missing = depth is None
        if depth is None:
            depth_m = np.full((h, w), np.nan)
        else:
            scale = (intr or {}).get("depth_scale", self._depth_scale_m)
            depth_m = np.asarray(depth).astype(np.float64) * scale
        return CameraFrame(rgb=rgb, depth_m=depth_m, intrinsics=k, timestamp_s=received_s)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sock.close()
