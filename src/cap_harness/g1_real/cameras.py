"""RGB-D client for the G1's composed camera server (msgpack frames over ZMQ)."""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Any

import imageio.v3 as iio
import msgpack
import msgpack_numpy
import numpy as np
import zmq

from cap_harness.yam_real.cameras import CameraFrame

logger = logging.getLogger(__name__)

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
        self._frame: CameraFrame | None = None
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
                frame, depth_missing, intrinsics_missing = self._assemble(msg, time.monotonic())
            except Exception as exc:
                # A bad message is dropped; the previous good frame stays.
                logger.warning("dropping undecodable camera message: %r", exc)
                continue
            with self._lock:
                self._frame = frame
                self.depth_missing = depth_missing
                self.intrinsics_missing = intrinsics_missing

    def _assemble(self, msg: dict[str, Any], received_s: float) -> tuple[CameraFrame, bool, bool]:
        images = msg["images"]
        rgb = _decode_rgb(images[self._role])
        h, w = rgb.shape[:2]
        intr = msg.get("intrinsics", {}).get(self._role)
        if intr is None:
            k = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], np.float64)
        else:
            k = np.array(
                [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], np.float64
            )
        depth = images.get(f"{self._role}_depth")
        if depth is None:
            depth_m = np.full((h, w), np.nan)
        else:
            scale = (intr or {}).get("depth_scale", self._depth_scale_m)
            depth_m = np.asarray(depth).astype(np.float64) * scale
        frame = CameraFrame(rgb=rgb, depth_m=depth_m, intrinsics=k, timestamp_s=received_s)
        return frame, depth is None, intr is None

    def read(self) -> CameraFrame | None:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sock.close()
