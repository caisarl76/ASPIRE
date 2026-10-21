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
from scipy.ndimage import maximum_filter, minimum_filter
import zmq

from cap_harness.yam_real.cameras import CameraFrame

logger = logging.getLogger(__name__)

_POLL_MS = 100
# Flying-pixel filter: a pixel whose 5x5 neighbourhood spans more depth than
# max(3 cm, 5% of its own depth) sits on a depth jump. The RealSense reports
# mixed depths there, which back-project into points between (or far behind)
# the object and its background; a single such pixel stretched a localized
# bottle's bounding box to 1.4 m on the robot (2026-10-20).
_EDGE_WINDOW_PX = 5
_EDGE_MIN_JUMP_M = 0.03
_EDGE_REL_JUMP = 0.05


def drop_depth_edges(depth_m: np.ndarray) -> np.ndarray:
    """``depth_m`` with pixels on depth discontinuities set to 0 (no return)."""
    valid = depth_m > 0.0
    if not np.any(valid):
        return depth_m
    high = maximum_filter(np.where(valid, depth_m, -np.inf), size=_EDGE_WINDOW_PX)
    low = minimum_filter(np.where(valid, depth_m, np.inf), size=_EDGE_WINDOW_PX)
    edge = valid & (high - low > np.maximum(_EDGE_MIN_JUMP_M, _EDGE_REL_JUMP * depth_m))
    return np.where(edge, 0.0, depth_m)


def _decode_rgb(value: Any) -> np.ndarray:
    if isinstance(value, str):  # legacy base64 JPEG
        value = base64.b64decode(value)
    if isinstance(value, bytes | bytearray):
        return np.asarray(iio.imread(io.BytesIO(bytes(value)), extension=".jpg"), np.uint8)
    return np.asarray(value, np.uint8)


_FLOOR_MARGIN_M = 0.03  # pixels this close above the floor plane (or below it) go
_FLOOR_FIT_TOL_M = 0.02
_FLOOR_FIT_SAMPLES = 4000
_FLOOR_FIT_ITERS = 60
_FLOOR_MIN_UP = 0.9  # plane normal's vertical component: level planes only
_FLOOR_MIN_INLIERS = 0.3


def drop_floor(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    below_z_m: float,
) -> np.ndarray:
    """``depth_m`` with the floor set to 0 (no return).

    The floor is the dominant level plane (RANSAC, then least squares) among
    points lower than ``below_z_m`` in the base frame. Fitted per frame because
    the robot's stance tilts it in the torso frame. With no such plane in view
    (a table filling the image, the floor hidden) nothing is dropped.
    """
    valid = depth_m > 0.0
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w]
    camera = np.stack(
        [
            (u - intrinsics[0, 2]) * depth_m / intrinsics[0, 0],
            (v - intrinsics[1, 2]) * depth_m / intrinsics[1, 1],
            depth_m,
        ],
        axis=-1,
    )
    base = camera @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]
    low = base[valid & (base[..., 2] < below_z_m)]
    if len(low) < 500:
        return depth_m
    rng = np.random.default_rng(0)  # deterministic per frame
    sample = low[rng.choice(len(low), min(len(low), _FLOOR_FIT_SAMPLES), replace=False)]
    best, best_count = None, 0
    for _ in range(_FLOOR_FIT_ITERS):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm * (1.0 if normal[2] >= 0.0 else -1.0)
        if normal[2] < _FLOOR_MIN_UP:
            continue
        count = int(np.sum(np.abs((sample - a) @ normal) < _FLOOR_FIT_TOL_M))
        if count > best_count:
            best, best_count = (normal, a), count
    if best is None or best_count < _FLOOR_MIN_INLIERS * len(sample):
        return depth_m
    normal, point = best
    inliers = sample[np.abs((sample - point) @ normal) < _FLOOR_FIT_TOL_M]
    design = np.c_[inliers[:, 0], inliers[:, 1], np.ones(len(inliers))]
    slope_x, slope_y, offset = np.linalg.lstsq(design, inliers[:, 2], rcond=None)[0]
    height = base[..., 2] - (slope_x * base[..., 0] + slope_y * base[..., 1] + offset)
    return np.where(valid & (height < _FLOOR_MARGIN_M), 0.0, depth_m)


class ComposedCameraRgbd:
    """Keeps the latest server message; ``read()`` turns it into a ``CameraFrame``."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        role: str = "ego_view",
        depth_scale_m: float = 0.001,
        base_from_camera: np.ndarray | None = None,
        floor_below_z_m: float | None = None,
        aux_roles: tuple[str, ...] = (),
    ):
        self._role = role
        self._aux_roles = tuple(aux_roles)
        self._aux: dict[str, CameraFrame] = {}
        self._depth_scale_m = depth_scale_m
        # ponytail: the floor is refit on every frame (~30 ms); reuse a plane
        # across frames if the camera thread ever falls behind the stream.
        self._floor = (
            None
            if base_from_camera is None or floor_below_z_m is None
            else (np.asarray(base_from_camera, dtype=np.float64), float(floor_below_z_m))
        )
        self.depth_missing = False
        self.intrinsics_missing = False
        self._logged: set[str] = set()  # one log line per condition, not per frame
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
            aux = self._assemble_aux(msg, frame.timestamp_s)
            with self._lock:
                self._frame = frame
                self._aux.update(aux)
                self.depth_missing = depth_missing
                self.intrinsics_missing = intrinsics_missing

    def _warn_once(self, condition: str, message: str) -> None:
        if condition not in self._logged:
            self._logged.add(condition)
            logger.warning("%s camera: %s", self._role, message)

    def _assemble(self, msg: dict[str, Any], received_s: float) -> tuple[CameraFrame, bool, bool]:
        """The frame plus ``(depth_missing, intrinsics_missing)``.

        Depth is usable only with intrinsics and at the colour resolution:
        a fabricated fx=fy=1 or a misaligned map would back-project into
        confident metre-scale garbage, so either case is the missing-depth
        path, logged once. Missing depth is an all-zero ``depth_m``: zero is
        the sensor convention for "no return" that the shared contracts
        accept and the projection code already drops (NaN is rejected by
        ``CameraObservation`` and would break every observation).
        """
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
        depth_m = None
        if depth is None:
            self._warn_once("no_depth", "no depth image; depth_m is all zero (no returns)")
        elif intr is None:
            self._warn_once(
                "no_intrinsics", "depth without intrinsics; depth_m is all zero (no returns)"
            )
        elif np.shape(depth)[:2] != (h, w):
            self._warn_once(
                "depth_shape",
                f"depth shape {np.shape(depth)[:2]} does not match colour {(h, w)}; "
                "depth_m is all zero (no returns)",
            )
        else:
            scale = intr.get("depth_scale", self._depth_scale_m)
            depth_m = drop_depth_edges(np.asarray(depth).astype(np.float64) * scale)
            if self._floor is not None:
                depth_m = drop_floor(depth_m, k, *self._floor)
        if depth_m is None:
            depth_m = np.zeros((h, w), dtype=np.float64)
        frame = CameraFrame(rgb=rgb, depth_m=depth_m, intrinsics=k, timestamp_s=received_s)
        return frame, not bool(np.any(depth_m > 0.0)), intr is None

    def _assemble_aux(self, msg: dict[str, Any], received_s: float) -> dict[str, CameraFrame]:
        """Image-only frames for the extra roles present in this message (RGB only)."""
        frames = {}
        for role in self._aux_roles:
            if role not in msg["images"]:
                continue
            try:
                rgb = _decode_rgb(msg["images"][role])
            except Exception as exc:  # a bad extra image must not cost the main frame
                self._warn_once(f"aux_{role}", f"undecodable {role} image: {exc!r}")
                continue
            h, w = rgb.shape[:2]
            intr = msg.get("intrinsics", {}).get(role)
            k = (
                np.array(
                    [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]],
                    np.float64,
                )
                if intr
                else np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], np.float64)
            )
            frames[role] = CameraFrame(
                rgb=rgb, depth_m=np.zeros((h, w)), intrinsics=k, timestamp_s=received_s
            )
        return frames

    def read(self) -> CameraFrame | None:
        with self._lock:
            return self._frame

    def read_aux(self, role: str) -> CameraFrame | None:
        with self._lock:
            return self._aux.get(role)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sock.close()
