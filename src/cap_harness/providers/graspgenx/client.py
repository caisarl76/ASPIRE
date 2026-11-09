"""Grasps from a GraspGenX server (``client-server/graspgenx_server.py`` in NVlabs/GraspGenX).

GraspGenX is one model for any gripper, conditioned per request on the gripper's swept
volume: two boxes in the gripper base frame G (+Z approach, +X closing) enclosing the inner
finger volume open and half-open. The server is talked to with its own msgpack wire
format (``infer_object``), so this client needs neither torch nor the graspgenx package.

``DEX3_RIGHT`` describes the G1's right Dex3 hand against ``right_grasp_site`` (the IK
frame): the palm faces the site's +y, the thumb (site -x) and the two fingers (site +x)
close along x. Boxes are measured from the hand's open and half-closed forward kinematics
in ``g1_full_body.xml``; they agree with GraspGenX's shipped ``unitree_g1`` description to
about 1 cm. Returned grasps are converted from G to the grasp site, so a candidate pose
is a ``move_to_pose`` target. Candidates whose approach (G +Z) is more than
``max_approach_deg`` from ``approach_dir`` (in the cloud's frame) are dropped: the right arm
reaches a bottle from its right, moving left (torso +y), close to level. GraspGenX treats
the hand as symmetric under a half turn about its approach, so a grasp whose closing axis
(G +X) points away from ``closing_dir`` is turned that half turn.

Where G lands is calibrated, not geometry. On the sim's SAM3 cloud of the 6.6 cm bottle, G
sits a median 8.3 cm (p10-p90 6.6-9.7) behind the true bottle axis along the approach and
3.7 cm (2.6-4.8) short of it along the closing axis: the cloud is the near surface. The
bottle skill holds the site 5.3 cm from the axis along the approach and 6 cm past it toward
the fingers, so the site is 3.0 cm ahead of G and 9.7 cm along its closing axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from cap_harness.contracts import GraspCandidate, GraspSet, ObjectGeometry, PointCloud
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose

DEFAULT_GRASPGENX_URL = "tcp://127.0.0.1:5586"
# The server's outlier removal builds an N x N distance matrix before the model resamples
# to its 3500 points: a 345k-point SAM3 mask asked for 477 GB. 8192 points is ~270 MB.
MAX_POINTS = 8192


@dataclass(frozen=True)
class Hand:
    """A gripper as GraspGenX sees it, plus where its base frame G sits on the IK frame."""

    sweep_volume: dict
    site_from_g: np.ndarray  # (4, 4): G expressed in the IK frame
    width_m: float
    approach_dir: tuple[float, float, float] | None = None  # in the cloud's frame; None keeps all
    max_approach_deg: float = 30.0
    closing_dir: tuple[float, float, float] | None = None  # in the cloud's frame; G +X turned toward it


def _site_from_g(origin: tuple[float, float, float]) -> np.ndarray:
    # G axes in site coordinates: X = site x (closing), Y = -site z, Z = site y (palm normal).
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    transform[:3, 3] = origin
    return transform


DEX3_RIGHT = Hand(
    sweep_volume={
        "extents_open": [0.10, 0.06, 0.06],
        "offset_open": [0.0, 0.0, 0.05],
        "extents_mid": [0.04, 0.06, 0.05],
        "offset_mid": [0.01, 0.0, 0.055],
        "gripper_type": 2,  # revolute_3f
        "fingertip_depth": 0.07,
    },
    # G: 9.7 cm behind the site along x (closing), 3.0 cm behind it along the approach.
    site_from_g=_site_from_g((-0.097, -0.030, 0.0)),
    width_m=0.10,
    approach_dir=(0.0, 1.0, 0.0),
    closing_dir=(1.0, 0.0, 0.0),  # fingers forward, thumb toward the robot
)


class GraspGenXProvider:
    """``GraspProvider`` over a GraspGenX ZMQ server; poses come back in the cloud's frame."""

    def __init__(
        self,
        url: str = DEFAULT_GRASPGENX_URL,
        *,
        hand: Hand = DEX3_RIGHT,
        planner: str = "graspmoe",
        num_grasps: int = 200,
        timeout_s: float = 30.0,
        max_points: int = MAX_POINTS,
    ) -> None:
        self.url, self.hand, self.planner = url, hand, planner
        self.max_points = int(max_points)
        self.num_grasps, self.timeout_ms = int(num_grasps), int(timeout_s * 1000)
        self._context = zmq.Context.instance()
        self._g_from_site = np.linalg.inv(hand.site_from_g)

    def _request(self, payload: dict) -> dict:
        # A REQ socket is unusable after a timeout, so each request gets a fresh one.
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        try:
            socket.connect(self.url)
            socket.send(msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True))
            return msgpack.unpackb(socket.recv(), object_hook=msgpack_numpy.decode, raw=False)
        finally:
            socket.close()

    def generate_grasps(
        self,
        point_cloud: PointCloud,
        *,
        scene_point_cloud: PointCloud | None = None,
        geometry: ObjectGeometry | None = None,
        max_candidates: int | None = None,
    ) -> GraspSet:
        # ponytail: no scene collision filter (the server skips it in params mode); add one
        # against scene_point_cloud when grasps start hitting the support surface.
        points = np.asarray(point_cloud.points, dtype=np.float32)
        if len(points) > self.max_points:  # uniform, seeded: the same cloud gives the same request
            pick = np.random.default_rng(0).choice(len(points), self.max_points, replace=False)
            points = points[pick]
        try:
            response = self._request(
                {
                    "action": "infer_object",
                    "point_cloud": points,
                    "sweep_volume_params": self.hand.sweep_volume,
                    "num_grasps": self.num_grasps,
                    "planner": self.planner,
                }
            )
        except zmq.Again:
            return _failure(ErrorCode.TIMEOUT, f"GraspGenX at {self.url} did not answer")
        except zmq.ZMQError as exc:
            return _failure(ErrorCode.PROVIDER_UNAVAILABLE, f"GraspGenX at {self.url}: {exc}")
        if "error" in response or "grasps" not in response:
            return _failure(ErrorCode.GRASP_FAILED, f"GraspGenX: {response.get('error', response)}")
        grasps = np.asarray(response["grasps"], dtype=np.float64).reshape(-1, 4, 4)
        scores = np.asarray(response["confidences"], dtype=np.float64).reshape(-1)
        if self.hand.closing_dir is not None:
            # The swept volume is symmetric under a half turn about G +Z; the hand is not
            # (thumb on one side), and half the grasps come back with the thumb outward.
            flip = grasps[:, :3, 0] @ np.asarray(self.hand.closing_dir) < 0
            grasps[flip] = grasps[flip] @ np.diag([-1.0, -1.0, 1.0, 1.0])
        keep = np.ones(len(scores), dtype=bool)
        if self.hand.approach_dir is not None:
            direction = np.asarray(self.hand.approach_dir) / np.linalg.norm(self.hand.approach_dir)
            cos = grasps[:, :3, 2] @ direction
            keep = cos >= np.cos(np.radians(self.hand.max_approach_deg))
        order = np.flatnonzero(keep)[np.argsort(-scores[keep])][:max_candidates]
        if not len(order):
            return _failure(
                ErrorCode.NO_GRASP,
                f"GraspGenX returned {len(scores)} grasps, none within "
                f"{self.hand.max_approach_deg:g} deg of approach {self.hand.approach_dir}",
            )
        candidates = [
            GraspCandidate(
                pose=matrix_to_pose(grasps[i] @ self._g_from_site, frame=point_cloud.frame),
                score=float(scores[i]),
                width_m=self.hand.width_m,
                metadata={"provider": "graspgenx", "rank": rank},
            )
            for rank, i in enumerate(order)
        ]
        diagnostics = {
            "provider": "graspgenx",
            "points_sent": len(points),
            "generated": len(scores),
            "in_approach_cone": int(keep.sum()),
        }
        return GraspSet(ok=True, grasps=candidates, diagnostics=diagnostics)


def _failure(code: ErrorCode, message: str) -> GraspSet:
    return GraspSet(
        ok=False, error=ApiError(code=code, message=message, details={"provider": "graspgenx"})
    )
