"""FK, IK and grasp-site Jacobians for the torso-rooted G1 upper body.

Mirrors :class:`cap_harness.yam_real.kinematics.YamKinematics` (same method
names and return conventions) over the generated ``g1_upper_body.xml``: legs and
freejoint removed, so ``pelvis`` is fixed in the world and the waist joints are
the only thing between the world and ``torso_link``.

Every public pose is expressed in the ``torso_link`` frame, the embodiment's
``base_frame``. FK composes ``inv(T_world_torso) @ T_world_site``; IK maps its
torso-frame targets to world with ``T_world_torso`` evaluated at the seeded
configuration (waist included) before handing them to mink.

The qpos layout is read off the model by joint name: waist ``[0:3]``, then per
side the seven arm joints in Unitree order followed by the seven Dex3 finger
joints. Callers pass and receive per-arm 7-vectors. During IK a high-cost
``PostureTask`` restricted to the non-arm DoF (waist and fingers) holds them at
the seeded value, so only the fourteen arm joints move.

Quaternions here are **xyzw** (mink/scipy); the harness contracts use wxyz.

Not thread safe: one mutable ``mink.Configuration`` per instance.
"""

from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np

from cap_harness.g1_real.codec import ARM_DOF, JOINT_NAMES

WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
HOLD_COST = 1e3


class G1Kinematics:
    """FK/IK over the torso-rooted G1 upper-body model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        lm_damping: float = 1.0,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"G1 upper-body model XML not found: {path}")
        self.model_path = path
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self.configuration = mink.Configuration(self._model)

        def qpos_index(names: tuple[str, ...]) -> np.ndarray:
            joint_ids = [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names
            ]
            if min(joint_ids) < 0:
                raise RuntimeError(f"joint missing from {path}: {names[joint_ids.index(-1)]}")
            return self._model.jnt_qposadr[joint_ids]

        # All joints are hinges (nq == nv), so one index set serves qpos and the tangent space.
        self._arm_index = {
            side: qpos_index(tuple(f"{side}_{name}" for name in JOINT_NAMES))
            for side in ("left", "right")
        }
        self._waist_index = qpos_index(WAIST_JOINTS)

        self.tasks = [
            mink.FrameTask(
                frame_name=f"{side}_grasp_site",
                frame_type="site",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=lm_damping,
            )
            for side in ("left", "right")
        ]
        self.left_end_effector_task, self.right_end_effector_task = self.tasks

        hold_cost = np.full(self._model.nv, HOLD_COST)
        hold_cost[np.concatenate(list(self._arm_index.values()))] = 0.0
        self._hold_task = mink.PostureTask(self._model, cost=hold_cost)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def seed(
        self,
        left_joint_pos: np.ndarray,
        right_joint_pos: np.ndarray,
        waist_joint_pos: np.ndarray | None = None,
    ) -> None:
        """Write both arms (and the waist when given) into the shared configuration."""
        qpos = self.configuration.data.qpos
        qpos[self._arm_index["left"]] = np.asarray(left_joint_pos, dtype=np.float64).reshape(
            ARM_DOF
        )
        qpos[self._arm_index["right"]] = np.asarray(right_joint_pos, dtype=np.float64).reshape(
            ARM_DOF
        )
        if waist_joint_pos is not None:
            qpos[self._waist_index] = np.asarray(waist_joint_pos, dtype=np.float64).reshape(3)
        self.configuration.update()

    def _world_from_torso(self) -> mink.SE3:
        return self.configuration.get_transform_frame_to_world("torso_link", "body")

    def forward_kinematics(
        self,
        left_joint_pos: np.ndarray,
        right_joint_pos: np.ndarray,
        waist_joint_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(left_pos, left_quat_xyzw, right_pos, right_quat_xyzw)`` in the torso frame."""
        self.seed(left_joint_pos, right_joint_pos, waist_joint_pos)
        torso_from_world = self._world_from_torso().inverse()
        poses = {
            side: torso_from_world
            @ self.configuration.get_transform_frame_to_world(f"{side}_grasp_site", "site")
            for side in ("left", "right")
        }
        return (
            poses["left"].translation(),
            poses["left"].rotation().wxyz[[1, 2, 3, 0]],
            poses["right"].translation(),
            poses["right"].rotation().wxyz[[1, 2, 3, 0]],
        )

    def grasp_site_jacobian(
        self,
        side: str,
        left_joint_pos: np.ndarray | None = None,
        right_joint_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(J_pos, J_rot)``, each ``(3, 7)``, for one arm's grasp site.

        Columns are that arm's own seven joints, so the result maps arm joint
        velocity to grasp-site linear/angular velocity in the world frame (the
        torso is held, so with a fixed waist this is the torso frame up to the
        constant rotation ``R_world_torso``). Pass both arms' positions to
        evaluate at a specific configuration, or omit them to reuse whatever
        was last written.
        """
        if side not in self._arm_index:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        if left_joint_pos is not None and right_joint_pos is not None:
            self.seed(left_joint_pos, right_joint_pos)
        elif left_joint_pos is not None or right_joint_pos is not None:
            raise ValueError("pass both arms' joint positions, or neither")

        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_site")
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site {side}_grasp_site not found in {self.model_path}")
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self._model, self.configuration.data, jacp, jacr, site_id)
        cols = self._arm_index[side]
        return jacp[:, cols].copy(), jacr[:, cols].copy()

    def inverse_kinematics(
        self,
        left_ee_pos: np.ndarray,
        left_ee_quat_xyzw: np.ndarray,
        right_ee_pos: np.ndarray,
        right_ee_quat_xyzw: np.ndarray,
        *,
        seeded: bool = False,
        dt: float = 0.01,
        solver: str = "daqp",
        damping: float = 1e-3,
        err_threshold: float = 1e-4,
        max_iters: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve both arms to torso-frame targets, returning per-arm 7-vectors.

        ``seeded=True`` continues from the current configuration (what a caller
        stepping along a path wants); ``seeded=False`` restarts from zero. The
        waist and finger joints stay where the configuration had them either
        way. Returns the best configuration reached within ``max_iters``;
        convergence is not guaranteed, so check with :meth:`forward_kinematics`.
        """
        if not seeded:
            self.configuration.update(np.zeros_like(self.configuration.data.qpos))
        self._hold_task.set_target_from_configuration(self.configuration)

        world_from_torso = self._world_from_torso()
        targets = {
            "left": (left_ee_pos, left_ee_quat_xyzw),
            "right": (right_ee_pos, right_ee_quat_xyzw),
        }
        for task, side in zip(self.tasks, ("left", "right")):
            position, quat_xyzw = targets[side]
            quat_wxyz = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)[[3, 0, 1, 2]]
            task.set_target(
                world_from_torso
                @ mink.SE3.from_rotation_and_translation(
                    mink.SO3(wxyz=quat_wxyz),
                    np.asarray(position, dtype=np.float64).reshape(3),
                )
            )

        all_tasks = [*self.tasks, self._hold_task]
        for _ in range(max_iters):
            velocity = mink.solve_ik(self.configuration, all_tasks, dt, solver, damping)
            self.configuration.integrate_inplace(velocity, dt)
            errors = [np.linalg.norm(task.compute_error(self.configuration)) for task in self.tasks]
            if max(errors) <= err_threshold:
                break

        qpos = self.configuration.q
        return qpos[self._arm_index["left"]].copy(), qpos[self._arm_index["right"]].copy()


__all__ = ["ARM_DOF", "WAIST_JOINTS", "G1Kinematics"]
