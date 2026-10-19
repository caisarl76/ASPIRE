"""The G1 plant: SONIC body link, camera, kinematics, and the one path that actuates.

``G1Env`` is the G1 twin of :class:`~cap_harness.yam_real.env.RealYamEnv` and
keeps its contract: :meth:`execute_action_batch` is the **only** way anything
commands the robot -- ``go_home``, ``set_gripper`` and ``set_hand_joints`` are
built on it -- and the episode recorder reads ``_last_action_input`` /
``_action_input_seq`` exactly as on YAM.

What is new is the stream. SONIC's planner mode drops back to its idle pose
when the ``planner`` topic goes quiet for a second, so between batches a
keep-alive thread re-sends the last emitted target every 20 ms. There is one
emitted-target chain (``_last_cmd``: q17 plus both hands): every publisher --
batch tick, keep-alive, engage hold -- steps it toward its goal under the
joint and hand step limits in :meth:`_send_tick`, so a target can never jump.
One re-entrant lock serialises every publish, and every publish refuses when
SONIC's feedback is older than ``sonic.feedback_stale_s``: a target sent blind
is a target nothing checks. When feedback returns after an outage the chain is
re-seeded from the measured pose (SONIC idled the arms meanwhile), and
:meth:`engage` runs again on request.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from cap_harness.g1_real import codec
from cap_harness.g1_real.body_client import G1BodyClient
from cap_harness.g1_real.codec import ARM_DOF, HAND_DOF, UPPER_BODY_DOF
from cap_harness.g1_real.config import G1StationConfig
from cap_harness.g1_real.control.controller import execute_upper_body_trajectory, step_toward
from cap_harness.g1_real.kinematics import G1Kinematics
from cap_harness.yam_real.action import YamActionBatch, resolve_action_batch

ARMS = ("left", "right")
CONTROL_CONTRACT_VERSION = "g1-sonic-v1-planner-stream"
KEEP_ALIVE_PERIOD_S = 0.02
#: Seconds a full open-to-closed hand transition is given when no duration is passed.
HAND_FULL_TRAVEL_S = 1.0
#: A gripper column within this of the measured scalar means "hold the hand as commanded".
HAND_SCALAR_HOLD_TOL = 1e-3

_log = logging.getLogger(__name__)


class G1Env:
    """Unitree G1 upper body over GEAR-SONIC, driven through a single action chokepoint."""

    dof = ARM_DOF
    #: How long engage() waits for SONIC's first feedback after sending ``start``.
    engage_feedback_timeout_s = 5.0
    # First engage only: SONIC's own start motion must finish before the seed.
    engage_settle_timeout_s = 5.0
    engage_settle_velocity_rad_s = 0.1
    engage_settle_hold_s = 0.3

    def __init__(
        self,
        config: G1StationConfig,
        body: G1BodyClient,
        *,
        camera: Any | None = None,
        kinematics: G1Kinematics | None = None,
    ) -> None:
        self.config = config
        self.body = body
        self._camera = camera
        self.kin = kinematics if kinematics is not None else G1Kinematics(config.model_xml)
        # Both observation and planning write the mink configuration; one lock
        # held across write-then-read, as on YAM.
        self._kin_lock = threading.Lock()
        # Serialises every publish. Re-entrant so a batch tick can call
        # _send_tick while the keep-alive loop holds it for its read-then-send.
        self._lock = threading.RLock()
        # The emitted-target chain: last published (q17, left_hand, right_hand).
        self._last_cmd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._streaming = False
        # Set while the keep-alive is paused on stale feedback; cleared by a fresh send.
        self._keep_alive_paused = False
        # Set by a stale refusal in _send_tick; the next fresh publish, whoever
        # makes it, re-seeds the chain from the measured pose first.
        self._resume_from_measured = False
        # Set when feedback was lost since the last engage; engage() then runs again.
        self._feedback_lost = False
        self._stop = threading.Event()
        self._keep_alive: threading.Thread | None = None
        self._engaged: dict[str, Any] | None = None
        self._close_hooks: list[Any] = []
        self._closed = False
        # Recorder contract, see RealYamEnv: last commanded (joints, gripper scalar) per arm.
        self._last_action_input: dict[str, tuple[np.ndarray, float]] = {}
        self._action_input_seq: dict[str, int] = dict.fromkeys(ARMS, 0)

    # -- observation -------------------------------------------------------

    def _raw(self) -> dict[str, Any]:
        raw = self.body.get_observations()
        if raw is None:
            raise RuntimeError("no SONIC feedback received yet")
        return raw

    def measured_command(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The measured state as a ``(q17, left_hand, right_hand)`` command triple."""
        raw = self._raw()
        q17 = codec.upper_body_from_arms(
            raw["waist"], raw["left_joint_pos"], raw["right_joint_pos"]
        )
        return (
            q17,
            np.asarray(raw["left_hand_q"], dtype=np.float64),
            np.asarray(raw["right_hand_q"], dtype=np.float64),
        )

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        """One arm's state: joints, hand, gripper scalar, and the torso-frame grasp pose."""
        if side not in ARMS:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        raw = self._raw()
        with self._kin_lock:
            left_pos, left_quat, right_pos, right_quat = self.kin.forward_kinematics(
                raw["left_joint_pos"], raw["right_joint_pos"], raw["waist"]
            )
        poses = {"left": (left_pos, left_quat), "right": (right_pos, right_quat)}
        hand_q = np.asarray(raw[f"{side}_hand_q"], dtype=np.float64).reshape(HAND_DOF)
        hand = self.config.hands[side]
        return {
            "joint_pos": np.asarray(raw[f"{side}_joint_pos"], dtype=np.float64).reshape(ARM_DOF),
            "joint_vel": np.asarray(raw[f"{side}_joint_vel"], dtype=np.float64).reshape(ARM_DOF),
            "gripper_pos": np.array([codec.hand_scalar(hand_q, hand.open, hand.closed)]),
            "hand_q": hand_q,
            "ee_pos": poses[side][0],
            "ee_quat": poses[side][1],
        }

    def read_camera(self) -> Any | None:
        return None if self._camera is None else self._camera.read()

    def read_aux_camera(self, role: str) -> Any | None:
        del role  # no image-only cameras on this station
        return None

    @property
    def aux_camera_roles(self) -> tuple[str, ...]:
        return ()

    @property
    def camera_aliases(self) -> tuple[str, ...]:
        return () if self._camera is None else (str(self.config.camera.role),)

    def camera_frame(self, alias: str) -> Any | None:
        if self._camera is not None and alias == str(self.config.camera.role):
            return self._camera.read()
        return None

    def render_rgb(self, alias: str) -> np.ndarray | None:
        frame = self.camera_frame(alias)
        return None if frame is None else np.asarray(frame.rgb, dtype=np.uint8)

    def add_close_hook(self, hook: Any) -> None:
        """Call ``hook`` on close, before the devices go down; failures are logged."""
        self._close_hooks.append(hook)

    # -- actuation ---------------------------------------------------------

    def last_command(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copies of the last emitted ``(q17, left_hand, right_hand)`` target."""
        with self._lock:
            if self._last_cmd is None:
                raise RuntimeError("no target has been emitted yet: call engage() first")
            q17, lh, rh = self._last_cmd
            return q17.copy(), lh.copy(), rh.copy()

    def _send_tick(
        self,
        q17: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        *,
        hz: float | None = None,
        allow_stale: bool = False,
    ) -> dict[str, Any] | None:
        """Step the chain toward the goal and publish it, or refuse when feedback is stale.

        The joint velocity sent is the step over one period at ``hz`` (the
        stream rate by default); a holding tick therefore sends zero. The first
        publish after a stale refusal re-seeds the chain from the measured pose
        before stepping: SONIC idled the arms meanwhile, and this is the one
        place every publisher passes through. ``allow_stale`` exists for one
        caller only, the pre-start hold in :meth:`engage`: SONIC publishes no
        feedback until its control loop runs, so that hold is sent blind by
        design and seeds nothing from a measurement.
        """
        with self._lock:
            age = self.body.feedback_age_s()
            limit = self.config.sonic.feedback_stale_s
            if age > limit and not allow_stale:
                self._resume_from_measured = True
                return {
                    "success": False,
                    "reason": f"feedback stale ({age:.3f} s > {limit:.3f} s)",
                }
            if self._last_cmd is None:
                raise RuntimeError("no target chain to step from: call engage() first")
            if self._resume_from_measured and not allow_stale:
                self._last_cmd = self.measured_command()
                self._resume_from_measured = False
                _log.warning("feedback resumed: chain re-seeded from the measured pose")
            prev_q, prev_lh, prev_rh = self._last_cmd
            hands = self.config.hands
            q = step_toward(
                prev_q,
                np.asarray(q17, dtype=np.float64).reshape(UPPER_BODY_DOF),
                float(self.config.controller.max_joint_step_rad),
            )
            lh = step_toward(
                prev_lh,
                np.asarray(left_hand, dtype=np.float64).reshape(HAND_DOF),
                float(hands["left"].max_step_rad),
            )
            rh = step_toward(
                prev_rh,
                np.asarray(right_hand, dtype=np.float64).reshape(HAND_DOF),
                float(hands["right"].max_step_rad),
            )
            rate = float(self.config.controller.command_stream_hz if hz is None else hz)
            self.body.command_upper_body(q, (q - prev_q) * rate, lh, rh)
            self._last_cmd = (q, lh, rh)
        return None

    def _keep_alive_loop(self) -> None:
        limit = self.config.sonic.feedback_stale_s
        while not self._stop.wait(KEEP_ALIVE_PERIOD_S):
            if self._streaming:
                continue
            # A dead keep-alive is a physical event (SONIC idles the arms after
            # 1 s of silence), so a failing send is logged and retried, never fatal.
            try:
                with self._lock:
                    if self._streaming or self._last_cmd is None:
                        continue
                    goal = self._last_cmd
                    if self._resume_from_measured and self.body.feedback_age_s() <= limit:
                        # Feedback is back after an outage: hold where the arms
                        # are now rather than ramp back to the pre-outage target
                        # (_send_tick re-seeds the chain itself).
                        goal = self.measured_command()
                    refused = self._send_tick(*goal)
            except Exception as exc:
                _log.warning("keep-alive send failed: %s: %s", type(exc).__name__, exc)
                continue
            if refused is None:
                self._keep_alive_paused = False
            elif not self._keep_alive_paused:
                self._keep_alive_paused = True
                self._feedback_lost = True
                _log.warning("keep-alive paused: %s", refused["reason"])

    @property
    def keep_alive_running(self) -> bool:
        return self._keep_alive is not None and self._keep_alive.is_alive()

    def engage(self) -> dict[str, Any]:
        """Hold the measured pose on the stream, then hand SONIC its ``start``.

        Seeds the chain (and the body client's hand limiter) from the measured
        pose first. On a healthy stream a second call reports ``already_engaged``
        and sends nothing; after a feedback outage, or with a dead keep-alive,
        the full hold and ``start`` run again. Refuses before any publish when
        the feedback is stale, since the hold command is seeded from it.
        """
        if self._engaged is not None and self.keep_alive_running and not self._feedback_lost:
            return {**self._engaged, "already_engaged": True}
        started = False
        if self.body.get_observations() is None:
            # SONIC publishes feedback only once its control loop runs (the
            # deploy's CONTROL state), so the very first engage has nothing to
            # seed from: send start alone, wait for feedback and still arms,
            # then continue exactly as a re-engage would.
            self._start_without_feedback()
            started = True
        age = self.body.feedback_age_s()
        limit = self.config.sonic.feedback_stale_s
        if age > limit:
            raise RuntimeError(f"feedback stale ({age:.3f} s > {limit:.3f} s); not engaging")
        q17, lh, rh = self.measured_command()
        self.body.seed_hands(lh, rh)
        with self._lock:
            self._last_cmd = (q17, lh, rh)
            self._resume_from_measured = False  # seeded from measured just now
        dt = 1.0 / float(self.config.controller.command_stream_hz)
        ticks = max(1, round(float(self.config.sonic.engage_hold_s) / dt))
        self._streaming = True  # the hold is the only publisher meanwhile
        try:
            for _ in range(ticks):
                refused = self._send_tick(q17, lh, rh)
                if refused is not None:
                    raise RuntimeError(refused["reason"])
                time.sleep(dt)
        finally:
            self._streaming = False
        if not started:
            self.body.send_start()
        self._feedback_lost = False
        if not self.keep_alive_running:
            self._keep_alive = threading.Thread(
                target=self._keep_alive_loop, name="g1-keep-alive", daemon=True
            )
            self._keep_alive.start()
        self._engaged = {"success": True, "hold_ticks": ticks, "already_engaged": False}
        return dict(self._engaged)

    def _start_without_feedback(self) -> None:
        """Send ``start`` with nothing streamed, then wait for feedback and still arms.

        Before ``start`` SONIC publishes no feedback, so ASPIRE cannot know where
        the arms are. Streaming any target here (this used to hold the profile
        home) makes SONIC's first control tick drive the arms straight to it from
        wherever they hang: an abrupt jump on the real robot, invisible in sim
        where the arms start near home. With only ``start``, SONIC moves the arms
        through its own idle motion, which is its normal start. Once feedback
        flows and the arms are still, :meth:`engage` seeds the chain from the
        measurement, so the first ASPIRE target is where the arms already are.
        Raises when SONIC stays silent past ``engage_feedback_timeout_s``.
        """
        dt = 1.0 / float(self.config.controller.command_stream_hz)
        self.body.send_start()
        deadline = time.monotonic() + float(self.engage_feedback_timeout_s)
        while self.body.get_observations() is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"no SONIC feedback within {self.engage_feedback_timeout_s:.1f} s "
                    "of start; not engaging"
                )
            time.sleep(dt)
        # Seeding mid-motion would stop the arms where they happen to be; wait
        # for SONIC's own start motion to finish first (bounded: a robot that
        # never settles is seeded anyway, which stops it rather than jumps it).
        still_since: float | None = None
        deadline = time.monotonic() + float(self.engage_settle_timeout_s)
        while time.monotonic() < deadline:
            obs = self.body.get_observations()
            if obs is not None:
                speed = max(
                    float(np.max(np.abs(obs["left_joint_vel"]))),
                    float(np.max(np.abs(obs["right_joint_vel"]))),
                )
                now = time.monotonic()
                if speed < self.engage_settle_velocity_rad_s:
                    still_since = now if still_since is None else still_since
                    if now - still_since >= self.engage_settle_hold_s:
                        _log.info("SONIC started; arms still, seeding from the measured pose")
                        return
                else:
                    still_since = None
            time.sleep(dt)
        _log.warning(
            "arms still moving %.1f s after SONIC start; seeding from the measured pose anyway",
            self.engage_settle_timeout_s,
        )

    def _not_ready(self) -> dict[str, Any] | None:
        """The refusal for a command before engage or after the keep-alive died, else ``None``."""
        if self._engaged is None:
            return {"success": False, "reason": "not engaged: call engage() first"}
        if not self.keep_alive_running:
            return {"success": False, "reason": "keep-alive not running; re-engage the station"}
        return None

    def execute_action_batch(
        self,
        batch: YamActionBatch | dict[str, Any],
        *,
        command_hz: float | None = None,
        start_interp_s: float = 0.0,
        settle_s: float = 0.2,
        playback_speed: float = 1.0,
    ) -> dict[str, Any]:
        """Execute one typed action batch. The only path to the robot.

        ``batch.meta["hand_joints"]`` may carry explicit ``{side: (N, 7)}`` rows
        for a side. Otherwise a hand is held as last commanded when its gripper
        is unspecified, or when its column equals the measured scalar or the
        chain hand's scalar on every row -- a joint-level grasp must not be
        projected onto the open/closed line by an arm move -- and only a
        changed scalar is mapped through ``codec.hand_pose``. A hold built from
        a measurement is racy while the hand still moves (the next feedback
        message changes the scalar), so holding callers pass the chain scalar
        or no gripper at all. The waist holds the profile's target on every tick.

        Refused before anything is resolved or published when the stream is
        not engaged (nothing would hold the target afterwards), and when a
        caller-supplied arm row lies outside the profile limits.
        """
        if (refused := self._not_ready()) is not None:
            return refused
        if isinstance(batch, dict):
            batch = YamActionBatch(**batch)
        resolved = resolve_action_batch(self, batch)
        for side, rows in (
            ("left", resolved.left_joint_positions),
            ("right", resolved.right_joint_positions),
        ):
            arm = self.config.arms[side]
            bad = np.argwhere((rows < arm.lower) | (rows > arm.upper))
            if bad.size:
                row, i = (int(v) for v in bad[0])
                return {
                    "success": False,
                    "reason": (
                        f"{side} joint {i} target {rows[row, i]:.4f} is outside the "
                        f"configured limits [{arm.lower[i]:.4f}, {arm.upper[i]:.4f}]"
                    ),
                }
        self._note_action_input(resolved)
        n = int(resolved.timestamps.size)
        overrides = resolved.meta.get("hand_joints") or {}
        scalars = {
            "left": resolved.left_gripper_positions,
            "right": resolved.right_gripper_positions,
        }
        raw = self._raw()
        _, *held = self.last_command()
        hands: dict[str, np.ndarray] = {}
        for side, held_hand in zip(ARMS, held, strict=True):
            if side in overrides:
                hands[side] = np.asarray(overrides[side], dtype=np.float64).reshape(n, HAND_DOF)
                continue
            hand = self.config.hands[side]
            # The resolver fills an unspecified gripper with a measurement; here
            # "unspecified" itself means hold (a joint7 row may carry its own).
            unspecified = (
                getattr(batch, f"{side}_gripper") is None
                and np.asarray(getattr(batch, side), dtype=np.float64).reshape(n, -1).shape[1]
                <= self.dof
            )
            column = np.asarray(scalars[side], dtype=np.float64).reshape(-1)
            holds = (
                codec.hand_scalar(raw[f"{side}_hand_q"], hand.open, hand.closed),
                codec.hand_scalar(held_hand, hand.open, hand.closed),
            )
            if unspecified or any(
                np.all(np.abs(column - s) <= HAND_SCALAR_HOLD_TOL) for s in holds
            ):
                hands[side] = np.tile(held_hand, (n, 1))
            else:
                hands[side] = np.stack([codec.hand_pose(p, hand.open, hand.closed) for p in column])
        self._streaming = True
        try:
            result = execute_upper_body_trajectory(
                self,
                resolved.timestamps,
                resolved.left_joint_positions,
                resolved.right_joint_positions,
                hands["left"],
                hands["right"],
                command_hz=float(command_hz or self.config.controller.command_stream_hz),
                start_interp_s=start_interp_s,
                settle_s=settle_s,
                playback_speed=playback_speed,
            )
        finally:
            self._streaming = False
        result.setdefault("source", resolved.source)
        result.setdefault("input_space", resolved.input_space)
        return result

    def _note_action_input(self, resolved: Any) -> None:
        """Publish the batch's final target for the episode recorder (see RealYamEnv)."""
        for side, rows, grip in (
            ("left", resolved.left_joint_positions, resolved.left_gripper_positions),
            ("right", resolved.right_joint_positions, resolved.right_gripper_positions),
        ):
            self._last_action_input[side] = (
                np.asarray(rows, dtype=np.float64)[-1].copy(),
                float(np.asarray(grip, dtype=np.float64).reshape(-1)[-1]),
            )
            self._action_input_seq[side] += 1

    def _hand_transition(
        self, side: str, target: np.ndarray, duration_s: float | None, *, source: str
    ) -> dict[str, Any]:
        """Two waypoints: arms held at the chain target, one hand moving to ``target``.

        The window scales with the travel. Both hands go as explicit
        ``hand_joints`` rows -- the other hand is the chain hand tiled -- so the
        batch never has to decide whether a scalar column means "hold".
        """
        if (refused := self._not_ready()) is not None:
            return refused
        q17, *held = self.last_command()
        start = held[ARMS.index(side)]
        hand = self.config.hands[side]
        rows = {name: np.tile(h, (2, 1)) for name, h in zip(ARMS, held, strict=True)}
        rows[side] = np.stack([start, target])
        columns = {
            name: [[codec.hand_scalar(r, cfg.open, cfg.closed)] for r in rows[name]]
            for name, cfg in self.config.hands.items()
        }
        travel = float(np.max(np.abs(target - start)) / np.max(np.abs(hand.closed - hand.open)))
        duration = (
            max(0.05, travel * HAND_FULL_TRAVEL_S)
            if duration_s is None
            else max(0.02, float(duration_s))
        )
        _, left, right = codec.arms_from_upper_body(q17)
        arms = self.config.arms
        left = np.clip(left, arms["left"].lower, arms["left"].upper)
        right = np.clip(right, arms["right"].lower, arms["right"].upper)
        batch = YamActionBatch.joint_abs(
            [0.0, duration],
            [left, left],
            [right, right],
            columns["left"],
            columns["right"],
            source=source,
            meta={"side": side, "hand_joints": rows},
        )
        return self.execute_action_batch(batch, settle_s=0.0)

    def set_gripper(
        self, side: str, position: float, *, duration_s: float | None = None
    ) -> dict[str, Any]:
        """Move one hand to a point on its open/closed line while both arms hold."""
        if side not in ARMS:
            return {"success": False, "reason": f"invalid side: {side!r}"}
        hand = self.config.hands[side]
        goal = float(np.clip(position, 0.0, 1.0))
        result = self._hand_transition(
            side, codec.hand_pose(goal, hand.open, hand.closed), duration_s, source="cap_gripper"
        )
        result.update({"side": side, "gripper": goal})
        return result

    def set_hand_joints(
        self, side: str, joints7: Any, *, duration_s: float | None = None
    ) -> dict[str, Any]:
        """Drive one Dex3 hand to explicit finger joints while both arms hold."""
        if side not in ARMS:
            return {"success": False, "reason": f"invalid side: {side!r}"}
        hand = self.config.hands[side]
        target = np.asarray(joints7, dtype=np.float64).reshape(-1)
        if target.shape != (HAND_DOF,) or not np.all(np.isfinite(target)):
            return {"success": False, "reason": "hand joints must be 7 finite values"}
        if np.any(target < hand.lower) or np.any(target > hand.upper):
            return {"success": False, "reason": f"{side} hand joints outside the profile limits"}
        result = self._hand_transition(side, target, duration_s, source="cap_hand_joints")
        result.update({"side": side, "hand_joints": target.tolist()})
        return result

    def go_home(self, *, duration_s: float = 3.0, keep_grippers: bool = True) -> dict[str, Any]:
        """Drive both arms to the profile's home pose with a start interpolation.

        Hands go as explicit ``hand_joints`` rows: the chain hand when
        ``keep_grippers``, else every joint at ``open`` -- the scalar cannot see
        a joint that is equal in ``open`` and ``closed`` (the thumb base).
        """
        if (refused := self._not_ready()) is not None:
            return refused
        _, *held = self.last_command()
        hands, columns = {}, {}
        for side, chain in zip(ARMS, held, strict=True):
            cfg = self.config.hands[side]
            hands[side] = chain if keep_grippers else codec.hand_pose(1.0, cfg.open, cfg.closed)
            columns[side] = [[codec.hand_scalar(hands[side], cfg.open, cfg.closed)]]
        batch = YamActionBatch.joint_abs(
            [0.0],
            [self.config.arms["left"].home_joints],
            [self.config.arms["right"].home_joints],
            columns["left"],
            columns["right"],
            source="cap_home",
            meta={"hand_joints": {side: hands[side].reshape(1, HAND_DOF) for side in ARMS}},
        )
        return self.execute_action_batch(
            batch, start_interp_s=max(0.0, float(duration_s)), settle_s=0.2
        )

    # -- lifecycle ---------------------------------------------------------

    def control_contract(self) -> dict[str, Any]:
        """The versioned plant description recorded actions were executed under."""
        controller = self.config.controller
        return {
            "version": CONTROL_CONTRACT_VERSION,
            "action_semantics": (
                "joint7+hand7 absolute targets per arm, SONIC-17 upper body stream "
                "at command_stream_hz"
            ),
            "action_space": "joint_abs",
            "command_stream_hz": float(controller.command_stream_hz),
            "max_joint_step_rad": float(controller.max_joint_step_rad),
            "interpolation": "linear, wall-clock time-sampled, per-tick step clamp",
            "tracker": "GEAR-SONIC planner mode over the ZMQ planner topic",
            "owner": "cap_harness.g1_real.control.controller (level 1)",
            "gripper_action": "continuous position target in [0, 1]; 0.0=closed, 1.0=open",
            "station": self.config.station,
            "config_sha256": self.config.config_sha256,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._keep_alive is not None:
            self._keep_alive.join(timeout=1.0)
        for hook in reversed(self._close_hooks):
            try:
                hook()
            except Exception as exc:  # a hook's failure is not the station's
                _log.warning("close hook failed: %s: %s", type(exc).__name__, exc)
        errors: list[str] = []
        for name, device in (("body", self.body), ("camera", self._camera)):
            if device is None:
                continue
            try:
                device.close()
            except Exception as exc:  # every device gets its close attempt
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("G1 station cleanup failed: " + "; ".join(errors))


__all__ = ["ARMS", "CONTROL_CONTRACT_VERSION", "G1Env"]
