"""Sim experiments for the G1 port; results go to docs/g1-real.md ("Sim experiments").

WARNING: moves the robot of the station it builds (g1-sim by default, on the sim's ports).
Each subcommand builds the g1-sim station itself, so no CAP server may hold the sim's
command port (5566). Start the sim first with ``scripts/g1_sim.sh up``. Verdicts come from
the sim's own truth: object positions and robot-object contacts in every camera message.

  engage    first engage on a fresh deploy (run right after `up`), then drop: the hold
            on the standing robot is still and on target
  heading   after `G1_SIM_YAW_DEG=60 scripts/g1_sim.sh up`: engage, drop, the heading holds
  retreat   hand over the case, then Home: the old straight Home and the retreating one
  grip      palm to the bottle, contact-aware close, lift 5 cm, put back, open
  outage    feedback cut for 2 s while SONIC runs, then resume and re-engage
  curobo    cuRobo planning with the live depth scene: over the case and back
  watch     record the sim's truth to a JSONL file (for agent runs); --summary reads one
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from cap_harness.api import CapApi
from cap_harness.contracts import MotionStrategy, Pose
from cap_harness.g1_real import codec
from cap_harness.g1_real import env as env_module
from cap_harness.g1_real.adapter import G1RealAdapter
from cap_harness.g1_real.body_client import SonicZmqBodyClient
from cap_harness.g1_real.cameras import ComposedCameraRgbd
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.g1_real.kinematics import G1Kinematics
from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider
from cap_harness.yam_real.cameras import SyntheticCamera

ROOT = Path(__file__).resolve().parents[2]
STATION = "g1-sim"
INTERP = MotionStrategy(ik_solver="mink", trajectory_planner="interpolation")
HANDSHAKE_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
BOTTLE_RADIUS_M = 0.033
DEX3_PALM = (0.06, 0.02)  # palm contact = site - a f + b n (bottle_touch recipe)


# -- sim plumbing ---------------------------------------------------------------


class SimTruth:
    """Latest object positions and every robot-object contact from the sim's camera stream."""

    def __init__(self, port: int) -> None:
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.subscribe(b"")
        self._sock.connect(f"tcp://127.0.0.1:{port}")
        self._lock = threading.Lock()
        self.latest: dict = {}
        self._contacts: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message = msgpack.unpackb(
                    self._sock.recv(), object_hook=msgpack_numpy.decode, raw=False
                )
            except zmq.Again:
                continue
            truth = message.get("sim_truth", {})
            with self._lock:
                self.latest = truth
                if truth.get("contacts"):
                    self._contacts.append((time.monotonic(), truth["contacts"]))

    def position(self, name: str) -> np.ndarray:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if name in self.latest:
                    return np.asarray(self.latest[name], dtype=float)
            time.sleep(0.05)
        raise RuntimeError(f"no sim truth for {name}")

    def touches(self, obj: str, since: float, prefix: str = "") -> dict[str, list[float]]:
        """Robot links that touched ``obj`` since ``since``: [peak force, x, y, z in the link]."""
        peak: dict[str, list[float]] = {}
        with self._lock:
            for stamp, contacts in self._contacts:
                if stamp >= since:
                    for link, hit in contacts.get(obj, {}).items():
                        if link.startswith(prefix) and hit[0] >= peak.get(link, [0.0])[0]:
                            peak[link] = list(hit)
        return peak

    def close(self) -> None:
        self._stop.set()


def standing(truth: SimTruth) -> None:
    """Drop the robot if it still hangs on the band, and let it settle."""
    deadline = time.monotonic() + 5.0
    while "band" not in truth.latest and time.monotonic() < deadline:
        time.sleep(0.1)
    if truth.latest.get("band"):
        sim("drop")
        time.sleep(10.0)


def sim(command: str, *args: str) -> None:
    subprocess.run(
        [str(ROOT / "scripts/g1_sim.sh"), command, *args], check=True, capture_output=True
    )


def build(*, camera: bool = False, feedback_port: int | None = None, curobo_url: str | None = None):
    cfg = load_g1_station_config(STATION)
    if feedback_port is not None:
        cfg = replace(cfg, sonic=replace(cfg.sonic, feedback_port=feedback_port))
    body = SonicZmqBodyClient(
        cfg.sonic, hand_max_step_rad={s: h.max_step_rad for s, h in cfg.hands.items()}
    )
    cam = (
        ComposedCameraRgbd(
            cfg.camera.host,
            cfg.camera.port,
            role=cfg.camera.role,
            depth_scale_m=cfg.camera.depth_scale_m,
            base_from_camera=cfg.camera.base_from_camera,
            floor_below_z_m=cfg.camera.floor_below_z_m,
        )
        if camera
        else SyntheticCamera()
    )
    env = G1Env(cfg, body, camera=cam)
    options: dict = {
        "ik_providers": {"mink": YamKinematicsIKProvider(G1Kinematics(cfg.model_xml), dof=7)}
    }
    if curobo_url:
        from cap_harness.providers.curobo import CuRoboProvider

        planner = CuRoboProvider(
            base_url=curobo_url, control_period_s=1 / cfg.controller.control_frequency_hz
        )
        options["ik_providers"]["curobo"] = planner
        options["trajectory_planning_providers"] = {"curobo": planner}
    adapter = G1RealAdapter(env, allow_physical_motion=True)
    api = CapApi(adapter, default_camera=cfg.camera.role, **options)
    return env, adapter, api


class ArmSampler:
    """Samples measured arm joints and velocities (and the base yaw) from SONIC feedback."""

    def __init__(self, env: G1Env, hz: float = 100.0) -> None:
        self.env, self.period = env, 1.0 / hz
        self.rows: list[tuple[float, np.ndarray, np.ndarray, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            raw = self.env.body.get_observations()
            if raw is not None and raw["age_s"] < 0.2:
                q = np.concatenate([raw["left_joint_pos"], raw["right_joint_pos"]])
                dq = np.concatenate([raw["left_joint_vel"], raw["right_joint_vel"]])
                w, x, y, z = np.asarray(raw["base_quat"], float)
                yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                self.rows.append((time.monotonic(), q, dq, yaw))

    def displacement(self, start: float, end: float) -> float:
        """Largest arm-joint excursion from the first sample at or after ``start``."""
        rows = [q for t, q, _, _ in self.rows if start <= t <= end]
        return float(np.max(np.abs(np.array(rows) - rows[0]))) if rows else float("nan")

    def peak_speed(self, start: float, end: float = math.inf) -> float:
        speeds = [float(np.max(np.abs(dq))) for t, _, dq, _ in self.rows if start <= t <= end]
        return max(speeds, default=float("nan"))

    def yaws(self) -> np.ndarray:
        return np.array([yaw for *_, yaw in self.rows])

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def report(rows: list[tuple[str, bool | None, str]]) -> int:
    width = max(len(name) for name, _, _ in rows)
    for name, ok, detail in rows:
        print(f"{'SKIP' if ok is None else 'PASS' if ok else 'FAIL':4s}  {name:{width}s}  {detail}")
    return 0 if all(ok is not False for _, ok, _ in rows) else 1


def hand(env: G1Env, side: str = "right") -> np.ndarray:
    return np.asarray(env.get_observations(side)["ee_pos"], dtype=float)


def current_quat(env: G1Env, side: str = "right") -> np.ndarray:
    return np.asarray(env.get_observations(side)["ee_quat"], dtype=float)[[3, 0, 1, 2]]


def move(
    api: CapApi, position, quat=HANDSHAKE_WXYZ, *, strategy=INTERP, side="right", tolerance=0.04
):
    """A setup move: SONIC's few-cm tracking offset is accepted (operator decision)."""
    target = Pose(np.asarray(position, float), np.asarray(quat, float), "torso")
    return api.move_to_pose(target, tolerance, arm=side, strategy=strategy, max_steps=600)


# -- experiments ----------------------------------------------------------------


def exp_engage(_: argparse.Namespace) -> int:
    """Fresh engage, then drop: once standing, ASPIRE's hold must be still and on target.

    While the robot hangs on the band SONIC swings the arms (several rad/s), so the
    takeover is judged on the standing robot: the commanded arms against the measured
    ones, and the arm speed during a 3 s hold.
    """
    env, adapter, _api = build()
    sampler = ArmSampler(env)
    try:
        t0 = time.monotonic()
        result = env.engage()
        t1 = time.monotonic()
        sim("drop")
        time.sleep(8.0)
        t2 = time.monotonic()
        time.sleep(3.0)
        q17, *_ = env.last_command()
        _, left, right = codec.arms_from_upper_body(q17)
        raw = env.body.get_observations()
        measured = np.concatenate([raw["left_joint_pos"], raw["right_joint_pos"]])
        gap = float(np.max(np.abs(np.concatenate([left, right]) - measured)))
        hold = sampler.peak_speed(t2, t2 + 3.0)
        return report(
            [
                ("fresh engage", not result["already_engaged"], f"{t1 - t0:.1f} s"),
                (
                    "arm speed while hanging",
                    None,
                    f"peak {sampler.peak_speed(t0, t1):.2f} rad/s (SONIC on the band)",
                ),
                ("hold once standing: speed", hold < 0.3, f"peak {hold:.2f} rad/s over 3 s"),
                (
                    "hold once standing: command vs measured",
                    gap < 0.15,
                    f"{gap:.3f} rad (SONIC's tracking offset)",
                ),
            ]
        )
    finally:
        sampler.stop()
        adapter.close()


def exp_heading(args: argparse.Namespace) -> int:
    if args.legacy_heading:  # the pre-fix facing (raw IMU yaw), to show the test catches it
        from cap_harness.g1_real.codec import heading_yaw_from_quat_wxyz

        SonicZmqBodyClient._heading = staticmethod(
            lambda msg: heading_yaw_from_quat_wxyz(np.asarray(msg["base_quat"]).reshape(-1))
        )
    env, adapter, _api = build()
    sampler = ArmSampler(env)
    try:
        env.engage()
        time.sleep(1.0)
        before = float(np.median(sampler.yaws()[-50:]))
        sim("drop")
        time.sleep(12.0)
        after = float(np.median(sampler.yaws()[-50:]))
        turn = (after - before + 180.0) % 360.0 - 180.0
        return report(
            [
                ("start heading", None, f"{before:+.1f} deg (sim started turned)"),
                (
                    "heading after engage and drop",
                    abs(turn) < 10.0,
                    f"{after:+.1f} deg, turned {turn:+.1f} deg",
                ),
            ]
        )
    finally:
        sampler.stop()
        adapter.close()


def exp_retreat(args: argparse.Namespace) -> int:
    truth = SimTruth(args.camera_port)
    env, adapter, api = build()
    rows = []
    try:
        env.engage()
        standing(truth)
        for label, retreat in (("straight Home (old)", False), ("retreating Home (new)", True)):
            sim("reset")
            time.sleep(1.5)
            api.go_home(arm="both")
            move(api, [0.40, -0.10, 0.10])
            start = hand(env)
            case = truth.position("aspire_case")
            if not (start[0] > case[0] - 0.20 and start[2] > case[2] + 0.42):
                rows.append(
                    (f"{label}: hand over the case", False, f"hand at {np.round(start, 3)}")
                )
                continue
            saved = env_module.RETREAT_ABOVE_HOME_M
            env_module.RETREAT_ABOVE_HOME_M = saved if retreat else 99.0
            try:
                since = time.monotonic()
                result = adapter.go_home(arm="both")
                elapsed = time.monotonic() - since
            finally:
                env_module.RETREAT_ABOVE_HOME_M = saved
            time.sleep(0.5)
            hit = sorted(truth.touches("aspire_case", since))
            waypoints = (
                result.diagnostics.get("retreat_waypoints", "-")
                if hasattr(result, "diagnostics")
                else "-"
            )
            ok = (not hit) if retreat else None
            rows.append(
                (
                    label,
                    ok,
                    f"from {np.round(start, 2)}: {elapsed:.1f} s, retreat waypoints {waypoints}, "
                    f"case contacts {hit or 'none'}",
                )
            )
        return report(rows)
    finally:
        truth.close()
        adapter.close()


def exp_grip(args: argparse.Namespace) -> int:
    if args.squeeze is not None:
        import cap_harness.g1_real.adapter as adapter_module

        adapter_module.GRIP_SQUEEZE = args.squeeze
    truth = SimTruth(args.camera_port)
    env, adapter, api = build()
    rows = []
    try:
        env.engage()
        standing(truth)
        sim("reset")
        time.sleep(2.0)
        api.go_home(arm="both")
        api.open_gripper(arm="right")
        axis = truth.position("aspire_bottle") + [0.0, 0.0, 0.02]  # a little above the body center
        f, n = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        a, b = DEX3_PALM
        contact_target = axis - BOTTLE_RADIUS_M * n
        site = contact_target + a * f - b * n
        move(api, [0.25, -0.25, axis[2] + 0.12])
        pre = move(api, site - 0.08 * n)
        rows.append(
            (
                "pre-contact pose",
                pre.ok,
                f"error {pre.final_errors.get('right', float('nan')) * 1000:.0f} mm",
            )
        )
        since = time.monotonic()
        touched: dict[str, float] = {}
        # 1.5 cm steps to the nominal contact site and up to 4 cm past it: SONIC falls
        # short moving the hand toward the body (as on the robot); contact decides.
        for step in np.arange(0.015, 0.12 + 1e-9, 0.015):
            move(api, site - (0.08 - step) * n)
            time.sleep(0.3)
            touched = truth.touches("aspire_bottle", since, prefix="right_")
            if touched:
                break
        first = max(touched, key=lambda link: touched[link][0]) if touched else None
        rows.append(
            (
                "hand reaches the bottle",
                bool(touched),
                f"after {step * 100:.1f} cm of approach, links {sorted(touched) or 'none'}",
            )
        )
        # The Dex3 palm is part of the wrist-yaw link; past ~5 cm from its origin is palm.
        palm_or_fingers = first is not None and (
            not first.startswith("right_wrist") or touched[first][1] > 0.05
        )
        rows.append(
            (
                "contact is palm or fingers",
                None if first is None else palm_or_fingers,
                f"strongest {first} at {touched[first][1:] if first else '-'} in the link",
            )
        )
        start_z = truth.position("aspire_bottle")[2]
        closing = time.monotonic()
        close = api.close_gripper(arm="right")
        d = close.diagnostics
        during = truth.touches("aspire_bottle", closing, prefix="right_")
        rows.append(
            (
                "close stops at contact",
                bool(d.get("contact")),
                f"{d.get('completion')} closure {d.get('closure')} lag {d.get('max_finger_lag_rad')} rad; "
                f"links touching while closing {sorted(during) or 'none'}",
            )
        )
        above = hand(env) + [0.0, 0.0, 0.05]
        move(api, above, env.get_observations("right")["ee_quat"][[3, 0, 1, 2]])
        time.sleep(0.5)
        lifted = truth.position("aspire_bottle")[2] - start_z
        rows.append(
            ("bottle lifted with the hand", lifted > 0.03, f"bottle rose {lifted * 100:.1f} cm")
        )
        move(
            api,
            hand(env) - [0.0, 0.0, 0.05],
            env.get_observations("right")["ee_quat"][[3, 0, 1, 2]],
        )
        api.open_gripper(arm="right")
        move(api, hand(env) - 0.08 * n)
        move(api, hand(env) + [0.0, 0.0, 0.10])
        bottle = truth.position("aspire_bottle")
        rows.append(
            (
                "bottle stands after release",
                abs(bottle[2] - start_z) < 0.02,
                f"z change {(bottle[2] - start_z) * 100:+.1f} cm",
            )
        )
        adapter.go_home(arm="both")
        return report(rows)
    finally:
        truth.close()
        adapter.close()


class FeedbackProxy:
    """Forwards SONIC feedback 5567 -> 5568 unless paused (cuts feedback, SONIC keeps running)."""

    def __init__(self, src: int, dst: int) -> None:
        context = zmq.Context.instance()
        self._sub = context.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVTIMEO, 100)
        self._sub.subscribe(b"")
        self._sub.connect(f"tcp://127.0.0.1:{src}")
        self._pub = context.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{dst}")
        self.paused = False
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._sub.recv()
            except zmq.Again:
                continue
            if not self.paused:
                self._pub.send(frame)

    def close(self) -> None:
        self._stop.set()


def exp_outage(_: argparse.Namespace) -> int:
    proxy = FeedbackProxy(5567, 5568)
    time.sleep(0.5)
    env, adapter, api = build(feedback_port=5568)
    sampler = ArmSampler(env)
    rows = []
    try:
        env.engage()
        time.sleep(1.5)
        proxy.paused = True
        time.sleep(2.0)
        refused = move(api, hand(env) + [0.0, 0.0, 0.03], current_quat(env))
        reason = refused.error.message if refused.error else "moved"
        rows.append(
            ("motion refused while blind", not refused.ok and "stale" in reason, reason[:70])
        )
        proxy.paused = False
        resumed = time.monotonic()
        time.sleep(1.5)
        moved = sampler.displacement(resumed, resumed + 1.5)
        rows.append(
            (
                "no jump at resume (hold measured)",
                moved < 0.1,
                f"arms moved {moved:.3f} rad, peak {sampler.peak_speed(resumed, resumed + 1.5):.2f} rad/s",
            )
        )
        re = env.engage()
        t = time.monotonic()
        time.sleep(1.5)
        rows.append(("re-engage runs hold + start", not re["already_engaged"], ""))
        moved = sampler.displacement(t, t + 1.5)
        rows.append(("no jump at re-engage", moved < 0.1, f"arms moved {moved:.3f} rad"))
        after = move(api, hand(env) + [0.0, 0.0, 0.03], current_quat(env))
        rows.append(
            (
                "moves again",
                after.ok,
                f"error {after.final_errors.get('right', float('nan')) * 1000:.0f} mm",
            )
        )
        return report(rows)
    finally:
        sampler.stop()
        adapter.close()
        proxy.close()


def exp_curobo(args: argparse.Namespace) -> int:
    """CuRobo with the live depth scene: from home, over the case beside the bottle, back.

    From home the hand sits below the case top a few cm from its near face, the narrow
    start the real robot has after Home; the second leg tests lifting it first with an
    interpolation move (the tool guide's advice on a refusal).
    """
    truth = SimTruth(args.camera_port)
    env, adapter, api = build(camera=True, curobo_url=args.curobo_url)
    plan = MotionStrategy(ik_solver="mink", trajectory_planner="curobo")
    cfg = env.config
    home_fk = env.kin.forward_kinematics(
        cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )
    home_pos, home_quat = np.asarray(home_fk[2]), np.asarray(home_fk[3])[[3, 0, 1, 2]]
    rows = []

    def leg(
        label: str, target, quat=HANDSHAKE_WXYZ, strategy=plan, expect: bool | None = True
    ) -> bool:
        t = time.monotonic()
        result = move(api, target, quat, strategy=strategy, tolerance=0.06)
        detail = f"{time.monotonic() - t:.1f} s" + (
            "" if result.ok else f" [{result.error.code}: {result.error.message[:80]}]"
        )
        rows.append((label, result.ok if expect else None, detail))
        return result.ok

    try:
        env.engage()
        standing(truth)
        sim("reset")
        time.sleep(2.0)
        adapter.go_home(arm="both")
        axis = truth.position("aspire_bottle")
        beside = axis + [0.06, -(BOTTLE_RADIUS_M + 0.02 + 0.08), 0.02]
        since = time.monotonic()
        leg("from home straight to beside the bottle", beside, expect=None)
        adapter.go_home(arm="both")
        lifted = [home_pos[0], home_pos[1], axis[2] + 0.12]
        leg("lift from home (interpolation, handshake)", lifted, HANDSHAKE_WXYZ, INTERP)
        leg("then cuRobo to beside the bottle", beside)
        leg("beside the bottle -> up and back", [0.25, -0.25, axis[2] + 0.12])
        leg("-> the home pose", home_pos, home_quat)
        hit = sorted(truth.touches("aspire_case", since))
        rows.append(("no case contact", not hit, str(hit or "none")))
        return report(rows)
    finally:
        truth.close()
        adapter.close()


def exp_watch(args: argparse.Namespace) -> int:
    """Record every truth message until interrupted (or --seconds), then summarize."""
    import json

    out = Path(args.out)
    if not args.summary:
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.subscribe(b"")
        sock.connect(f"tcp://127.0.0.1:{args.camera_port}")
        deadline = time.monotonic() + args.seconds
        with out.open("w") as stream:
            try:
                while time.monotonic() < deadline:
                    try:
                        message = msgpack.unpackb(
                            sock.recv(), object_hook=msgpack_numpy.decode, raw=False
                        )
                    except zmq.Again:
                        continue
                    stream.write(
                        json.dumps({"t": time.time(), **message.get("sim_truth", {})}) + "\n"
                    )
                    stream.flush()
            except KeyboardInterrupt:
                pass
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    bottle = np.array([r["aspire_bottle"] for r in rows if "aspire_bottle" in r])
    upright = [r.get("aspire_bottle_upright", 1.0) for r in rows]
    contacts: dict[str, dict[str, float]] = {}
    for r in rows:
        for obj, links in r.get("contacts", {}).items():
            for link, hit in links.items():
                contacts.setdefault(obj, {})[link] = max(
                    contacts.get(obj, {}).get(link, 0.0), hit[0]
                )
    start_z = float(np.median(bottle[:15, 2]))
    print(f"{len(rows)} messages over {rows[-1]['t'] - rows[0]['t']:.0f} s")
    print(
        f"bottle: highest {(bottle[:, 2].max() - start_z) * 100:+.1f} cm above its start, "
        f"ends {(bottle[-1] - bottle[0]) * 100} cm from its start (torso frame), "
        f"min upright {min(upright):.2f}, final upright {upright[-1]:.2f}"
    )
    for obj, links in contacts.items():
        print(
            f"contacts with {obj}: " + ", ".join(f"{k} {v:.0f} N" for k, v in sorted(links.items()))
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "experiment", choices=["engage", "heading", "retreat", "grip", "outage", "curobo", "watch"]
    )
    parser.add_argument(
        "--out", default="/tmp/aspire-g1-sim-watch.jsonl", help="watch: the JSONL file"
    )
    parser.add_argument("--seconds", type=float, default=3600.0, help="watch: stop after this long")
    parser.add_argument("--summary", action="store_true", help="watch: only summarize --out")
    parser.add_argument("--camera-port", type=int, default=5565)
    parser.add_argument("--curobo-url", default="http://192.168.75.174:8118")
    parser.add_argument(
        "--legacy-heading", action="store_true", help="heading: use the pre-fix facing"
    )
    parser.add_argument(
        "--squeeze", type=float, help="grip: closure past contact (default: the adapter's)"
    )
    args = parser.parse_args()
    return globals()[f"exp_{args.experiment}"](args)


if __name__ == "__main__":
    sys.exit(main())
