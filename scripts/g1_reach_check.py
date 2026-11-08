"""Reach map check on the real G1 (g1-222, Dex3): the sim grip at chosen bottle cells.

WARNING: moves the real robot. Builds the g1-222 station itself, so no CAP server may
hold the command port (5556). Start the PC2 camera server and deploy first (runbook
section 2), harness on, spotter ready, keyboard ``O`` at hand. The script engages once.

Same side grasp as ``scripts/g1_sim/experiments.py grip`` (hand level, thumb up, palm
facing +y), so results compare with the sim table in
examples/g1_real/bottle_pick/TASK_CONTEXT.md ("Where grasps work"). There is no sim
truth here: the bottle comes from SAM3 + depth, approach contact from the measured hand
reaching the contact site, close contact from the Dex3 finger
lag, and when the hand hides the bottle the operator answers y/n.

SONIC turns the torso (and with it the head camera) as the right arm reaches, so object
positions are carried through each posture change (pelvis IMU x waist FK) and the object
is localized again at the hover. ``--measure`` grasps nothing: it splits the grasp point
error into SAM3 spread, posture change, tracking, and the operator's correction.

  PYTHONPATH= VIRTUAL_ENV= .venv-system2/bin/python scripts/g1_reach_check.py [--object apple]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from cap_harness.g1_real import codec
import cap_harness.g1_real.adapter as adapter_module
from cap_harness.yam_real.action import YamActionBatch

sys.path.insert(0, str(Path(__file__).resolve().parent / "g1_sim"))
import experiments as ex

ex.STATION = "g1-222"
CELLS = ((0.34, -0.10), (0.40, -0.10), (0.34, -0.02), (0.46, -0.10))
# The apple grasp's IK is least contorted farther out and to the right (x 0.34 needs the
# elbow overextended and wrist pitch near its limit over the case).
APPLE_CELLS = ((0.46, -0.10), (0.40, -0.18))
F, N = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Grasp:
    name: str
    query: str  # SAM3 prompt
    quat: np.ndarray  # hand orientation, wxyz, torso frame
    site_offset: np.ndarray  # grasp site minus the object's SAM3 center, torso frame
    approach: np.ndarray  # unit direction the hand moves in for the last 8 cm
    overshoot_m: float  # how far past the site the approach may command
    # Set: center z = the cloud's top minus half this height. The bounding-box center of a
    # small round object seen from above is dragged down by depth-edge points (apple: 1 cm
    # below the cloud's bottom), and the grasp hangs ~9 cm of hand under its site.
    height_m: float | None = None


_a, _b = ex.DEX3_PALM
GRASPS = {
    # Side grasp, hand level, palm facing +y, 2 cm above the body center (sim grip).
    "bottle": Grasp(
        "bottle",
        "water bottle",
        ex.HANDSHAKE_WXYZ,
        np.array([0.0, 0.0, 0.02]) - ex.BOTTLE_RADIUS_M * N + _a * F - _b * N,
        N,
        0.02,
    ),
    # Top-side grasp, hand orientation from teleop episode pnp_table #52 at 11.78 s (right
    # hand, as the fingers start closing): rolled ~55 deg, palm down and to the left. The
    # site sat 5.5 cm above the apple's center (center from the video: 76 px at fx 606.6
    # for 8 cm); +1 cm here, since the open hand reaches 8.6 cm below the site. Robot run
    # (two closes on nothing): with the episode's site 1.6 cm left of center and its
    # down-and-right approach the hand hovered left of the apple, hiding it, and closed
    # beside it; now 1.5 cm right of center, straight down. Tune with --site-dy-cm.
    "apple": Grasp(
        "apple",
        "apple",
        np.array([0.863, -0.472, 0.136, 0.122]) / np.linalg.norm([0.863, -0.472, 0.136, 0.122]),
        np.array([0.0, -0.015, 0.065]),
        np.array([0.0, 0.0, -1.0]),
        0.01,
        0.07,
    ),
}
GRASP = GRASPS["bottle"]
SITE_SHIFT = np.zeros(3)  # --site-dx-cm / --site-dy-cm: calibration at the robot
PLACE_TOLERANCE_M = 0.02
WAYPOINT_XY = (0.25, -0.25)  # beside the case: every approach starts here
APPROACH_TOLERANCE_M = 0.008
# SAM3's rise of a held bottle is unreliable between these (sim: 2.7 cm for a 3+ cm lift,
# 2.4 cm for one that failed); in between the operator decides.
LIFT_UNSURE_M = (0.015, 0.035)
TRUTH: ex.SimTruth | None = None  # --sim: answers the operator questions, checks each verdict
SINCE = 0.0  # --sim: when the approach started (contacts after it count)
Z0 = 0.0  # --sim: true bottle height when the attempt started
# A re-localization this far (xy) from the carried estimate is taken as the hand confusing
# SAM3, not the object moving; the carried estimate stays.
REFRESH_MAX_JUMP_M = 0.04
MEASURE_SHOTS = 5
SEEN_FROM: np.ndarray | None = None  # torso pose (torso_pose) of locate's last image


def torso_pose(env) -> np.ndarray:
    """World-from-torso 4x4: the pelvis IMU orientation times the measured waist FK.

    SONIC turns the torso (waist and pelvis) when the right arm reaches, and the head camera
    and every torso-frame target turn with it, so a position measured in one posture is
    stale in the next. ponytail: the pelvis position is held fixed (no base odometry; feet
    planted), so only rotations are tracked; measure mode shows what that leaves.
    """
    raw = env.body.get_observations()
    with env._kin_lock:
        env.kin.seed(raw["left_joint_pos"], raw["right_joint_pos"], raw["waist"])
        pelvis_from_torso = env.kin._world_from_torso().as_matrix()
    w, x, y, z = np.asarray(raw["base_quat"], float).reshape(4)
    world_from_pelvis = np.eye(4)
    world_from_pelvis[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return world_from_pelvis @ pelvis_from_torso


def carry(p: np.ndarray, seen_from: np.ndarray, env) -> np.ndarray:
    """A world-fixed point seen at torso pose ``seen_from``, in the torso frame now."""
    world = seen_from @ np.append(p, 1.0)
    return (np.linalg.inv(torso_pose(env)) @ world)[:3]


def posture_change(before: np.ndarray, after: np.ndarray) -> str:
    yaw, pitch, roll = Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).as_euler(
        "ZYX", degrees=True
    )
    return f"torso turned yaw {yaw:+.1f} pitch {pitch:+.1f} roll {roll:+.1f} deg"


def cm(v: np.ndarray) -> str:
    return "[" + " ".join(f"{c * 100:+.1f}" for c in v) + "] cm"


def _thumb(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb)
    image.thumbnail((960, 540))
    return np.asarray(image)


class Viz:
    """Browser view of the camera, SAM3 and the attempt's decisions.

    Live camera, the last SAM3 mask, the geometry in the torso frame (bottle cloud and
    center green, contact site red, last target orange, measured hand blue, after each move), and the log.
    """

    def __init__(self, port: int, env) -> None:
        import viser

        self.env = env
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_frame("/torso", axes_length=0.1, axes_radius=0.003)
        blank = np.zeros((1, 1, 3), np.uint8)
        self.live = self.server.gui.add_image(blank, label="camera (live)", format="jpeg")
        self.sam = self.server.gui.add_image(blank, label="SAM3: none yet", format="jpeg")
        self.log = self.server.gui.add_markdown("")
        self.lines = [""]
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="reach-viz").start()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            try:  # a viewer hiccup must not touch the run
                frame = self.env.read_camera()
                if frame is not None:
                    self.live.image = _thumb(frame.rgb)
            except Exception:
                pass

    def point(self, name: str, position, color) -> None:
        self.server.scene.add_icosphere(
            name, radius=0.01, color=color, position=tuple(float(v) for v in position)
        )

    def localized(self, r, detail: str, accepted: bool) -> None:
        frame = self.env.read_camera()
        if frame is None or r.segmentation is None:
            return
        image = Image.fromarray(frame.rgb)
        mask = Image.fromarray(np.asarray(r.segmentation.mask, bool)).resize(image.size)
        tint = Image.new("RGB", image.size, (0, 255, 0) if accepted else (255, 0, 0))
        image = Image.composite(Image.blend(image, tint, 0.45), image, mask)
        if r.segmentation.box_xyxy is not None:
            ImageDraw.Draw(image).rectangle(list(map(float, r.segmentation.box_xyxy)), width=3)
        self.sam.image = _thumb(np.asarray(image))
        self.sam.label = f"SAM3 {'accepted' if accepted else 'rejected'}: {detail}"
        if r.point_cloud is not None:
            pts = np.asarray(r.point_cloud.points, np.float32)
            self.server.scene.add_point_cloud(
                "/bottle/cloud", points=pts, colors=(0, 200, 0), point_size=0.004
            )
        if r.geometry is not None:
            self.point("/bottle/center", r.geometry.pose.position, (0, 160, 0))

    def write(self, text: str) -> None:
        *done, last = (self.lines.pop() + text).split("\n")
        self.lines = (self.lines + done)[-40:] + [last]
        self.log.content = "```\n" + "\n".join(self.lines) + "\n```"

    def close(self) -> None:
        self._stop.set()
        self.server.stop()


class _Tee:
    """stdout to the terminal and the viewer's log (input() prompts included)."""

    def __init__(self, out, viz: Viz) -> None:
        self.out, self.viz = out, viz

    def write(self, text: str) -> int:
        self.viz.write(text)
        return self.out.write(text)

    def flush(self) -> None:
        self.out.flush()


VIZ: Viz | None = None
ENV = None  # the station, for the viewer's hand marker
_ex_move = ex.move


def _move(api, position, *args, **kwargs):
    if VIZ is not None:
        VIZ.point("/target", position, (255, 140, 0))
    result = _ex_move(api, position, *args, **kwargs)
    if VIZ is not None and ENV is not None:  # main thread only: env.kin is not thread-safe
        VIZ.point("/hand", ex.hand(ENV), (0, 120, 255))
    return result


ex.move = _move


def truth(kind: str) -> bool | None:
    """The sim's answer for a check, None on the robot."""
    if TRUTH is None:
        return None
    if kind == "contact":  # any right-arm link on the bottle since SINCE
        return bool(TRUTH.touches("aspire_bottle", SINCE, prefix="right_"))
    if kind == "close":  # fingers only: the palm is part of the wrist link
        return bool(TRUTH.touches("aspire_bottle", SINCE, prefix="right_hand_"))
    if kind == "upright":
        return TRUTH.latest.get("aspire_bottle_upright", 1.0) > 0.9
    dz = TRUTH.position("aspire_bottle")[2] - Z0
    return dz > 0.03 if kind == "lifted" else abs(dz) < 0.02


def verdict(kind: str, ok: bool, detail: str) -> tuple[bool, str]:
    """Append the sim truth to a verdict the script judged itself."""
    t = truth(kind)
    if t is None:
        return ok, detail
    return ok, f"{detail} [truth {'yes' if t else 'no'}{'' if t == ok else ', MISMATCH'}]"


def locate(api) -> tuple[np.ndarray | None, str]:
    """Bottle center in the torso frame, or None with the reason (recipe size rule).

    Sets SEEN_FROM to the torso pose the image was taken from.
    """
    global SEEN_FROM
    SEEN_FROM = torso_pose(ENV)
    r = api.localize_object(GRASP.query, target_frame="torso")
    if not r.ok:
        detail = f"localization failed: {r.error.message if r.error else '?'}"
        if VIZ is not None:
            VIZ.localized(r, detail, False)
        return None, detail
    pts = r.point_cloud.points
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    across, tall = float(max(hi[:2] - lo[:2])), float(hi[2] - lo[2])
    center = np.array(r.geometry.pose.position, float)  # a copy: the pose array is read-only
    if GRASP.height_m is not None:
        center[2] = hi[2] - GRASP.height_m / 2
    # A standing 500 ml bottle's body center is ~10.5 cm above its base (the cloud's
    # bottom); the difference is SAM3's z offset (sim: +1.7 cm).
    detail = (
        f"score {r.segmentation.score:.2f}, {across * 100:.0f} cm across, {tall * 100:.0f} cm tall, "
        f"center {(center[2] - lo[2]) * 100:.1f} cm above the cloud's bottom"
    )
    accepted = not (across > 0.12 or tall > 0.30)
    if VIZ is not None:
        VIZ.localized(r, detail, accepted)
    if not accepted:
        return None, f"rejected box ({detail})"
    return center, detail


def ask(question: str, kind: str) -> bool:
    if TRUTH is not None:
        answer = bool(truth(kind))
        print(f"  {question} [sim truth: {'y' if answer else 'n'}]")
        return answer
    return input(f"  {question} [y/n] ").strip().lower().startswith("y")


def place(api, cell: tuple[float, float]) -> np.ndarray | None:
    """Guide the operator until the bottle sits on the cell; None to skip the cell."""
    if TRUTH is not None:
        ex.sim("reset", str(cell[0]), str(cell[1]))
        time.sleep(2.0)
        for _ in range(3):
            center, detail = locate(api)
            if center is not None:
                err = (center - TRUTH.position("aspire_bottle")) * 100
                print(f"  SAM3 center minus truth {err.round(1)} cm ({detail})")
                return center
            print(f"  {detail}")
        return None
    while True:
        center, detail = locate(api)
        if center is None:
            print(f"  {detail}")
        else:
            dx, dy = cell[0] - center[0], cell[1] - center[1]
            ok = max(abs(dx), abs(dy)) <= PLACE_TOLERANCE_M
            print(
                f"  {GRASP.name} at x {center[0]:.3f} y {center[1]:.3f} ({detail}); "
                f"move it {dx * 100:+.0f} cm in x (away from robot +), {dy * 100:+.0f} cm in y "
                f"(robot's left +){'  -- on the cell' if ok else ''}"
            )
        key = input("  Enter = re-check, g = go, s = skip cell: ").strip().lower()
        if key == "s":
            return None
        if key == "g" and center is not None:
            return center


def refresh(env, api, center, seen_from) -> tuple[np.ndarray, np.ndarray, str]:
    """The object in the current posture: SAM3 again, else the old estimate carried over.

    Returns ``(center, seen_from, note)``.
    """
    carried, now = carry(center, seen_from, env), torso_pose(env)
    turn = (
        f"{posture_change(seen_from, now)}, object moved {cm(carried - center)} in the torso frame"
    )
    seen, detail = locate(api)
    if seen is not None and np.linalg.norm((seen - carried)[:2]) <= REFRESH_MAX_JUMP_M:
        return seen, SEEN_FROM, f"{turn}; SAM3 again, {cm(seen - carried)} from carried"
    why = detail if seen is None else f"SAM3 {cm(seen - carried)} from carried, ignored"
    return carried, now, f"{turn}; carried ({why})"


def attempt(
    env, api, center: np.ndarray, seen_from: np.ndarray
) -> list[tuple[str, bool | None, str]]:
    g, name = GRASP, GRASP.name
    rows: list[tuple[str, bool | None, str]] = []
    api.go_home(arm="both")
    api.open_gripper(arm="right")

    def site() -> np.ndarray:  # the object stays put in the world while the torso turns
        return carry(center, seen_from, env) + g.site_offset + SITE_SHIFT

    if VIZ is not None:
        VIZ.point("/site", site(), (255, 0, 0))
    # Turn the wrist here, beside the case: from the level hand mink cannot reach the apple's
    # pre-contact (15 cm miss at x 0.46, fake plant).
    ex.move(api, [*WAYPOINT_XY, site()[2] + 0.10], g.quat)
    ex.move(api, site() - 0.08 * g.approach, g.quat)
    # In the reach posture now: look again, then go to the refreshed pre-contact.
    center, seen_from, note = refresh(env, api, center, seen_from)
    print(f"  grasp point refresh: {note}")
    if VIZ is not None:
        VIZ.point("/site", site(), (255, 0, 0))
    pre = ex.move(api, site() - 0.08 * g.approach, g.quat)
    rows.append(
        (
            "pre-contact pose",
            pre.ok,
            f"error {pre.final_errors.get('right', float('nan')) * 1000:.0f} mm"
            + (f" [{pre.error.code}: {pre.error.message}]" if pre.error else ""),
        )
    )
    if pre.steps_executed == 0:
        return rows  # refused: nothing moved toward the object
    # 1.5 cm steps to the grasp site and a little past it (the robot has no contact
    # sensing, so stop soon): stop once the measured hand reaches the site. Not on a SAM3
    # shift: its center jumps 2-3 cm as the hand enters the view (sim).
    global SINCE, Z0
    SINCE = time.monotonic()
    Z0 = TRUTH.position("aspire_bottle")[2] if TRUTH is not None else 0.0
    reached, gap, rounds = False, float("nan"), 0
    for step in np.arange(0.015, 0.08 + g.overshoot_m + 1e-9, 0.015):
        # Tight tolerance so the correction rounds remove SONIC's tracking offset;
        # ex.move's 4 cm setup tolerance accepts it and leaves the hand 1-2 cm short.
        r = ex.move(
            api, site() - (0.08 - step) * g.approach, g.quat, tolerance=APPROACH_TOLERANCE_M
        )
        rounds += int(r.diagnostics.get("correction_rounds", 0))
        time.sleep(0.3)
        gap = float((ex.hand(env) - site()) @ g.approach)
        reached = gap >= -0.005
        if reached:
            break
    if not reached:
        reached = ask(f"No contact sensed. Is the hand on the {name}?", "contact")
    rows.append(
        (
            f"hand reaches the {name}",
            *verdict(
                "contact",
                reached,
                f"after {step * 100:.1f} cm of approach, hand {gap * 100:+.1f} cm from the site "
                f"along the approach, {rounds} correction rounds",
            ),
        )
    )
    if not reached:  # a close and lift without contact only risks the object
        api.open_gripper(arm="right")
        ex.move(api, ex.hand(env) - 0.08 * g.approach, g.quat)
        ex.move(api, ex.hand(env) + [0.0, 0.0, 0.10], g.quat)
        return rows
    seen = locate(api)[0] is not None
    upright = seen or ask(f"{name} hidden by the hand; is it still in place?", "upright")
    rows.append((f"{name} in place before close", *verdict("upright", upright, "")))
    SINCE = time.monotonic()  # truth for the close: contacts while closing
    close = api.close_gripper(arm="right")
    d = close.diagnostics
    rows.append(
        (
            "close stops at contact",
            *verdict(
                "close",
                bool(d.get("contact")),
                f"{d.get('completion')} closure {d.get('closure')} "
                f"lag {d.get('max_finger_lag_rad')} rad",
            ),
        )
    )
    quat = ex.current_quat(env)
    ex.move(api, ex.hand(env) + [0.0, 0.0, 0.05], quat)
    time.sleep(0.5)
    now, detail = locate(api)
    rise = None if now is None else float(now[2] - carry(center, seen_from, env)[2])
    if rise is None or LIFT_UNSURE_M[0] <= rise <= LIFT_UNSURE_M[1]:
        seen = detail if rise is None else f"SAM3 rise {rise * 100:.1f} cm"
        lifted_ok, note = (
            ask(f"({seen}) Did the {name} rise 3 cm or more with the hand?", "lifted"),
            f"operator; {seen}",
        )
    else:
        lifted_ok, note = verdict(
            "lifted", rise > LIFT_UNSURE_M[1], f"{name} rose {rise * 100:.1f} cm"
        )
    rows.append((f"{name} lifted with the hand", lifted_ok, note))
    ex.move(api, ex.hand(env) - [0.0, 0.0, 0.05], quat)
    api.open_gripper(arm="right")
    ex.move(api, ex.hand(env) - 0.08 * g.approach, quat)
    ex.move(api, ex.hand(env) + [0.0, 0.0, 0.10], quat)
    now, detail = locate(api)
    if now is None:
        stands, note = ask(f"({detail}) Is the {name} back on the case?", "stands"), "operator"
    else:
        dz = float(now[2] - carry(center, seen_from, env)[2])
        stands, note = verdict("stands", abs(dz) < 0.02, f"z change {dz * 100:+.1f} cm")
    rows.append((f"{name} back in place after release", stands, note))
    return rows


def correction() -> list[float] | None:
    """The operator's hand correction in cm (+forward, +left), None when skipped."""
    if TRUTH is not None:
        return None  # the sim has no hand-vs-object truth in the torso frame
    while True:
        text = input(
            "  Look down on the hand: how far must it move for a good grasp? "
            "'forward left' cm (+forward, +left; Enter = 0 0, s = skip): "
        ).strip()
        if text.lower() == "s":
            return None
        try:
            dx, dy = (float(v) for v in text.split()) if text else (0.0, 0.0)
            return [dx, dy]
        except ValueError:
            print("  two numbers, e.g. '0 -1.5'")


def measure(env, api, center, seen_from) -> tuple[list[tuple[str, bool | None, str]], dict]:
    """Where the grasp point error comes from, one source at a time; nothing is grasped.

    1. SAM3 repeatability: MEASURE_SHOTS localizations at Home.
    2. Posture: the torso's turn from Home to the hover, and how far that moves the object
       in the torso frame (the carry).
    3. SAM3 in the hover posture against the carried estimate (validates the carry; a
       hidden object is reported).
    4. Tracking: the measured hand (FK) against the hover target.
    5. Operator: how far the hand must move for a good grasp (all errors together).
    """
    g = GRASP
    rows: list[tuple[str, bool | None, str]] = []
    data: dict = {}
    shots = []
    for _ in range(MEASURE_SHOTS):
        c, detail = locate(api)
        if c is not None:
            shots.append(c)
    if not shots:
        return [("SAM3 sees the object at Home", False, detail)], data
    spread = np.ptp(shots, axis=0)
    center, seen_from = np.mean(shots, axis=0), SEEN_FROM  # Home posture, still
    data["sam3_spread_cm"] = (spread * 100).round(1).tolist()
    rows.append(
        (
            "SAM3 repeatable at Home",
            bool(spread[:2].max() <= 0.01),
            f"{len(shots)}/{MEASURE_SHOTS} seen, spread {cm(spread)}",
        )
    )

    def site() -> np.ndarray:
        return carry(center, seen_from, env) + g.site_offset + SITE_SHIFT

    api.open_gripper(arm="right")
    ex.move(api, [*WAYPOINT_XY, site()[2] + 0.10], g.quat)
    ex.move(api, site() - 0.08 * g.approach, g.quat)
    # Tight, so the hand is where the target says and the operator's correction is the rest.
    hover = site() - 0.08 * g.approach
    ex.move(api, hover, g.quat, tolerance=APPROACH_TOLERANCE_M)
    time.sleep(1.0)
    now = torso_pose(env)
    carried = carry(center, seen_from, env)
    data["object_shift_cm"] = ((carried - center) * 100).round(1).tolist()
    data["posture"] = posture_change(seen_from, now)
    print(f"  posture Home -> hover: {data['posture']}, object moved {cm(carried - center)}")
    seen, detail = locate(api)
    if seen is None:
        print(f"  SAM3 in the hover posture: {detail}")
    else:
        data["sam3_vs_carried_cm"] = ((seen - carried) * 100).round(1).tolist()
        print(f"  SAM3 in the hover posture: {cm(seen - carried)} from carried ({detail})")
    err = ex.hand(env) - hover
    data["tracking_cm"] = (err * 100).round(1).tolist()
    rows.append(("hand at the hover target", bool(np.linalg.norm(err) <= 0.01), cm(err)))
    fix = correction()
    data["correction_cm"] = fix
    print(f"  operator correction: {'skipped' if fix is None else fix} cm")
    ex.move(api, ex.hand(env) + [0.0, 0.0, 0.10], g.quat)
    return rows, data


def glide(env, left: np.ndarray, right: np.ndarray) -> str:
    """Both arms to these joints in a straight joint-space line, hands held as commanded."""
    q17, lh, rh = env.last_command()
    _, now_l, now_r = codec.arms_from_upper_body(q17)
    span = float(max(np.abs(left - now_l).max(), np.abs(right - now_r).max()))
    # The player eases with a cosine (peak speed pi/2 x the mean); keep the peak at the
    # planner's joint speed limit, as every planned move does.
    duration = max(1.0, np.pi / 2 * span / env.config.controller.plan_max_joint_velocity_rad_s)
    hands = env.config.hands
    batch = YamActionBatch.joint_abs(
        [duration],
        [left],
        [right],
        [[codec.hand_scalar(lh, hands["left"].open, hands["left"].closed)]],
        [[codec.hand_scalar(rh, hands["right"].open, hands["right"].closed)]],
        source="reach_check_glide",
        meta={"hand_joints": {"left": lh[None], "right": rh[None]}},
    )
    result = env.execute_action_batch(batch, start_interp_s=duration, settle_s=0.2)
    return f"{span:.2f} rad over {duration:.1f} s, settled {result.get('settled')}"


def soft_release(env, api, idle) -> None:
    """Leave the arms where SONIC will hold them: Home, then a glide to SONIC's idle pose.

    When the stream stops SONIC drops the arms to its own idle pose after 1 s of silence,
    a fast swing from wherever they are (seen after a crash mid-approach). The glide ends
    on the idle pose measured right after engage, so that drop has nothing left to move.
    A second Ctrl-C skips the rest.
    """
    try:
        print("soft release: Home, then SONIC's idle arms (Ctrl-C again to skip)")
        home = api.go_home(arm="both")
        print("  Home:", home.ok, home.error or "")
        _, left, right = codec.arms_from_upper_body(idle[0])
        print("  idle:", glide(env, left, right))
    except BaseException as exc:  # incl. a second Ctrl-C: stop moving, let the stream end
        print(f"  soft release stopped ({type(exc).__name__}: {exc})")


def hold() -> bool:
    """After a crash or Ctrl-C the keep-alive still streams the last command: the robot stays.

    True: the operator asked for Home and the soft release. False: stop the stream here.
    """
    try:
        input(
            "\nThe robot holds its last commanded pose. Enter = Home, then release; "
            "Ctrl-C = stop the stream here (SONIC drops the arms after 1 s): "
        )
        return True
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--object", choices=sorted(GRASPS), default="bottle")
    parser.add_argument("--cells", nargs="+", help="x,y cells (default: per object)")
    parser.add_argument(
        "--site-dx-cm", type=float, default=0.0, help="shift the grasp site (+x away)"
    )
    parser.add_argument(
        "--site-dy-cm", type=float, default=0.0, help="shift the grasp site (+y left)"
    )
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="no grasp: measure SAM3 spread, the posture change, tracking, and the operator's "
        "hand correction at each cell; prints the suggested --site-dx-cm/--site-dy-cm",
    )
    parser.add_argument("--sam3-url", default="http://192.168.75.174:8114")
    parser.add_argument("--out", type=Path, default=Path("/tmp/jihun/aspire-g1-reach.jsonl"))
    parser.add_argument(
        "--squeeze", type=float, help="closure past first contact (adapter default)"
    )
    parser.add_argument(
        "--sim", action="store_true", help="rehearse on the g1-sim station (scripts/g1_sim.sh up)"
    )
    parser.add_argument(
        "--viz-port", type=int, default=8082, help="browser viewer port; 0 turns it off"
    )
    args = parser.parse_args()
    global TRUTH, VIZ, GRASP, ENV
    GRASP = GRASPS[args.object]
    SITE_SHIFT[:2] = args.site_dx_cm / 100, args.site_dy_cm / 100
    if args.sim:
        ex.STATION = "g1-sim"
        load = ex.load_g1_station_config

        def load_as_robot(station: str):  # g1-222 has hold_heading off; rehearse it that way
            cfg = load(station)
            return replace(cfg, sonic=replace(cfg.sonic, hold_heading=False))

        ex.load_g1_station_config = load_as_robot
        TRUTH = ex.SimTruth(5575)
    if args.squeeze is not None:
        adapter_module.GRIP_SQUEEZE = args.squeeze
    print("grip squeeze past contact:", adapter_module.GRIP_SQUEEZE)
    cells = (
        [tuple(float(v) for v in c.split(",")) for c in args.cells]
        if args.cells
        else list(APPLE_CELLS if args.object == "apple" else CELLS)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env, adapter, api = ex.build(camera=True, sam3_url=args.sam3_url)
    ENV = env
    if args.viz_port:
        VIZ = Viz(args.viz_port, env)
        sys.stdout = _Tee(sys.stdout, VIZ)
        print(f"viewer: http://localhost:{args.viz_port}")
    idle = None  # SONIC's own arm pose right after engage; the soft release ends there
    fixes: list[list[float]] = []  # --measure: operator corrections, cm
    try:
        print("engage:", env.engage())
        idle = env.measured_command()
        if TRUTH is not None:
            ex.standing(TRUTH)  # the sim robot starts held by the band
        home = api.go_home(arm="both")
        print("startup Home:", home.ok, home.final_errors, home.error)
        for cell in cells:
            for n in range(1, args.attempts + 1):
                print(f"\n== cell x {cell[0]:.2f} y {cell[1]:+.2f}, attempt {n}/{args.attempts}")
                center = place(api, cell)
                if center is None:
                    break
                if args.measure:
                    rows, data = measure(env, api, center, SEEN_FROM)
                    if data.get("correction_cm"):
                        fixes.append(data["correction_cm"])
                else:
                    rows, data = attempt(env, api, center, SEEN_FROM), {}
                code = ex.report(rows)
                with args.out.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "mode": "measure" if args.measure else "grasp",
                                "object": args.object,
                                "cell": cell,
                                "attempt": n,
                                "bottle": center.round(3).tolist(),
                                "success": code == 0 and len(rows) > 1,
                                "squeeze": adapter_module.GRIP_SQUEEZE,
                                "site_shift": SITE_SHIFT.round(3).tolist(),
                                "rows": [
                                    [name, None if ok is None else bool(ok), detail]
                                    for name, ok, detail in rows
                                ],
                                **data,
                            }
                        )
                        + "\n"
                    )
                api.go_home(arm="both")
        if fixes:
            fix = SITE_SHIFT[:2] * 100 + np.mean(fixes, axis=0)
            print(
                f"\nmean correction over {len(fixes)} cells: {np.mean(fixes, axis=0).round(1)} cm"
                f" (spread {np.ptp(fixes, axis=0).round(1)} cm); next run: "
                f"--site-dx-cm {fix[0]:.1f} --site-dy-cm {fix[1]:.1f}"
            )
        return 0
    except BaseException as exc:  # incl. Ctrl-C: hold, then the operator decides
        if isinstance(exc, KeyboardInterrupt):
            print("\ninterrupted")
        else:
            traceback.print_exc()
        if idle is not None and not hold():
            idle = None  # no soft release: the stream stops where the robot is
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if idle is not None:
            soft_release(env, api, idle)
        adapter.close()
        if TRUTH is not None:
            TRUTH.close()
        if VIZ is not None:
            sys.stdout = sys.stdout.out
            VIZ.close()


if __name__ == "__main__":
    sys.exit(main())
