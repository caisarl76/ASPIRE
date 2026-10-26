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

  PYTHONPATH= VIRTUAL_ENV= .venv-system2/bin/python scripts/g1_reach_check.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

import cap_harness.g1_real.adapter as adapter_module

sys.path.insert(0, str(Path(__file__).resolve().parent / "g1_sim"))
import experiments as ex

ex.STATION = "g1-222"
QUERY = "water bottle"
CELLS = ((0.34, -0.10), (0.40, -0.10), (0.34, -0.02), (0.46, -0.10))
F, N = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
PLACE_TOLERANCE_M = 0.02
# SAM3's rise of a held bottle is unreliable between these (sim: 2.7 cm for a 3+ cm lift,
# 2.4 cm for one that failed); in between the operator decides.
LIFT_UNSURE_M = (0.015, 0.035)
TRUTH: ex.SimTruth | None = None  # --sim: answers the operator questions, checks each verdict
SINCE = 0.0  # --sim: when the approach started (contacts after it count)
Z0 = 0.0  # --sim: true bottle height when the attempt started


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
    """Bottle center in the torso frame, or None with the reason (recipe size rule)."""
    r = api.localize_object(QUERY, target_frame="torso")
    if not r.ok:
        return None, f"localization failed: {r.error.message if r.error else '?'}"
    pts = r.point_cloud.points
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    across, tall = float(max(hi[:2] - lo[:2])), float(hi[2] - lo[2])
    center = np.asarray(r.geometry.pose.position, float)
    # A standing 500 ml bottle's body center is ~10.5 cm above its base (the cloud's
    # bottom); the difference is SAM3's z offset (sim: +1.7 cm).
    detail = (
        f"score {r.segmentation.score:.2f}, {across * 100:.0f} cm across, {tall * 100:.0f} cm tall, "
        f"center {(center[2] - lo[2]) * 100:.1f} cm above the cloud's bottom"
    )
    if across > 0.12 or tall > 0.30:
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
                f"  bottle at x {center[0]:.3f} y {center[1]:.3f} ({detail}); "
                f"move it {dx * 100:+.0f} cm in x (away from robot +), {dy * 100:+.0f} cm in y "
                f"(robot's left +){'  -- on the cell' if ok else ''}"
            )
        key = input("  Enter = re-check, g = go, s = skip cell: ").strip().lower()
        if key == "s":
            return None
        if key == "g" and center is not None:
            return center


def attempt(env, api, center: np.ndarray) -> list[tuple[str, bool | None, str]]:
    rows: list[tuple[str, bool | None, str]] = []
    api.go_home(arm="both")
    api.open_gripper(arm="right")
    axis = center + [0.0, 0.0, 0.02]  # a little above the body center
    a, b = ex.DEX3_PALM
    site = axis - ex.BOTTLE_RADIUS_M * N + a * F - b * N
    ex.move(api, [0.25, -0.25, axis[2] + 0.12])
    pre = ex.move(api, site - 0.08 * N)
    rows.append(
        (
            "pre-contact pose",
            pre.ok,
            f"error {pre.final_errors.get('right', float('nan')) * 1000:.0f} mm"
            + (f" [{pre.error.code}: {pre.error.message}]" if pre.error else ""),
        )
    )
    if pre.steps_executed == 0:
        return rows  # refused: nothing moved toward the bottle
    # 1.5 cm steps to the contact site and 2 cm past it (sim went 4 cm past; the robot
    # has no contact sensing, so stop sooner): stop once the measured hand reaches the
    # site. Not on a SAM3 shift: its center jumps 2-3 cm as the hand enters the view (sim).
    global SINCE, Z0
    SINCE = time.monotonic()
    Z0 = TRUTH.position("aspire_bottle")[2] if TRUTH is not None else 0.0
    reached, gap = False, float("nan")
    for step in np.arange(0.015, 0.10 + 1e-9, 0.015):
        ex.move(api, site - (0.08 - step) * N)
        time.sleep(0.3)
        gap = float((ex.hand(env) - site) @ N)
        reached = gap >= -0.005
        if reached:
            break
    if not reached:
        reached = ask("No contact sensed. Is the palm or are the fingers on the bottle?", "contact")
    rows.append(
        (
            "hand reaches the bottle",
            *verdict(
                "contact",
                reached,
                f"after {step * 100:.1f} cm of approach, hand {gap * 100:+.1f} cm from the site "
                "along +y",
            ),
        )
    )
    seen = locate(api)[0] is not None
    upright = seen or ask("Bottle hidden by the hand; is it upright?", "upright")
    rows.append(("bottle upright before close", *verdict("upright", upright, "")))
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
    rise = None if now is None else float(now[2] - center[2])
    if rise is None or LIFT_UNSURE_M[0] <= rise <= LIFT_UNSURE_M[1]:
        seen = detail if rise is None else f"SAM3 rise {rise * 100:.1f} cm"
        lifted_ok, note = (
            ask(f"({seen}) Did the bottle rise 3 cm or more with the hand?", "lifted"),
            f"operator; {seen}",
        )
    else:
        lifted_ok, note = verdict(
            "lifted", rise > LIFT_UNSURE_M[1], f"bottle rose {rise * 100:.1f} cm"
        )
    rows.append(("bottle lifted with the hand", lifted_ok, note))
    ex.move(api, ex.hand(env) - [0.0, 0.0, 0.05], quat)
    api.open_gripper(arm="right")
    ex.move(api, ex.hand(env) - 0.08 * N)
    ex.move(api, ex.hand(env) + [0.0, 0.0, 0.10])
    now, detail = locate(api)
    if now is None:
        stands, note = ask(f"({detail}) Is the bottle standing on the case?", "stands"), "operator"
    else:
        stands, note = verdict(
            "stands",
            abs(now[2] - center[2]) < 0.02,
            f"z change {(now[2] - center[2]) * 100:+.1f} cm",
        )
    rows.append(("bottle stands after release", stands, note))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", nargs="+", default=[f"{x},{y}" for x, y in CELLS])
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--sam3-url", default="http://192.168.75.174:8114")
    parser.add_argument("--out", type=Path, default=Path("/tmp/jihun/aspire-g1-reach.jsonl"))
    parser.add_argument(
        "--squeeze", type=float, help="closure past first contact (adapter default)"
    )
    parser.add_argument(
        "--sim", action="store_true", help="rehearse on the g1-sim station (scripts/g1_sim.sh up)"
    )
    args = parser.parse_args()
    global TRUTH
    if args.sim:
        ex.STATION = "g1-sim"
        load = ex.load_g1_station_config

        def load_as_robot(station: str):  # g1-222 has hold_heading off; rehearse it that way
            cfg = load(station)
            return replace(cfg, sonic=replace(cfg.sonic, hold_heading=False))

        ex.load_g1_station_config = load_as_robot
        TRUTH = ex.SimTruth(5565)
    if args.squeeze is not None:
        adapter_module.GRIP_SQUEEZE = args.squeeze
    print("grip squeeze past contact:", adapter_module.GRIP_SQUEEZE)
    cells = [tuple(float(v) for v in c.split(",")) for c in args.cells]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env, adapter, api = ex.build(camera=True, sam3_url=args.sam3_url)
    try:
        print("engage:", env.engage())
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
                rows = attempt(env, api, center)
                code = ex.report(rows)
                with args.out.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "cell": cell,
                                "attempt": n,
                                "bottle": center.round(3).tolist(),
                                "success": code == 0 and len(rows) > 1,
                                "squeeze": adapter_module.GRIP_SQUEEZE,
                                "rows": [
                                    [name, None if ok is None else bool(ok), detail]
                                    for name, ok, detail in rows
                                ],
                            }
                        )
                        + "\n"
                    )
                api.go_home(arm="both")
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted: no further motion; the stream stops and SONIC idles the arms")
        return 130
    finally:
        adapter.close()
        if TRUTH is not None:
            TRUTH.close()


if __name__ == "__main__":
    sys.exit(main())
