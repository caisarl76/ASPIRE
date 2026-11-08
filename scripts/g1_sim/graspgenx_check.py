"""GraspGenX grasps on the sim bottle, executed and judged by sim truth.

Needs the sim with a CAP server that has a GraspGenX provider:
``GRASPGENX_URL=tcp://127.0.0.1:5586 scripts/g1_sim.sh smoke --keep``. Each trial resets the
layout, homes, asks ``cap_plan_grasps`` for candidates, prints how the best ones compare
with the bottle skill's site (level side grasp), then runs the first candidate the arm
reaches, the way the bottle skill does: 4 cm above a point 8 cm back along its approach (the
skill rises 12 cm, out of reach for these higher grasps),
re-localize the bottle there and shift the grasp by how far it moved in the torso frame (SONIC
turns the torso as the arm reaches), then down, in by 4/2/0 cm, close, lift 5 cm, put back,
open, back away.

    .venv-system2/bin/python scripts/g1_sim/graspgenx_check.py --trials 3
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import subprocess
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import numpy as np
from scipy.spatial.transform import Rotation

import experiments as ex
import smoke

ROOT = Path(__file__).resolve().parents[2]
STRATEGY = {"ik_solver": "mink", "trajectory_planner": "interpolation"}


def approach_axis(quat_wxyz) -> np.ndarray:
    """The palm normal (site +y), the direction the hand moves in to grasp."""
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).apply([0.0, 1.0, 0.0])


async def call(s, name, args=None) -> dict:
    return smoke._terminal(await smoke._call(s, name, args or {}))


async def move(s, position, quat) -> tuple[bool, float]:
    r = await call(s, "cap_move_to_pose", {
        "arm": "right", "max_steps": 300, "strategy": STRATEGY,
        "target_pose": {"position": [float(v) for v in position],
                        "quaternion_wxyz": [float(v) for v in quat], "frame": "torso"},
    })
    code = (r.get("error") or {}).get("code")
    err = (r.get("final_errors") or {}).get("right")
    if code not in (None, "controller_failed"):
        print(f"    move refused: {code}: {(r.get('error') or {}).get('message', '')[:160]}", flush=True)
    return code in (None, "controller_failed"), float(err) if err is not None else float("nan")


def skill_site(c) -> np.ndarray:
    d = np.hypot(c[0], c[1])
    a = np.asarray(c[:2]) + 0.03 * np.asarray(c[:2]) / d
    return np.array([a[0] + 0.06, a[1] - 0.053, c[2] - 0.03])


async def bottle_center(s) -> np.ndarray | None:
    """SAM3's bottle center, one retry; None when both fail (scores run 0.03-0.05 in the sim)."""
    for _ in range(2):
        loc = await call(s, "cap_localize_object", {
            "query": "water bottle", "camera_name": "ego_view", "target_frame": "torso"})
        loc = loc.get("result", loc)
        seg = loc.get("segmentation") or {}
        print(f"    localize: score {seg.get('score')}, box {np.round(seg.get('box_xyxy') or [], 0)}, "
              f"error {(loc.get('error') or {}).get('code')}", flush=True)
        if loc.get("geometry"):
            return np.asarray(loc["geometry"]["pose"]["position"])
    return None


async def trial(s, truth: ex.SimTruth, index: int, max_tries: int) -> dict:
    # Home first: a reset with the arm still out over the case places the case into it.
    await call(s, "cap_program_go_home")
    subprocess.run(["scripts/g1_sim.sh", "reset"], cwd=ROOT, check=True, capture_output=True)
    time.sleep(3.0)
    # Torso height and pitch/roll at plan time: the reset re-places only the case and the
    # bottle, so a crouch or lunge left by an earlier trial carries over.
    base = {"trial": index, "stance": [round(v, 3) for v in truth.latest["torso_world"][2:5]]}
    plan = await call(s, "cap_plan_grasps", {"query": "water bottle", "max_candidates": 12})
    grasps = plan.get("grasps") or (plan.get("result") or {}).get("grasps") or []
    print(f"  plan: {len(grasps)} candidates, {plan.get('diagnostics')}", flush=True)
    if not plan.get("ok") or not grasps:
        return {**base, "perception_failure": (plan.get("error") or {}).get("message")}
    c0 = await bottle_center(s)
    if c0 is None:
        return {**base, "perception_failure": "localize_object failed twice at Home"}
    site = skill_site(c0)
    for rank, g in enumerate(grasps[:5]):
        p, q = np.asarray(g["pose"]["position"]), g["pose"]["quaternion_wxyz"]
        a = approach_axis(q)
        print(f"  #{rank} score {g['score']:.2f} pos {np.round(p, 3)} approach {np.round(a, 2)} "
              f"(elev {np.degrees(np.arcsin(-a[2])):+.0f} deg) | skill site {np.round(site, 3)}, "
              f"off {np.linalg.norm(p - site) * 100:.1f} cm", flush=True)
    start = np.asarray(truth.latest["aspire_bottle_world"])
    feet = np.asarray(truth.latest["torso_world"][:2])  # a step shows here; the trial is then void
    for rank, g in enumerate(grasps[:max_tries]):
        p, q = np.asarray(g["pose"]["position"]), g["pose"]["quaternion_wxyz"]
        a = approach_axis(q)
        ok, err = await move(s, p - 0.08 * a + [0, 0, 0.04], q)
        c1 = await bottle_center(s) if ok else None
        shift = np.zeros(3) if c1 is None else c1 - c0
        # ponytail: a localization jump past 10 cm is taken as a SAM3 miss, not body motion.
        if np.linalg.norm(shift) > 0.10:
            print(f"  #{rank} re-localized {np.round(shift, 3)} m away; keeping the plan", flush=True)
            shift = np.zeros(3)
        p = p + shift
        if ok:
            ok, err = await move(s, p - 0.08 * a + [0, 0, 0.04], q)
        if ok:
            ok, err = await move(s, p - 0.08 * a, q)
        if not ok or not err < 0.03:
            print(f"  #{rank} pre-grasp not reached (error {err:.3f} m); next", flush=True)
            await call(s, "cap_program_go_home")
            continue
        for back in (0.04, 0.02, 0.0):
            await move(s, p - back * a, q)
        close = await call(s, "cap_close_gripper", {"arm": "right"})
        dg = close.get("diagnostics", {})
        h = close["terminal_state"]["end_effector_poses"]["right"]
        await move(s, np.asarray(h["position"]) + [0, 0, 0.05], h["quaternion_wxyz"])
        top = np.asarray(truth.latest["aspire_bottle_world"])
        await move(s, h["position"], h["quaternion_wxyz"])
        await call(s, "cap_open_gripper", {"arm": "right"})
        await move(s, np.asarray(h["position"]) - 0.08 * a, h["quaternion_wxyz"])
        time.sleep(1.0)
        end = np.asarray(truth.latest["aspire_bottle_world"])
        return {
            **base, "rank": rank, "score": round(g["score"], 3),
            "shift_cm": np.round(shift * 100, 1).tolist(),
            "approach": np.round(a, 2).tolist(), "contact": dg.get("contact"),
            "rise_cm": round((top[2] - start[2]) * 100, 1),
            "upright": round(float(truth.latest.get("aspire_bottle_upright", 1.0)), 3),
            "placement_cm": round(float(np.hypot(*(end[:2] - start[:2]))) * 100, 1),
            "torso_moved_cm": round(float(np.hypot(*(truth.latest["torso_world"][:2] - feet))) * 100, 1),
        }
    return {**base, "error": f"none of the first {max_tries} candidates reached"}


async def main_async(args) -> None:
    truth = ex.SimTruth(5575)
    time.sleep(1.0)
    async with streamablehttp_client("http://127.0.0.1:8223/mcp/") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for index in range(1, args.trials + 1):
                print(await trial(s, truth, index, args.max_tries), flush=True)
            await call(s, "cap_program_go_home")
    truth.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tries", type=int, default=4, help="candidates to try per trial")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
