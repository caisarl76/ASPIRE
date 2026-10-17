"""On-robot tracking acceptance for g1_real (spec section 9).

WARNING: this MOVES THE ROBOT. Run it only after the runbook preflight
(docs/g1-real.md), with the operator ready: keyboard ``O`` on PC2 is the e-stop.

Phases: ``hold`` (5 s of the measured pose, joint drift <= 0.05 rad), ``segments``
(ten 0.15 m torso-X moves on the primary arm, <= 1.5 cm and <= 5 deg), ``loaded``
(``hold`` with a 300 g object in the primary hand). ``--fake`` runs the plumbing
against the fake body. Exit 0 PASS, 1 FAIL, 2 setup error.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

from cap_harness.api import CapApi
from cap_harness.contracts import MotionStrategy, Pose
from cap_harness.errors import ErrorCode
from cap_harness.g1_real.acceptance import (
    HOLD_MAX_ERROR_RAD,
    SEGMENT_MAX_ORIENTATION_RAD,
    SEGMENT_MAX_POSITION_M,
    hold_error,
    segment_error,
)
from cap_harness.g1_real.adapter import G1RealAdapter
from cap_harness.g1_real.kinematics import G1Kinematics
from cap_harness.g1_real.station import build_fake_station, build_real_station
from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider

HOLD_S, HOLD_HZ = 5.0, 50.0
SEGMENTS, SEGMENT_S = 10, 3.0


def run_hold(env) -> dict:
    samples = []
    for _ in range(round(HOLD_S * HOLD_HZ)):
        raw = env.body.get_observations()
        samples.append(np.concatenate([raw["left_joint_pos"], raw["right_joint_pos"]]))
        time.sleep(1.0 / HOLD_HZ)
    measured = np.asarray(samples, dtype=float)
    err = hold_error(np.tile(measured[0], (len(measured), 1)), measured)
    ok = max(err.values()) <= HOLD_MAX_ERROR_RAD
    return {
        "ok": ok,
        "summary": f"max joint error left {err['left']:.4f} rad, right {err['right']:.4f} rad "
        f"(<= {HOLD_MAX_ERROR_RAD})",
        "data": {"max_error_rad": err, "samples": measured.tolist()},
    }


def run_segments(env, segment_m: float) -> dict:
    api = CapApi(
        G1RealAdapter(env, allow_physical_motion=True),
        ik_providers={"mink": YamKinematicsIKProvider(G1Kinematics(env.config.model_xml), dof=7)},
        default_camera=env.config.camera.role,
    )
    side = env.config.default_arm
    # ponytail: the interpolation planner has no speed field (time_dilation_factor is curobo's),
    # so segments use its default timing; the actual speed is the logged duration_s.
    strategy = MotionStrategy(ik_solver="mink", trajectory_planner="interpolation")
    rows = []
    for i in range(SEGMENTS):
        obs = env.get_observations(side)
        start = obs["ee_pos"]
        quat_wxyz = np.roll(obs["ee_quat"], 1)
        sign = -1.0 if i % 2 == 0 else 1.0  # retract first: forward is out of reach from home
        target = Pose(start + np.array([sign * segment_m, 0.0, 0.0]), quat_wxyz, "torso")
        t0 = time.monotonic()
        result = api.move_to_pose(target, arm=side, strategy=strategy)
        duration = time.monotonic() - t0
        if result.error is not None and result.error.code == ErrorCode.IK_UNREACHABLE:
            raise RuntimeError(f"segment {i}: {result.error}")
        end = env.get_observations(side)
        measured = Pose(end["ee_pos"], np.roll(end["ee_quat"], 1), "torso")
        pos, rot = segment_error(target, measured)
        ok = (
            bool(result.ok) and pos <= SEGMENT_MAX_POSITION_M and rot <= SEGMENT_MAX_ORIENTATION_RAD
        )
        rows.append(
            {
                "segment": i,
                "dx_m": sign * segment_m,
                "execution_ok": bool(result.ok),
                "position_error_m": pos,
                "orientation_error_rad": rot,
                "duration_s": duration,
                "ok": ok,
            }
        )
        print(
            f"  segment {i}: ok={result.ok} pos {pos * 1000:.1f} mm rot {np.degrees(rot):.2f} deg"
        )
    worst_pos = max(r["position_error_m"] for r in rows)
    worst_rot = max(r["orientation_error_rad"] for r in rows)
    return {
        "ok": all(r["ok"] for r in rows),
        "summary": f"worst position {worst_pos * 1000:.1f} mm (<= {SEGMENT_MAX_POSITION_M * 1000}),"
        f" worst orientation {np.degrees(worst_rot):.2f} deg "
        f"(<= {np.degrees(SEGMENT_MAX_ORIENTATION_RAD):.0f}); "
        f"target speed {segment_m / SEGMENT_S:.3f} m/s not enforced, see duration_s",
        "data": {"segments": rows},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--station", default="g1-lab")
    ap.add_argument("--station-config-root")
    ap.add_argument("--phase", required=True, choices=["hold", "segments", "loaded"])
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/g1-acceptance"))
    ap.add_argument(
        "--segment-m",
        type=float,
        default=0.15,
        help="segment length (0.3 is unreachable from home)",
    )
    ap.add_argument("--fake", action="store_true", help="use the fake body (no robot)")
    args = ap.parse_args()

    try:
        if args.fake:
            env = build_fake_station(args.station, config_root=args.station_config_root)
        else:
            env = build_real_station(args.station, config_root=args.station_config_root)
            env.engage()
    except Exception as exc:
        print(f"SETUP ERROR: {exc}")
        return 2
    try:
        if args.phase == "loaded":
            input("Place a 300 g object in the primary hand and press Enter")
        phase = run_segments(env, args.segment_m) if args.phase == "segments" else run_hold(env)
    except Exception as exc:
        print(f"SETUP ERROR: {exc}")
        return 2
    finally:
        env.close()
    verdict = "PASS" if phase["ok"] else "FAIL"
    print(f"{verdict} {args.phase}: {phase['summary']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    record = {
        "phase": args.phase,
        "fake": args.fake,
        "verdict": verdict,
        "thresholds": {
            "hold_max_error_rad": HOLD_MAX_ERROR_RAD,
            "segment_max_position_m": SEGMENT_MAX_POSITION_M,
            "segment_max_orientation_rad": SEGMENT_MAX_ORIENTATION_RAD,
        },
        "summary": phase["summary"],
        **phase["data"],
    }
    (args.output_dir / f"{stamp}.json").write_text(json.dumps(record, indent=2))
    return 0 if phase["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
