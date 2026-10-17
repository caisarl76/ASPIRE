import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from cap_harness.contracts import Pose
from cap_harness.g1_real.acceptance import (
    HOLD_MAX_ERROR_RAD,
    SEGMENT_MAX_POSITION_M,
    hold_error,
    segment_error,
)

ROOT = Path(__file__).resolve().parents[1]


def _hold(offset: float) -> dict[str, float]:
    targets = np.zeros((3, 14))
    measured = targets.copy()
    measured[1, 3] = offset
    measured[2, 10] = -offset
    return hold_error(targets, measured)


def test_hold_error_threshold():
    ok, bad = _hold(0.04), _hold(0.06)
    assert ok == {"left": 0.04, "right": 0.04}
    assert max(ok.values()) <= HOLD_MAX_ERROR_RAD < max(bad.values())


def test_segment_error_position_miss_fails():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    target = Pose(np.array([0.3, 0.2, 0.1]), q, "torso")
    measured = Pose(np.array([0.32, 0.2, 0.1]), q, "torso")
    pos, rot = segment_error(target, measured)
    assert pos > SEGMENT_MAX_POSITION_M
    assert rot == 0.0


def test_segment_error_quaternion_sign_invariant():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    pos, rot = segment_error(Pose(np.zeros(3), q, "torso"), Pose(np.zeros(3), -q, "torso"))
    assert pos == 0.0 and math.isclose(rot, 0.0, abs_tol=1e-9)


def test_fake_hold_writes_json(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/g1_tracking_acceptance.py", "--fake", "--phase", "hold",
         "--output-dir", str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
    (path,) = tmp_path.glob("*.json")
    assert json.loads(path.read_text())["verdict"] == "PASS"


def test_fake_segments_default_length(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/g1_tracking_acceptance.py", "--fake", "--phase", "segments",
         "--output-dir", str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (path,) = tmp_path.glob("*.json")
    assert len(json.loads(path.read_text())["segments"]) == 10
