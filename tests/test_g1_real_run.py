import json
from pathlib import Path

import pytest

from cap_harness.g1_real import codec
from cap_harness.g1_real import station as g1_station
from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.run import run_program
from cap_harness.yam_real.cameras import SyntheticCamera


def test_run_program_on_fake_g1_station(tmp_path):
    outcome = run_program(
        benchmark="g1_real",
        suite="joint_motion",
        task_id=0,
        seed=1,
        program_path=Path("examples/g1_real/joint_motion.py"),
        output_root=tmp_path,
        station="g1-lab",
        allow_motion=True,
        station_backend="fake",
    )
    assert outcome.program_ok and outcome.termination_reason == "program_completed"
    assert any(tmp_path.rglob("right-joint_pos.npy"))


def _unengaged_env(body_type=FakeBodyClient):
    cfg = load_g1_station_config("g1-lab")
    home = codec.upper_body_from_arms(
        cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    )
    return G1Env(cfg, body_type(tau_s=0.02, initial_q17=home), camera=SyntheticCamera())


def _run(tmp_path, allow_motion):
    return run_program(
        benchmark="g1_real",
        suite="joint_motion",
        task_id=0,
        seed=1,
        program_path=Path("examples/g1_real/joint_motion.py"),
        output_root=tmp_path,
        station="g1-lab",
        allow_motion=allow_motion,
        station_backend="real",
    )


def test_run_program_engages_the_real_station_with_allow_motion(monkeypatch, tmp_path):
    env = _unengaged_env()
    monkeypatch.setattr(g1_station, "build_real_station", lambda *a, **k: env)
    outcome = _run(tmp_path, allow_motion=True)
    assert env.body.start_count == 1
    assert outcome.program_ok and outcome.termination_reason == "program_completed"


def test_run_program_does_not_engage_without_allow_motion(monkeypatch, tmp_path):
    env = _unengaged_env()
    monkeypatch.setattr(g1_station, "build_real_station", lambda *a, **k: env)
    _run(tmp_path, allow_motion=False)
    assert env.body.start_count == 0 and not env.keep_alive_running


def test_run_program_records_a_refused_engage_as_harness_error(monkeypatch, tmp_path):
    class StaleBody(FakeBodyClient):
        def feedback_age_s(self):
            return 99.0

    env = _unengaged_env(StaleBody)
    monkeypatch.setattr(g1_station, "build_real_station", lambda *a, **k: env)
    with pytest.raises(RuntimeError, match="stale"):
        _run(tmp_path, allow_motion=True)
    assert env.body.start_count == 0
    (outcome,) = tmp_path.rglob("outcome.json")
    record = json.loads(outcome.read_text())
    assert record["termination_reason"] == "harness_error"
    assert "stale" in json.dumps(record)
