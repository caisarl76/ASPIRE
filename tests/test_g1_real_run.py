from pathlib import Path

from cap_harness.run import run_program


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
