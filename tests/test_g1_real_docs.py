from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_g1_runbook_has_the_deployment_commands() -> None:
    text = (ROOT / "docs/g1-real.md").read_text()
    for needle in (
        "--input-type zmq_manager --output-type zmq --zmq-host 192.168.0.62 real",
        "--realsense-enable-depth",
        "--engage",
        "5557",
    ):
        assert needle in text


def test_skill_and_task_context_cover_the_g1_station() -> None:
    skill = (ROOT / ".agents/skills/cap-tool-usage/SKILL.md").read_text()
    assert "cap_g1_real__set_hand_joints" in skill
    assert "(g1-real.md)" in (ROOT / "docs/system2.md").read_text()
    assert (ROOT / "examples/g1_real/bottle_pick/TASK_CONTEXT.md").is_file()
