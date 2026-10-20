"""Check launch permissions and startup instructions without starting an agent."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _launch(tmp_path, script, arguments, *, check=True):
    """Run a launcher copy with stubbed python and codex that print their argv."""
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    launcher = scripts / script
    shutil.copyfile(root / "scripts" / script, launcher)
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    guide = (root / "docs/system2-agent-guide.md").read_text().rstrip("\n")
    (docs / "system2-agent-guide.md").write_text(guide)
    binaries = tmp_path / ".venv-system2/bin"
    binaries.mkdir(parents=True, exist_ok=True)
    # Stub the health check / MCP runtime and capture argv. Neither a real
    # service nor a real agent runs in these tests.
    printer = f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    python = binaries / "python"
    python.write_text("#!/bin/sh\nexit 0\n" if script == "start_system2_agent.sh" else printer)
    python.chmod(0o755)
    codex = binaries / "codex"
    codex.write_text(printer)
    codex.chmod(0o755)
    launched = subprocess.run(
        ["bash", str(launcher), *arguments],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=check,
    )
    return launched, guide


@pytest.mark.parametrize(
    "arguments",
    [[], ["--allow-motion"], ["--allow-motion", "exec", "Pick up the cube."]],
)
def test_launcher_selects_startup_mode_and_preserves_task(tmp_path, arguments):
    launched, guide = _launch(tmp_path, "start_system2_agent.sh", arguments)
    argv = json.loads(launched.stdout)
    prompt = argv[-1]
    assert argv[argv.index("-C") + 1] == str(tmp_path)
    assert 'approval_policy="never"' in argv
    assert 'sandbox_mode="workspace-write"' in argv
    assert "sandbox_workspace_write.network_access=true" in argv
    assert prompt.startswith(f"{guide}\n\nSession startup:")
    motion_approvals = {
        f'mcp_servers.cap.tools.{tool}.approval_mode="approve"'
        for tool in (
            "cap_program_go_home",
            "cap_move_to_pose",
            "cap_move_synchronized",
            "cap_move_to_joints",
            "cap_set_gripper",
            "cap_g1_real__set_hand_joints",
        )
    }
    for tool in ("cap_g1_real__get_controller_metadata", "cap_g1_real__get_hand_state"):
        assert f'mcp_servers.cap.tools.{tool}.approval_mode="approve"' in argv
    assert "Session startup:" in prompt
    if "--allow-motion" in arguments:
        assert motion_approvals <= set(argv)
        assert "call cap_program_go_home exactly once before the operator task" in prompt
        assert "Do not move the robot during readiness checks" not in prompt
    else:
        assert motion_approvals.isdisjoint(argv)
        assert "observation-only startup" in prompt
    if "exec" in arguments:
        assert argv[0] == "exec"
        assert prompt.endswith("Operator task:\nPick up the cube.")
    else:
        assert prompt.endswith(
            "Operator task:\nAfter startup completes, wait for the operator task."
        )


def _approved(argv, tool):
    return f'mcp_servers.cap.tools.{tool}.approval_mode="approve"' in argv


@pytest.mark.parametrize("order", ["robot_first", "motion_first"])
def test_agent_launcher_tells_the_dex3_robot_apart(tmp_path, order):
    args = ["--robot", "222", "--allow-motion"]
    if order == "motion_first":
        args = ["--allow-motion", "--robot", "222"]
    launched, _ = _launch(tmp_path, "start_system2_agent.sh", args)
    argv = json.loads(launched.stdout)
    prompt = argv[-1]
    assert "Dex3 hands" in prompt and "station g1-222" in prompt
    assert "call cap_program_go_home exactly once" in prompt
    for tool in ("cap_close_gripper", "cap_open_gripper", "cap_g1_real__set_hand_joints"):
        assert _approved(argv, tool)


def test_agent_launcher_withholds_hand_tools_on_the_rubber_hand_robot(tmp_path):
    launched, _ = _launch(tmp_path, "start_system2_agent.sh", ["--robot", "223", "--allow-motion"])
    argv = json.loads(launched.stdout)
    prompt = argv[-1]
    assert "motorless rubber hands" in prompt and "station g1-lab" in prompt
    assert _approved(argv, "cap_move_to_pose")
    for tool in ("cap_set_gripper", "cap_close_gripper", "cap_g1_real__set_hand_joints"):
        assert not _approved(argv, tool)


@pytest.mark.parametrize(("robot", "station"), [("222", "g1-222"), ("223", "g1-lab")])
def test_server_launcher_maps_robot_to_station(tmp_path, robot, station):
    launched, _ = _launch(
        tmp_path, "start_system2.sh", ["--robot", robot, "--allow-motion", "--engage"]
    )
    argv = json.loads(launched.stdout)
    assert argv[:2] == ["-m", "cap_harness.agent"]
    assert argv[2:] == [
        "--benchmark",
        "g1_real",
        "--station",
        station,
        "--allow-motion",
        "--engage",
    ]


@pytest.mark.parametrize("script", ["start_system2.sh", "start_system2_agent.sh"])
def test_unknown_robot_is_refused(tmp_path, script):
    launched, _ = _launch(tmp_path, script, ["--robot", "224"], check=False)
    assert launched.returncode == 2 and "222 (Dex3 hands) or 223" in launched.stderr
