"""System 2 over the fake G1 station: MCP hand tools, engage flag, camera-only Viser."""

import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cap_harness.agent.mcp import CapMcpBridge
from cap_harness.agent.server import build_runtime, main
from cap_harness.agent.visualizer import System2Visualizer
from cap_harness.g1_real.adapter import G1RealAdapter
from cap_harness.g1_real.station import build_fake_station
from cap_harness.registry import G1_REAL_PUBLIC_TOOL_NAMES


def test_g1_mcp_exposes_hand_tools_and_seven_joint_state():
    env = build_fake_station()
    adapter, runtime = build_runtime(
        env,
        adapter_type=G1RealAdapter,
        public_tool_names=G1_REAL_PUBLIC_TOOL_NAMES,
        family="g1_real",
        allow_motion=True,
    )
    try:
        bridge = CapMcpBridge(runtime.session)
        names = {t.name for t in bridge.tool_specs()}
        assert {
            "cap_g1_real__set_hand_joints",
            "cap_g1_real__get_hand_state",
            "cap_g1_real__get_controller_metadata",
            "cap_move_to_joints",
        } <= names
        assert "cap_g1_real__get_task_metadata" not in names
        state = bridge.call("cap_get_robot_state", {}).structuredContent["result"]
        assert len(state["joint_positions"]["left"]) == 7
        hand = bridge.call("cap_g1_real__set_hand_joints", {"arm": "left", "joints": [0.0] * 7})
        assert hand.structuredContent["ok"] is True
        measured = bridge.call("cap_g1_real__get_hand_state", {})
        assert measured.structuredContent["ok"] is True
        assert "job" not in measured.structuredContent  # a read, not a job
    finally:
        runtime.close()
        adapter.close()
        env.close()


@pytest.mark.parametrize(
    "argv",
    [
        ["--benchmark", "g1_real", "--station", "g1-lab", "--fake", "--engage"],
        ["--benchmark", "yam_real", "--engage", "--allow-motion"],
        ["--benchmark", "yam_real", "--fake"],
        ["--benchmark", "g1_real", "--sim"],
    ],
)
def test_server_main_rejects_invalid_flag_combinations(capsys, argv):
    assert main([*argv, "--no-viser", "--port", "0"]) == 2
    assert capsys.readouterr().err.strip()


def _free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return candidate.getsockname()[1]


def test_camera_only_viser_offers_engage_button():
    snapshot = {"cameras": {}, "camera_metadata": {}, "observation_seq": 1, "state": {}}
    session = SimpleNamespace(
        cached_visual_snapshot=Mock(return_value=snapshot),
        job=Mock(return_value=None),
        start_named_program=Mock(return_value="home-job"),
        steer=Mock(),
    )
    engage = Mock(return_value={"success": True, "hold_ticks": 3})
    visualizer = System2Visualizer(
        session, allow_motion=True, port=_free_port(), urdf_path=None, engage=engage
    )
    try:
        assert visualizer._robot is None
        with visualizer._lock:
            visualizer._refresh()
            assert "Robot model hidden" not in visualizer._status.content
        assert not visualizer._engage.disabled
        visualizer._on_engage(None)
        visualizer._engage_thread.join(3)
        engage.assert_called_once_with()
        with visualizer._lock:
            visualizer._refresh()
            assert "hold_ticks" in visualizer._status.content

        engage.side_effect = RuntimeError("feedback stale")
        visualizer._on_engage(None)
        visualizer._engage_thread.join(3)
        with visualizer._lock:
            visualizer._refresh()
            assert "feedback stale" in visualizer._status.content
    finally:
        visualizer.close()

    observer = System2Visualizer(
        session, allow_motion=False, port=_free_port(), urdf_path=None, engage=engage
    )
    try:
        assert observer._engage.disabled
    finally:
        observer.close()


def test_server_main_builds_fake_g1_and_engages(monkeypatch, capsys):
    import uvicorn

    from cap_harness.g1_real.env import G1Env

    monkeypatch.setattr(uvicorn, "run", lambda app, **_kwargs: None)
    argv = ["--benchmark", "g1_real", "--station", "g1-lab", "--fake", "--allow-motion"]
    assert main([*argv, "--engage", "--no-viser", "--port", "0"]) == 0
    assert "'already_engaged': True" in capsys.readouterr().out

    engage = G1Env.engage

    def stale(self):
        if self._engaged is not None:  # the fake station engaged itself at build time
            raise RuntimeError("feedback stale (1.000 s > 0.200 s); not engaging")
        return engage(self)

    monkeypatch.setattr(G1Env, "engage", stale)
    assert main([*argv, "--engage", "--no-viser", "--port", "0"]) == 1
    assert "feedback stale" in capsys.readouterr().err
