"""The sim's settle gate and contact log (scripts/g1_sim), without a simulator."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import time

import msgpack
import pytest
import zmq

SIM = Path(__file__).resolve().parents[1] / "scripts/g1_sim"
STANDING = {"pelvis_z": 0.786, "pelvis_tilt_deg": 0.5, "feet_on_floor": [True, True]}


def _settled(port: int, *args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(SIM / "settled.py"), "--port", str(port), *args],
        stdout=subprocess.PIPE,
        text=True,
    )


def _publish(sock: zmq.Socket, until_s: float) -> None:
    while time.monotonic() < until_s:
        sock.send(msgpack.packb({"sim_truth": STANDING}))
        time.sleep(0.05)


def test_settled_needs_telemetry_for_the_whole_hold() -> None:
    sock = zmq.Context.instance().socket(zmq.PUB)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    try:
        # Qualifying samples (1.5 s covers a slow interpreter start), then the stream stalls
        # well before the 3 s hold ends; the old gate passed this on wall time.
        stalled = _settled(port, "--timeout", "5", "--hold", "3")
        _publish(sock, time.monotonic() + 1.5)
        assert stalled.wait(10) == 1, stalled.stdout.read()
        # The same stance streamed throughout passes.
        steady = _settled(port, "--timeout", "4", "--hold", "1")
        _publish(sock, time.monotonic() + 3.0)
        assert steady.wait(10) == 0, steady.stdout.read()
    finally:
        sock.close(linger=0)


def test_dropped_snapshot_contacts_keep_each_links_peak() -> None:
    pytest.importorskip("mujoco")
    spec = importlib.util.spec_from_file_location("sim_rgbd", SIM / "sim_rgbd.py")
    sim_rgbd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim_rgbd)
    older = {"aspire_bottle": {"right_hand_index_1_link": (4.0, 0.0, 0.0, 0.01)}}
    newer = {
        "aspire_bottle": {"right_hand_index_1_link": (1.5, 0.0, 0.0, 0.02)},
        "aspire_case": {"right_wrist_yaw_link": (0.8, 0.0, 0.0, 0.0)},
    }
    merged = sim_rgbd.merge_touches(older, newer)
    assert merged == {
        "aspire_bottle": {"right_hand_index_1_link": (4.0, 0.0, 0.0, 0.01)},
        "aspire_case": {"right_wrist_yaw_link": (0.8, 0.0, 0.0, 0.0)},
    }
    assert older["aspire_bottle"]["right_hand_index_1_link"][0] == 4.0  # inputs untouched
