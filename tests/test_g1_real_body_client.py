import time

from g1_real_wire import decode_packed
import msgpack
import numpy as np
import pytest
import zmq

from cap_harness.g1_real.body_client import FakeBodyClient, SonicZmqBodyClient
from cap_harness.g1_real.config import G1SonicConfig


def _cfg(port):
    return G1SonicConfig(
        command_bind=f"tcp://127.0.0.1:{port}",
        feedback_host="127.0.0.1",
        feedback_port=port + 1,
        feedback_topic="g1_debug",
        mode=0,
        speed=-1.0,
        height=-1.0,
        feedback_stale_s=0.5,
        engage_hold_s=0.2,
    )


def test_fake_client_lags_then_converges():
    fake = FakeBodyClient(tau_s=0.02)
    fake.command_upper_body(np.full(17, 0.5), np.zeros(17), None, None)
    first = fake.get_observations()["left_joint_pos"][0]
    time.sleep(0.2)
    assert first < 0.5 and abs(fake.get_observations()["left_joint_pos"][0] - 0.5) < 1e-3
    assert len(fake.commands) == 1 and fake.feedback_age_s() == 0.0
    fake.stale = True
    assert fake.feedback_age_s() == float("inf")


def test_sonic_client_publishes_a_planner_message_the_cpp_parser_accepts():
    port = 5700
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.subscribe(b"")
    sub.connect(f"tcp://127.0.0.1:{port}")
    client = SonicZmqBodyClient(_cfg(port))
    time.sleep(0.3)
    client.command_upper_body(np.arange(17.0), np.zeros(17), np.zeros(7), np.full(7, 2.0))
    raw = sub.recv()
    assert raw.startswith(b"planner")
    f = decode_packed(raw, b"planner")
    assert f["mode"].tolist() == [0] and np.allclose(f["facing"], [1, 0, 0])
    assert np.allclose(f["upper_body_position"], np.arange(17))
    assert np.allclose(f["right_hand_joints"], np.full(7, 0.25))  # step-clamped from 0
    client.close()
    sub.close()


def test_sonic_client_reads_feedback_and_heading():
    port = 5710
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(f"tcp://127.0.0.1:{port + 1}")
    client = SonicZmqBodyClient(_cfg(port))
    assert client.get_observations() is None and client.feedback_age_s() == float("inf")
    time.sleep(0.3)
    q = [np.cos(0.3), 0, 0, np.sin(0.3)]
    pub.send(
        b"g1_debug"
        + msgpack.packb(
            {
                "body_q": list(np.arange(29.0)),
                "body_dq": [0.0] * 29,
                "base_quat": q,
                "left_hand_q": [0.0] * 7,
                "right_hand_q": [0.0] * 7,
                "reference_heading_quat": q,
                "planner_reference_active": [1],
            }
        )
    )
    time.sleep(0.2)
    obs = client.get_observations()
    assert np.allclose(obs["right_joint_pos"], np.arange(22.0, 29.0))
    assert abs(obs["heading_yaw"]) < 1e-6  # base == reference: hold, do not turn
    assert client.feedback_age_s() < 0.5
    time.sleep(0.6)  # feedback stops: age must grow, not be refreshed by polling
    assert client.feedback_age_s() > 0.5
    client.close()
    pub.close()


def test_start_message_is_only_sent_on_request():
    port = 5720
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.subscribe(b"command")
    sub.connect(f"tcp://127.0.0.1:{port}")
    client = SonicZmqBodyClient(_cfg(port))
    time.sleep(0.3)
    client.send_start()
    f = decode_packed(sub.recv(), b"command")
    assert f["start"].tolist() == [1] and f["stop"].tolist() == [0] and f["planner"].tolist() == [1]
    client.close()
    sub.close()


def test_hand_step_limit_comes_from_the_profile_value():
    port = 5740
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.subscribe(b"")
    sub.connect(f"tcp://127.0.0.1:{port}")
    client = SonicZmqBodyClient(_cfg(port), hand_max_step_rad={"left": 0.25, "right": 0.1})
    time.sleep(0.3)
    client.command_upper_body(np.zeros(17), np.zeros(17), np.full(7, 2.0), np.full(7, 2.0))
    f = decode_packed(sub.recv(), b"planner")
    assert np.allclose(f["left_hand_joints"], np.full(7, 0.25))
    assert np.allclose(f["right_hand_joints"], np.full(7, 0.1))
    client.seed_hands(np.full(7, 1.0), np.full(7, 1.0))
    client.command_upper_body(np.zeros(17), np.zeros(17), np.full(7, 1.0), np.full(7, 1.05))
    f = decode_packed(sub.recv(), b"planner")
    assert np.allclose(f["left_hand_joints"], np.full(7, 1.0))  # seeded, so no ramp from 0.25
    assert np.allclose(f["right_hand_joints"], np.full(7, 1.05))
    client.close()
    sub.close()


def test_heading_never_sends_the_raw_imu_yaw_without_a_reference():
    # The old fallback returned the IMU yaw (0.6 here) and turned the real robot on engage.
    q = [np.cos(0.3), 0, 0, np.sin(0.3)]
    yaw = SonicZmqBodyClient._heading({"planner_reference_active": [1], "base_quat": q})
    assert yaw == 0.0


def test_poll_thread_survives_a_subscriber_error():
    from unittest.mock import Mock

    client = SonicZmqBodyClient(_cfg(5750))
    real_sub = client._sub
    client._sub = Mock(get_msg=Mock(side_effect=RuntimeError("boom")))
    time.sleep(0.05)
    assert client._thread.is_alive() and client.feedback_age_s() == float("inf")
    client.close()
    real_sub.close()


def _quat_yaw(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _feedback(**fields) -> dict:
    return {
        "body_q": [0.0] * 29,
        "body_dq": [0.0] * 29,
        "left_hand_q": [0.0] * 7,
        "right_hand_q": [0.0] * 7,
        **fields,
    }


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        # Real robot on engage: IMU heading 90 deg, planner frame initialised there -> hold.
        (_feedback(base_quat=_quat_yaw(np.pi / 2), init_base_quat=_quat_yaw(np.pi / 2)), 0.0),
        # Turned 0.3 rad since init (by hand or a prior session): hold where it is now.
        (_feedback(base_quat=_quat_yaw(0.9), init_base_quat=_quat_yaw(0.6)), 0.3),
        (
            _feedback(base_quat=_quat_yaw(0.9), init_base_quat=_quat_yaw(0.6), delta_heading=0.1),
            0.2,
        ),
        # Fork build: the applied reference wins over init_base_quat.
        (
            _feedback(
                base_quat=_quat_yaw(1.0),
                init_base_quat=_quat_yaw(0.0),
                reference_heading_quat=_quat_yaw(0.6),
                planner_reference_active=[1],
            ),
            0.4,
        ),
        # Planner still initialising: its frame is at yaw zero, send zero.
        (
            _feedback(
                base_quat=_quat_yaw(1.0),
                reference_heading_quat=_quat_yaw(0.6),
                planner_reference_active=[0],
            ),
            0.0,
        ),
        # No heading telemetry at all: never send the raw IMU yaw.
        (_feedback(base_quat=_quat_yaw(np.pi / 2)), 0.0),
        # Wraps: IMU -170 deg, init +170 deg is a 20 deg difference, not 340.
        (_feedback(base_quat=_quat_yaw(-2.96706), init_base_quat=_quat_yaw(2.96706)), 0.349066),
    ],
)
def test_facing_yaw_is_relative_to_the_planner_frame(msg, expected):
    assert SonicZmqBodyClient._heading(msg) == pytest.approx(expected, abs=1e-5)


def _at(yaw: float) -> dict:
    return _feedback(
        base_quat=_quat_yaw(yaw), reference_heading_quat=_quat_yaw(0.0), planner_reference_active=[1]
    )


def test_hold_heading_keeps_the_engage_heading_until_the_next_engage():
    from dataclasses import replace

    client = SonicZmqBodyClient(replace(_cfg(5760), hold_heading=True))
    try:
        assert client._facing_yaw(_at(0.10)) == pytest.approx(0.10)  # latched
        # an arm reach turned the torso: command the latched heading back, not the new one
        assert client._facing_yaw(_at(-0.08)) == pytest.approx(0.10)
        client.send_start()  # a new engage latches afresh
        assert client._facing_yaw(_at(-0.08)) == pytest.approx(-0.08)
    finally:
        client.close()


def test_hold_heading_relatches_instead_of_commanding_a_large_turn():
    from dataclasses import replace

    client = SonicZmqBodyClient(replace(_cfg(5770), hold_heading=True))
    try:
        client._facing_yaw(_at(0.0))
        assert client._facing_yaw(_at(0.7)) == pytest.approx(0.7)  # past 0.5 rad: follow
        # an unknown heading (planner re-initialising) drops the latch
        assert client._facing_yaw(_feedback(base_quat=_quat_yaw(0.2))) == 0.0
        assert client._facing_yaw(_at(0.3)) == pytest.approx(0.3)
    finally:
        client.close()


def test_without_hold_heading_the_current_heading_is_sent():
    client = SonicZmqBodyClient(_cfg(5780))
    try:
        client._facing_yaw(_at(0.10))
        assert client._facing_yaw(_at(-0.08)) == pytest.approx(-0.08)
    finally:
        client.close()


def test_hold_heading_survives_a_planner_reference_reset():
    # A start to a running deploy resets its planner reference to the current heading;
    # a heading held in the old frame then commands a turn (30-45 deg in a sim rehearsal).
    from dataclasses import replace

    def at(base: float, reference: float) -> dict:
        return _feedback(
            base_quat=_quat_yaw(base),
            reference_heading_quat=_quat_yaw(reference),
            planner_reference_active=[1],
        )

    client = SonicZmqBodyClient(replace(_cfg(5790), hold_heading=True))
    try:
        assert client._facing_yaw(at(0.40, 0.0)) == pytest.approx(0.40)  # latched
        # the reference jumps to the robot's heading, the robot has not moved:
        # hold the same physical heading, which is zero in the new frame
        assert client._facing_yaw(at(0.40, 0.40)) == pytest.approx(0.0)
        # turned 0.1 rad by an arm reach: command back to the held heading
        assert client._facing_yaw(at(0.30, 0.40)) == pytest.approx(0.0)
    finally:
        client.close()
