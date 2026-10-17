import time

from g1_real_wire import decode_packed
import msgpack
import numpy as np
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
    assert abs(obs["heading_yaw"] - 0.6) < 1e-6
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


def test_heading_falls_back_to_base_quat_without_a_reference():
    q = [np.cos(0.3), 0, 0, np.sin(0.3)]
    yaw = SonicZmqBodyClient._heading({"planner_reference_active": [1], "base_quat": q})
    assert abs(yaw - 0.6) < 1e-6


def test_poll_thread_survives_a_subscriber_error():
    from unittest.mock import Mock

    client = SonicZmqBodyClient(_cfg(5750))
    real_sub = client._sub
    client._sub = Mock(get_msg=Mock(side_effect=RuntimeError("boom")))
    time.sleep(0.05)
    assert client._thread.is_alive() and client.feedback_age_s() == float("inf")
    client.close()
    real_sub.close()
