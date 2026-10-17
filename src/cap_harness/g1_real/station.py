"""Builds the G1 plant: the real SONIC link, or a fake body for off-robot runs."""

from __future__ import annotations

from cap_harness.g1_real.body_client import FakeBodyClient, SonicZmqBodyClient
from cap_harness.g1_real.codec import upper_body_from_arms
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env


def build_real_station(
    station: str = "g1-lab",
    *,
    config_root: str | None = None,
    enable_camera: bool = True,
) -> G1Env:
    """Connect to SONIC. Issues no command: ``engage()`` is the operator's step."""
    del enable_camera  # camera=None until the RGB-D client arrives (Task 12)
    cfg = load_g1_station_config(station, config_root=config_root)
    return G1Env(cfg, SonicZmqBodyClient(cfg.sonic), camera=None)


def build_fake_station(
    station: str = "g1-lab",
    *,
    config_root: str | None = None,
    tau_s: float = 0.02,
) -> G1Env:
    """An engaged G1 over a first-order fake body, starting at the profile's home."""
    from cap_harness.yam_real.cameras import SyntheticCamera

    cfg = load_g1_station_config(station, config_root=config_root)
    home = cfg.home_joint_positions
    body = FakeBodyClient(
        tau_s,
        initial_q17=upper_body_from_arms(cfg.waist_targets, home["left"], home["right"]),
    )
    env = G1Env(cfg, body, camera=SyntheticCamera())
    env.engage()
    return env
