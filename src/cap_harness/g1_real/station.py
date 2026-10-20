"""Builds the G1 plant: the real SONIC link, or a fake body for off-robot runs."""

from __future__ import annotations

from cap_harness.g1_real.body_client import FakeBodyClient, SonicZmqBodyClient
from cap_harness.g1_real.cameras import ComposedCameraRgbd
from cap_harness.g1_real.codec import upper_body_from_arms
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.yam_real.cameras import SyntheticCamera


def build_real_station(
    station: str = "g1-lab",
    *,
    config_root: str | None = None,
    enable_camera: bool = True,
) -> G1Env:
    """Connect to SONIC. Issues no command: ``engage()`` is the operator's step."""
    cfg = load_g1_station_config(station, config_root=config_root)
    # Before the camera: a failure must not leak it.
    body = SonicZmqBodyClient(
        cfg.sonic, hand_max_step_rad={side: hand.max_step_rad for side, hand in cfg.hands.items()}
    )
    camera = (
        ComposedCameraRgbd(
            cfg.camera.host,
            cfg.camera.port,
            role=cfg.camera.role,
            depth_scale_m=cfg.camera.depth_scale_m,
        )
        if enable_camera
        # The Observation contract needs a camera, so "no camera" is synthetic
        # frames: enough for a run that only measures the arms.
        else SyntheticCamera()
    )
    return G1Env(cfg, body, camera=camera)


def build_fake_station(
    station: str = "g1-lab",
    *,
    config_root: str | None = None,
    tau_s: float = 0.02,
) -> G1Env:
    """An engaged G1 over a first-order fake body, starting at the profile's home."""
    cfg = load_g1_station_config(station, config_root=config_root)
    home = cfg.home_joint_positions
    body = FakeBodyClient(
        tau_s,
        initial_q17=upper_body_from_arms(cfg.waist_targets, home["left"], home["right"]),
    )
    env = G1Env(cfg, body, camera=SyntheticCamera())
    env.engage()
    return env
