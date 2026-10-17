"""Explicit, immutable G1 station profile loading.

One station is one YAML file. Unlike YAM there is no hashed calibration bundle:
the camera extrinsic is the URDF mount, so the profile hash alone attributes an
episode to its configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from cap_harness.g1_real.codec import ARM_DOF, HAND_DOF
from cap_harness.yam_real.config import (
    _homogeneous,
    _port,
    _positive,
    _readonly_array,
    _sha256_file,
)

ARM_NAMES = ("left", "right")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_ROOT = _PACKAGE_ROOT / "configs" / "g1_real"


@dataclass(frozen=True, slots=True)
class G1ArmConfig:
    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    home_joints: np.ndarray


@dataclass(frozen=True, slots=True)
class G1HandConfig:
    joint_names: tuple[str, ...]
    open: np.ndarray
    closed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    max_step_rad: float


@dataclass(frozen=True, slots=True)
class G1SonicConfig:
    command_bind: str
    feedback_host: str
    feedback_port: int
    feedback_topic: str
    mode: int
    speed: float
    height: float
    feedback_stale_s: float
    engage_hold_s: float


@dataclass(frozen=True, slots=True)
class G1ControllerConfig:
    control_frequency_hz: float
    command_stream_hz: float
    max_joint_step_rad: float
    final_joint_tolerance_rad: float


@dataclass(frozen=True, slots=True)
class G1CameraConfig:
    role: str
    host: str
    port: int
    depth_scale_m: float
    max_acquisition_elapsed_s: float
    base_from_camera: np.ndarray


@dataclass(frozen=True, slots=True)
class G1StationConfig:
    station: str
    base_frame: str
    default_arm: str
    arm_aliases: Mapping[str, str]
    arms: Mapping[str, G1ArmConfig]
    hands: Mapping[str, G1HandConfig]
    waist_targets: np.ndarray
    sonic: G1SonicConfig
    controller: G1ControllerConfig
    camera: G1CameraConfig
    planning_workspace_lower: np.ndarray
    planning_workspace_upper: np.ndarray
    model_xml: Path
    model_urdf: Path
    config_sha256: str
    source_path: Path

    @property
    def joint_limits_lower(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({name: arm.lower for name, arm in self.arms.items()})

    @property
    def joint_limits_upper(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({name: arm.upper for name, arm in self.arms.items()})

    @property
    def home_joint_positions(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({name: arm.home_joints for name, arm in self.arms.items()})


def _station_path(station: str, config_root: str | Path | None) -> Path:
    root = _DEFAULT_CONFIG_ROOT if config_root is None else Path(config_root).expanduser().resolve()
    flat = root / f"{station}.yaml"
    if flat.is_file():
        return flat
    for candidate in sorted(root.glob("*/station.yaml")):
        declared = yaml.safe_load(candidate.read_bytes())
        if isinstance(declared, dict) and declared.get("station") == station:
            return candidate
    raise FileNotFoundError(f"no G1 station profile {station!r} under {root}")


def _within(value: np.ndarray, lower: np.ndarray, upper: np.ndarray, name: str) -> None:
    if np.any(value < lower) or np.any(value > upper):
        raise ValueError(f"{name} must lie within its joint limits")


def _names(raw: object, count: int, name: str) -> tuple[str, ...]:
    names = tuple(str(item) for item in raw)  # type: ignore[attr-defined]
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"{name} must contain {count} unique names")
    return names


def load_g1_station_config(
    station: str,
    *,
    config_root: str | Path | None = None,
) -> G1StationConfig:
    """Load and validate one explicit G1 station profile."""
    path = _station_path(station, config_root)
    data = yaml.safe_load(path.read_bytes())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("G1 station profile must use schema_version 1")
    if data.get("station") != station:
        raise ValueError("station profile identity does not match the requested station")

    default_arm = data.get("default_arm")
    aliases = data.get("arm_aliases")
    if default_arm not in ARM_NAMES:
        raise ValueError(f"default_arm must be one of {ARM_NAMES}, got {default_arm!r}")
    if not isinstance(aliases, dict) or set(aliases) - {"primary", "secondary"}:
        raise ValueError("arm_aliases may only map 'primary' and 'secondary'")
    if set(aliases.values()) - set(ARM_NAMES) or len(set(aliases.values())) != len(aliases):
        raise ValueError("arm_aliases must point at distinct real arms")
    if aliases.get("primary") != default_arm:
        raise ValueError("arm_aliases['primary'] must be the default_arm")

    raw_arms, raw_hands = data.get("arms"), data.get("hands")
    if not isinstance(raw_arms, dict) or set(raw_arms) != set(ARM_NAMES):
        raise ValueError("G1 station must define exactly left and right arms")
    if not isinstance(raw_hands, dict) or set(raw_hands) != set(ARM_NAMES):
        raise ValueError("G1 station must define exactly left and right hands")

    arms: dict[str, G1ArmConfig] = {}
    hands: dict[str, G1HandConfig] = {}
    for side in ARM_NAMES:
        raw = raw_arms[side]
        lower = _readonly_array(raw["lower"], f"arms.{side}.lower", (ARM_DOF,))
        upper = _readonly_array(raw["upper"], f"arms.{side}.upper", (ARM_DOF,))
        if np.any(lower >= upper):
            raise ValueError(f"arms.{side} lower bounds must be below upper bounds")
        home = _readonly_array(raw["home_joints"], f"arms.{side}.home_joints", (ARM_DOF,))
        _within(home, lower, upper, f"arms.{side}.home_joints")
        arms[side] = G1ArmConfig(
            joint_names=_names(raw["joint_names"], ARM_DOF, f"arms.{side}.joint_names"),
            lower=lower,
            upper=upper,
            home_joints=home,
        )

        raw = raw_hands[side]
        lower = _readonly_array(raw["lower"], f"hands.{side}.lower", (HAND_DOF,))
        upper = _readonly_array(raw["upper"], f"hands.{side}.upper", (HAND_DOF,))
        if np.any(lower >= upper):
            raise ValueError(f"hands.{side} lower bounds must be below upper bounds")
        open_ = _readonly_array(raw["open"], f"hands.{side}.open", (HAND_DOF,))
        closed = _readonly_array(raw["closed"], f"hands.{side}.closed", (HAND_DOF,))
        _within(open_, lower, upper, f"hands.{side}.open")
        _within(closed, lower, upper, f"hands.{side}.closed")
        hands[side] = G1HandConfig(
            joint_names=_names(raw["joint_names"], HAND_DOF, f"hands.{side}.joint_names"),
            open=open_,
            closed=closed,
            lower=lower,
            upper=upper,
            max_step_rad=_positive(raw["max_step_rad"], f"hands.{side}.max_step_rad"),
        )

    raw_sonic = data["sonic"]
    sonic = G1SonicConfig(
        command_bind=str(raw_sonic["command_bind"]),
        feedback_host=str(raw_sonic["feedback_host"]),
        feedback_port=_port(raw_sonic["feedback_port"], "sonic.feedback_port"),
        feedback_topic=str(raw_sonic["feedback_topic"]),
        mode=int(raw_sonic["mode"]),
        speed=float(raw_sonic["speed"]),
        height=float(raw_sonic["height"]),
        feedback_stale_s=_positive(raw_sonic["feedback_stale_s"], "sonic.feedback_stale_s"),
        engage_hold_s=_positive(raw_sonic["engage_hold_s"], "sonic.engage_hold_s"),
    )

    raw_controller = data["controller"]
    controller = G1ControllerConfig(
        control_frequency_hz=_positive(raw_controller["control_frequency_hz"], "control frequency"),
        command_stream_hz=_positive(raw_controller["command_stream_hz"], "command stream rate"),
        max_joint_step_rad=_positive(raw_controller["max_joint_step_rad"], "max joint step"),
        final_joint_tolerance_rad=_positive(
            raw_controller["final_joint_tolerance_rad"], "final joint tolerance"
        ),
    )

    raw_camera = data["camera"]
    camera = G1CameraConfig(
        role=str(raw_camera["role"]),
        host=str(raw_camera["host"]),
        port=_port(raw_camera["port"], "camera.port"),
        depth_scale_m=_positive(raw_camera["depth_scale_m"], "camera.depth_scale_m"),
        max_acquisition_elapsed_s=_positive(
            raw_camera["max_acquisition_elapsed_s"], "camera.max_acquisition_elapsed_s"
        ),
        base_from_camera=_homogeneous(raw_camera["base_from_camera"], "camera.base_from_camera"),
    )

    raw_workspace = data["planning_workspace"]
    workspace_lower = _readonly_array(raw_workspace["lower"], "planning_workspace.lower", (3,))
    workspace_upper = _readonly_array(raw_workspace["upper"], "planning_workspace.upper", (3,))
    if np.any(workspace_lower >= workspace_upper):
        raise ValueError("planning workspace lower bounds must be below upper bounds")

    raw_model = data["model"]
    return G1StationConfig(
        station=station,
        base_frame=str(data["base_frame"]),
        default_arm=default_arm,
        arm_aliases=MappingProxyType(dict(aliases)),
        arms=MappingProxyType(arms),
        hands=MappingProxyType(hands),
        waist_targets=_readonly_array(data["waist_targets"], "waist_targets", (3,)),
        sonic=sonic,
        controller=controller,
        camera=camera,
        planning_workspace_lower=workspace_lower,
        planning_workspace_upper=workspace_upper,
        model_xml=_PACKAGE_ROOT / str(raw_model["xml"]),
        model_urdf=_PACKAGE_ROOT / str(raw_model["urdf"]),
        config_sha256=_sha256_file(path),
        source_path=path,
    )
