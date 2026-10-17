# G1 Real Embodiment (ASPIRE on Unitree G1 + GEAR-SONIC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `g1_real` embodiment to ASPIRE so System 2 drives the user's Unitree G1 arms and Dex3 hands through GEAR-SONIC's `zmq_manager` planner stream, with the same CAP tool contract as YAM.

**Architecture:** A new `cap_harness/g1_real/` package mirrors `yam_real/` (one `execute_action_batch` chokepoint, a 50 Hz level-1 streamer, a thin adapter) but the plant is a `G1BodyClient` that publishes 17 upper-body joints + 2×7 hand joints on SONIC's ZMQ `planner` topic and reads `g1_debug` feedback. Shared code is generalized where it hard-codes YAM (joint dimension, codec injection in the adapter, recorder width, benchmark switches); IK is mink over a torso-rooted MuJoCo model, planning is a new cuRobo `g1_real` model, perception is the robot-side RealSense server patched to publish aligned depth and intrinsics.

**Tech Stack:** Python 3.12, numpy, pyzmq, msgpack + msgpack-numpy, mink + mujoco 3.3, cuRobo service (existing), SAM3 service (existing), FastAPI/MCP server (existing), pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-g1-sonic-aspire-design.md`

## Global Constraints

- Every motion goes through `G1Env.execute_action_batch`; no other code path publishes a planner message with joint targets (spec anti-goal).
- Arm joint order per side is Unitree order `[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]`; SONIC 17-order is `[waist_yaw, waist_roll, waist_pitch, L/R shoulder_pitch, L/R shoulder_roll, L/R shoulder_yaw, L/R elbow, L/R wrist_roll, L/R wrist_pitch, L/R wrist_yaw]`; feedback `body_q` is MuJoCo order where the 17 upper-body indices are `[12,13,14,15,22,16,23,17,24,18,25,19,26,20,27,21,28]` (D1, §3).
- Base frame name is `"torso"`; waist targets are a fixed profile vector sent every tick (D2).
- Stream and keep-alive rate 50 Hz; per-message joint step clamp 0.03 rad; hand step clamp 0.25 rad; feedback older than 0.5 s refuses the next command; SONIC's planner timeout is 1.0 s (D4).
- Hands: `g1_real.set_hand_joints(arm, joints)` and `g1_real.get_hand_state()`; `set_gripper(p)` is linear between profile `hand_open` (zeros) and `hand_closed` (left `[0,0,1.75,-1.57,-1.75,-1.57,-1.75]`, right `[0,0,-1.75,1.57,1.75,1.57,1.75]`) (D5).
- ASPIRE never sends `command.stop`; `command.start` is sent only by the operator engage path, never by an agent tool (D11).
- Vendored SONIC files are copied verbatim from `~/work/GR00T-WholeBodyControl` with an Apache-2.0 provenance header; no import of `xr_upperbody_bridge` (D3).
- Home pose (SONIC-17) `[0,0,0, 0.2,0.2, 0.2,-0.2, 0,0, 0.6,0.6, 0,0,0,0,0,0]`, i.e. per arm left `[0.2,0.2,0,0.6,0,0,0]`, right `[0.2,-0.2,0,0.6,0,0,0]` (D10).
- Residual tolerance 0.10 rad after a trajectory; home diagnostic tolerance 0.03 rad (unchanged from YAM).
- Hosts/ports in the shipped profile: PUB bind `tcp://*:5556`, feedback `tcp://192.168.0.223:5557` topic `g1_debug`, camera `tcp://192.168.0.223:5555` (D12, §8).
- Commit after every task; run `pytest tests -q` before each commit.

## Review Focus

1. **Feedback stops while a trajectory is streaming** (deploy killed, cable out): the batch must abort with `EXECUTION_FAILED` within one tick rather than stream blind against SONIC's idle fallback. Test in Task 6.
2. **A `move_to_pose` whose first waypoint is far from the measured pose** (IK branch flip): the 0.03 rad/msg clamp must stretch the motion, not teleport, and the residual check must then report the true arrival. Test in Task 6.
3. **`set_gripper` while a hand is already partly closed on an object**: reported scalar must come from measured hand joints, and the command must not reopen the hand when a later joint-only motion holds the gripper. Test in Task 6.
4. **Engage called twice or with stale feedback**: second engage must be a no-op and a stale-feedback engage must refuse before any publish. Test in Task 6.
5. **Camera message without depth** (operator forgot `--realsense-enable-depth`): `observe` works for images, `cap_localize_object` returns a typed failure naming the missing depth, nothing crashes. Test in Task 12.

---

### Task 1: Register the `g1_real` embodiment in the shared contracts

**Files:**
- Modify: `src/cap_harness/contracts.py:15-22`
- Modify: `src/cap_harness/registry.py:74-83`
- Modify: `src/cap_harness/providers/curobo/service.py:68,617,747`
- Test: `tests/test_g1_real_contracts.py`

**Interfaces:**
- Produces: `JOINT_DIMENSIONS["g1_real"] == 7`; `G1_REAL_PUBLIC_TOOL_NAMES` frozenset; `"g1_real."` in `_APPROVED_EMBODIMENT_PREFIXES`; cuRobo `BaseRequest.model` accepts `"g1_real"`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from cap_harness.contracts import JOINT_DIMENSIONS, Pose, RobotState
from cap_harness.registry import G1_REAL_PUBLIC_TOOL_NAMES, ToolRegistry

def test_g1_real_joint_dimension_is_seven():
    assert JOINT_DIMENSIONS["g1_real"] == 7

def test_robot_state_accepts_seven_joint_g1_arms():
    pose = Pose(position=np.zeros(3), quaternion_wxyz=np.array([1.0, 0, 0, 0]), frame="torso")
    state = RobotState(
        joint_positions={"left": np.zeros(7), "right": np.zeros(7)},
        joint_velocities={"left": np.zeros(7), "right": np.zeros(7)},
        end_effector_poses={"left": pose, "right": pose},
        gripper_positions={"left": 1.0, "right": 1.0},
        base_frame="torso",
        embodiment="g1_real",
    )
    assert state.joint_positions["left"].shape == (7,)

def test_g1_real_extension_names_are_approved():
    assert G1_REAL_PUBLIC_TOOL_NAMES == frozenset({
        "g1_real.get_task_metadata", "g1_real.get_controller_metadata",
        "g1_real.set_hand_joints", "g1_real.get_hand_state",
    })
    ToolRegistry(public_extension_allowlist=G1_REAL_PUBLIC_TOOL_NAMES)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_g1_real_contracts.py -v`
Expected: FAIL with `KeyError: 'g1_real'` and `ImportError: G1_REAL_PUBLIC_TOOL_NAMES`.

- [ ] **Step 3: Add `"g1_real": 7` to `JOINT_DIMENSIONS`; add `G1_REAL_PUBLIC_TOOL_NAMES` (the four names above) and `"g1_real."` to `_APPROVED_EMBODIMENT_PREFIXES`; export it in `__all__`.**

- [ ] **Step 4: In `providers/curobo/service.py` widen `model: Literal["panda", "dual_panda", "yam_real", "g1_real"]` and replace both `JOINT_DIMENSIONS["yam_real" if request.model == "yam_real" else "robosuite"]` lookups (lines 617, 747) with `JOINT_DIMENSIONS.get(request.model, JOINT_DIMENSIONS["robosuite"])`.**

- [ ] **Step 5: Run the full suite**

Run: `pytest tests -q`
Expected: all pass (the service file is imported by `tests/test_providers.py` without cuRobo installed).

- [ ] **Step 6: Commit**

```bash
git add src/cap_harness/contracts.py src/cap_harness/registry.py src/cap_harness/providers/curobo/service.py tests/test_g1_real_contracts.py
git commit -m "feat: register g1_real embodiment with seven-joint arms"
```

---

### Task 2: Vendor the SONIC wire modules and define the joint-order codec

**Files:**
- Create: `src/cap_harness/g1_real/__init__.py` (empty)
- Create: `src/cap_harness/g1_real/sonic/__init__.py` (empty)
- Create: `src/cap_harness/g1_real/sonic/zmq_planner_sender.py` (verbatim copy of `~/work/GR00T-WholeBodyControl/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`, 224 lines)
- Create: `src/cap_harness/g1_real/sonic/zmq_state_subscriber.py` (verbatim copy of `~/work/GR00T-WholeBodyControl/gear_sonic/utils/data_collection/zmq_state_subscriber.py`, 138 lines)
- Create: `src/cap_harness/g1_real/codec.py`
- Modify: `pyproject.toml:34-39` (add extra `g1-real`)
- Test: `tests/test_g1_real_codec.py`, `tests/g1_real_wire.py` (helper, not a test module)

**Interfaces:**
- Produces, in `codec.py`: `EMBODIMENT = "g1_real"`, `ARMS = ("left", "right")`, `ARM_DOF = 7`, `BASE_FRAME = "torso"`, `UPPER_BODY_DOF = 17`, `HAND_DOF = 7`, `JOINT_NAMES` (7 Unitree names without side prefix), `MUJOCO_UPPER_BODY_INDICES = (12,13,14,15,22,16,23,17,24,18,25,19,26,20,27,21,28)`, `MUJOCO_ARM_SLICES = {"left": slice(15, 22), "right": slice(22, 29)}`, `MUJOCO_WAIST_SLICE = slice(12, 15)`;
  `upper_body_from_arms(waist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray` (17,), `arms_from_upper_body(q17) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (waist, left, right), `upper_body_from_body_q(body_q29) -> np.ndarray` (17,), `hand_scalar(hand_q, hand_open, hand_closed) -> float`, `hand_pose(p, hand_open, hand_closed) -> np.ndarray` (7,), `heading_yaw_from_quat_wxyz(q) -> float`, plus `robot_state(env, *, timestamp_s=None) -> RobotState`, `observation(env, *, task_context=None) -> Observation`, `arm_command_targets(action) -> tuple[dict, dict]` with the same signatures as `yam_real.codec` (bodies in Task 6 once `G1Env` exists; this task ships the pure functions).
- Produces, in `tests/g1_real_wire.py`: `decode_packed(message: bytes, topic: bytes) -> dict[str, np.ndarray]` implementing the C++ header parser (1280-byte NUL-padded JSON header; dtypes `u8→np.uint8, i32→"<i4", i64→"<i8", f32→"<f4", f64→"<f8"`; fields read in header order with their `shape`).

- [ ] **Step 1: Add the `g1-real` extra to `pyproject.toml`**

```toml
g1-real = [
  "imageio[ffmpeg]>=2.34",
  "mink>=0.0.4",
  "mujoco==3.3.0",
  "pyzmq>=25",
  "msgpack>=1.0",
  "msgpack-numpy>=0.4.8",
]
```

Then run `uv lock` and commit the updated `uv.lock`. Add `--extra g1-real` to the `uv sync` line in `scripts/bootstrap_system2.sh` and re-run it so `.venv-system2` has pyzmq/msgpack.

- [ ] **Step 2: Copy the two upstream files verbatim, prepending a comment header** `# Vendored verbatim from NVlabs/GR00T-WholeBodyControl @ 45c03fa (Apache-2.0). Do not edit; re-vendor.` Verify with `diff` against the fork that only the header differs.

- [ ] **Step 3: Write the failing codec tests**

```python
import numpy as np
from cap_harness.g1_real import codec
from g1_real_wire import decode_packed
from cap_harness.g1_real.sonic.zmq_planner_sender import build_planner_message

# Copied from the fork: gear_sonic/envs/manager_env/robots/g1.py
G1_ISAACLAB_TO_MUJOCO_DOF = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]
# Upstream C++: upper_body_joint_isaaclab_order_in_isaaclab_index
UPPER_BODY_ISAACLAB = [2,5,8,11,12,15,16,19,20,21,22,23,24,25,26,27,28]

def test_mujoco_upper_body_indices_match_the_isaaclab_table():
    expected = [G1_ISAACLAB_TO_MUJOCO_DOF[i] for i in UPPER_BODY_ISAACLAB]
    assert list(codec.MUJOCO_UPPER_BODY_INDICES) == expected

def test_arm_split_round_trips():
    body_q = np.arange(29.0)
    q17 = codec.upper_body_from_body_q(body_q)
    waist, left, right = codec.arms_from_upper_body(q17)
    np.testing.assert_array_equal(left, body_q[15:22])
    np.testing.assert_array_equal(right, body_q[22:29])
    np.testing.assert_array_equal(waist, body_q[12:15])
    np.testing.assert_array_equal(codec.upper_body_from_arms(waist, left, right), q17)

def test_hand_scalar_is_inverse_of_hand_pose():
    opened = np.zeros(7); closed = np.array([0,0,1.75,-1.57,-1.75,-1.57,-1.75])
    for p in (0.0, 0.25, 1.0):
        assert abs(codec.hand_scalar(codec.hand_pose(p, opened, closed), opened, closed) - p) < 1e-9
    assert codec.hand_scalar(closed * 1.3, opened, closed) == 0.0  # clamped

def test_heading_yaw_of_identity_is_zero():
    assert codec.heading_yaw_from_quat_wxyz([1, 0, 0, 0]) == 0.0
    assert abs(codec.heading_yaw_from_quat_wxyz([np.cos(0.4), 0, 0, np.sin(0.4)]) - 0.8) < 1e-9

def test_wire_decoder_reads_a_planner_message():
    msg = build_planner_message(0, [0,0,0], [1,0,0], upper_body_position=list(range(17)),
                                upper_body_velocity=[0.0]*17, left_hand_position=[1.0]*7,
                                right_hand_position=[2.0]*7)
    fields = decode_packed(msg, b"planner")
    assert fields["mode"].tolist() == [0]
    np.testing.assert_allclose(fields["upper_body_position"], np.arange(17, dtype=np.float32))
    assert fields["right_hand_position"].shape == (7,)
```

- [ ] **Step 4: Run to verify they fail**

Run: `pytest tests/test_g1_real_codec.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 5: Implement `codec.py` pure functions and `tests/g1_real_wire.py::decode_packed`.** `hand_scalar` is `1 - clip(dot(q - open, closed - open) / |closed - open|², 0, 1)`; `heading_yaw_from_quat_wxyz` is `atan2(2(wz + xy), 1 - 2(y² + z²))`. Leave `robot_state`, `observation`, `arm_command_targets` as stubs raising `NotImplementedError` for Task 6.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_g1_real_codec.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock scripts/bootstrap_system2.sh src/cap_harness/g1_real tests/test_g1_real_codec.py tests/g1_real_wire.py
git commit -m "feat(g1_real): vendor SONIC wire modules and joint-order codec"
```

---

### Task 3: Station profile loader for G1

**Files:**
- Create: `src/cap_harness/g1_real/config.py`
- Create: `src/cap_harness/configs/g1_real/g1-lab/station.yaml`
- Test: `tests/test_g1_real_config.py`

**Interfaces:**
- Produces: frozen dataclasses `G1ArmConfig(joint_names: tuple[str,...], lower: np.ndarray(7), upper: np.ndarray(7), home_joints: np.ndarray(7))`, `G1HandConfig(joint_names, open: np.ndarray(7), closed: np.ndarray(7), lower, upper, max_step_rad: float)`, `G1SonicConfig(command_bind: str, feedback_host: str, feedback_port: int, feedback_topic: str, mode: int, speed: float, height: float, feedback_stale_s: float, engage_hold_s: float)`, `G1ControllerConfig(control_frequency_hz: float, command_stream_hz: float, max_joint_step_rad: float, final_joint_tolerance_rad: float)`, `G1CameraConfig(role: str, host: str, port: int, depth_scale_m: float, max_acquisition_elapsed_s: float, base_from_camera: np.ndarray(4,4))`, `G1StationConfig(station, base_frame, default_arm, arm_aliases, arms: Mapping[str, G1ArmConfig], hands: Mapping[str, G1HandConfig], waist_targets: np.ndarray(3), sonic: G1SonicConfig, controller: G1ControllerConfig, camera: G1CameraConfig, planning_workspace_lower, planning_workspace_upper, model_xml: Path, model_urdf: Path, config_sha256: str, source_path: Path)` with properties `joint_limits_lower/upper` (per-arm mapping) and `home_joint_positions`;
  `load_g1_station_config(station: str, *, config_root: str | Path | None = None) -> G1StationConfig` (resolves `*/station.yaml` under the root exactly like `yam_real.config.station_config_path`; default root is the package's `configs/g1_real`).
- Reuses `_readonly_array`, `_positive`, `_port`, `_homogeneous`, `_sha256_file` from `yam_real/config.py` by import.

- [ ] **Step 1: Write the shipped profile `g1-lab/station.yaml`** with: `station: g1-lab`, `base_frame: torso`, `default_arm: right`, `arm_aliases: {primary: right, secondary: left}`, per-arm `joint_names` (Unitree names with side prefix), `home_joints` from Global Constraints, `lower/upper` copied from the fork URDF `gear_sonic/data/robots/g1/g1_dual_arm.urdf` joint limits, `hands.{left,right}` with `open`, `closed` (Global Constraints), `lower/upper` from the same URDF's `*_hand_*` joints, `max_step_rad: 0.25`; `waist_targets: [0,0,0]`; `sonic: {command_bind: "tcp://*:5556", feedback_host: 192.168.0.223, feedback_port: 5557, feedback_topic: g1_debug, mode: 0, speed: -1.0, height: -1.0, feedback_stale_s: 0.5, engage_hold_s: 0.2}`; `controller: {control_frequency_hz: 50, command_stream_hz: 50, max_joint_step_rad: 0.03, final_joint_tolerance_rad: 0.03}`; `camera: {role: ego_view, host: 192.168.0.223, port: 5555, depth_scale_m: 0.001, max_acquisition_elapsed_s: 0.5, base_from_camera: <4x4 from the URDF d435_joint origin expressed in torso_link>}`; `planning_workspace: {lower: [-0.2,-0.8,-0.6], upper: [0.9,0.8,0.6]}`; `model: {xml: g1_real/description/g1_upper_body.xml, urdf: g1_real/description/g1_dual_arm.urdf}`.

- [ ] **Step 2: Write the failing tests**

```python
import numpy as np, xml.etree.ElementTree as ET
from pathlib import Path
from cap_harness.g1_real.config import load_g1_station_config

FORK_URDF = Path.home() / "work/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_dual_arm.urdf"

def test_profile_loads_with_seven_joint_arms_and_torso_base():
    cfg = load_g1_station_config("g1-lab")
    assert cfg.base_frame == "torso" and cfg.arms["left"].home_joints.shape == (7,)
    assert cfg.arm_aliases["primary"] == "right"
    assert cfg.hands["right"].closed[2] == -1.75 and cfg.sonic.feedback_port == 5557

def test_profile_limits_match_the_fork_urdf():
    if not FORK_URDF.is_file():
        import pytest; pytest.skip("fork URDF not present")
    cfg = load_g1_station_config("g1-lab")
    limits = {j.get("name"): j.find("limit") for j in ET.parse(FORK_URDF).getroot().iter("joint") if j.find("limit") is not None}
    for side in ("left", "right"):
        for i, name in enumerate(cfg.arms[side].joint_names):
            assert float(limits[name].get("lower")) == cfg.arms[side].lower[i]
            assert float(limits[name].get("upper")) == cfg.arms[side].upper[i]

def test_unknown_station_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_g1_station_config("nope")
```

- [ ] **Step 3: Run to verify they fail** — `pytest tests/test_g1_real_config.py -v` → `ModuleNotFoundError`.

- [ ] **Step 4: Implement `config.py`.** Validation rules: every arm vector shape (7,), finite; `home_joints` inside limits; `closed`/`open` inside hand limits; `base_from_camera` via `_homogeneous`; `config_sha256` of the YAML bytes.

- [ ] **Step 5: Run tests → PASS; commit**

```bash
git add src/cap_harness/g1_real/config.py src/cap_harness/configs/g1_real tests/test_g1_real_config.py
git commit -m "feat(g1_real): station profile loader and g1-lab profile"
```

---

### Task 4: Body client port, SONIC ZMQ implementation, and the lagging fake

**Files:**
- Create: `src/cap_harness/g1_real/body_client.py`
- Test: `tests/test_g1_real_body_client.py`

**Interfaces:**
- Produces: `class G1BodyClient(Protocol)` with
  `get_observations() -> dict[str, np.ndarray | float] | None` (keys `left_joint_pos(7), right_joint_pos(7), waist(3), left_joint_vel(7), right_joint_vel(7), left_hand_q(7), right_hand_q(7), base_quat(4), heading_yaw: float, age_s: float, timestamp_s: float`; `None` before the first feedback),
  `command_upper_body(q17: np.ndarray, dq17: np.ndarray, left_hand: np.ndarray | None, right_hand: np.ndarray | None) -> None`,
  `send_start() -> None`, `feedback_age_s() -> float` (inf before first feedback), `close() -> None`.
- `class SonicZmqBodyClient(config: G1SonicConfig)`: PUB bound to `config.command_bind`; SUB via vendored `ZMQStateSubscriber(host, port, topic)`; `command_upper_body` encodes with `build_planner_message(mode=config.mode, movement=(0,0,0), facing=(cos ψ, sin ψ, 0), speed=config.speed, height=config.height, upper_body_position=q17, upper_body_velocity=dq17, left_hand_position=..., right_hand_position=...)` where ψ is `heading_yaw` from the latest feedback (`reference_heading_quat` when `planner_reference_active == 1`, else `base_quat`), and `(1,0,0)` when no feedback; `send_start` sends `build_command_message(start=True, stop=False, planner=True)`. Hand arrays are clamped to the per-message step `0.25` rad against the last sent hand command (same rule as the fork's `HandCommandLimiter`).
- `class FakeBodyClient(tau_s: float = 0.05, initial_q17=None)`: first-order lag toward the last command using monotonic time, with `tau_s` kept as a public mutable attribute (tests slow it mid-run); records every `command_upper_body` call in `.commands: list[tuple[q17, dq17, lh, rh]]`; `.start_count`; `.feedback_age_s()` returns 0 unless `.stale = True`. Hands lag with the same `tau_s`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, time, zmq
from cap_harness.g1_real.body_client import FakeBodyClient, SonicZmqBodyClient
from cap_harness.g1_real.config import G1SonicConfig
from g1_real_wire import decode_packed

def _cfg(port):
    return G1SonicConfig(command_bind=f"tcp://127.0.0.1:{port}", feedback_host="127.0.0.1",
        feedback_port=port + 1, feedback_topic="g1_debug", mode=0, speed=-1.0, height=-1.0,
        feedback_stale_s=0.5, engage_hold_s=0.2)

def test_fake_client_lags_then_converges():
    fake = FakeBodyClient(tau_s=0.02)
    fake.command_upper_body(np.full(17, 0.5), np.zeros(17), None, None)
    first = fake.get_observations()["left_joint_pos"][0]
    time.sleep(0.2)
    assert first < 0.5 and abs(fake.get_observations()["left_joint_pos"][0] - 0.5) < 1e-3

def test_sonic_client_publishes_a_planner_message_the_cpp_parser_accepts():
    port = 5700; sub = zmq.Context.instance().socket(zmq.SUB); sub.subscribe(b""); sub.connect(f"tcp://127.0.0.1:{port}")
    client = SonicZmqBodyClient(_cfg(port)); time.sleep(0.3)
    client.command_upper_body(np.arange(17.0), np.zeros(17), np.zeros(7), np.full(7, 2.0))
    raw = sub.recv()
    assert raw.startswith(b"planner")
    f = decode_packed(raw, b"planner")
    assert f["mode"].tolist() == [0] and np.allclose(f["facing"], [1, 0, 0])
    assert np.allclose(f["upper_body_position"], np.arange(17))
    assert np.allclose(f["right_hand_position"], np.full(7, 0.25))  # step-clamped from 0
    client.close()

def test_sonic_client_reads_feedback_and_heading():
    port = 5710; pub = zmq.Context.instance().socket(zmq.PUB); pub.bind(f"tcp://127.0.0.1:{port+1}")
    client = SonicZmqBodyClient(_cfg(port)); time.sleep(0.3)
    import msgpack
    body_q = list(np.arange(29.0)); q = [np.cos(0.3), 0, 0, np.sin(0.3)]
    pub.send(b"g1_debug" + msgpack.packb({"body_q": body_q, "body_dq": [0.0]*29, "base_quat": q,
        "left_hand_q": [0.0]*7, "right_hand_q": [0.0]*7, "reference_heading_quat": q, "planner_reference_active": [1]}))
    time.sleep(0.2)
    obs = client.get_observations()
    assert np.allclose(obs["right_joint_pos"], np.arange(22.0, 29.0)) and abs(obs["heading_yaw"] - 0.6) < 1e-6
    assert client.feedback_age_s() < 0.5
    client.close()

def test_start_message_is_only_sent_on_request():
    port = 5720; sub = zmq.Context.instance().socket(zmq.SUB); sub.subscribe(b"command"); sub.connect(f"tcp://127.0.0.1:{port}")
    client = SonicZmqBodyClient(_cfg(port)); time.sleep(0.3)
    client.send_start(); f = decode_packed(sub.recv(), b"command")
    assert f["start"].tolist() == [1] and f["stop"].tolist() == [0] and f["planner"].tolist() == [1]
    client.close()
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_g1_real_body_client.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `body_client.py`.** Feedback polling: a daemon thread calls `ZMQStateSubscriber.get_msg(clear=False)` at 100 Hz and stores the decoded dict plus `time.monotonic()`; `get_observations` slices with `codec.MUJOCO_ARM_SLICES` / `MUJOCO_WAIST_SLICE`. Use `zmq.Context.instance()`; PUB `LINGER=0`.

- [ ] **Step 4: Run tests → PASS; commit**

```bash
git add src/cap_harness/g1_real/body_client.py tests/test_g1_real_body_client.py
git commit -m "feat(g1_real): SONIC ZMQ body client and lagging fake"
```

---

### Task 5: Kinematics model and mink IK for the torso-rooted G1

**Files:**
- Create: `scripts/build_g1_models.py`
- Create: `src/cap_harness/g1_real/description/g1_upper_body.xml` (generated, committed)
- Create: `src/cap_harness/g1_real/description/g1_dual_arm.urdf` (generated, committed)
- Create: `src/cap_harness/g1_real/kinematics.py`
- Modify: `src/cap_harness/providers/yam_kinematics/client.py:40-60` (parameterize `ARM_DOF`)
- Test: `tests/test_g1_real_kinematics.py`

**Interfaces:**
- `scripts/build_g1_models.py --fork ~/work/GR00T-WholeBodyControl --out src/cap_harness/g1_real/description`: from `gear_sonic_deploy/g1/g1_29dof_with_hand.xml` removes the `floating_base_joint`, the `left_hip_pitch_link` and `right_hip_pitch_link` subtrees, every `<geom type="mesh">` and the `<asset>` meshes, and adds `<site name="{side}_grasp_site" pos="0.19 0 0" quat="1 0 0 0"/>` as a child of `{side}_wrist_yaw_link`; from `gear_sonic/data/robots/g1/g1_dual_arm.urdf` strips `<visual>`/`<collision>` mesh elements and adds fixed links `{side}_grasp` (parent `{side}_wrist_yaw_link`, same xyz as the site). Both outputs get a provenance comment.
- `class G1Kinematics(model_path, *, position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)` with the **same method names and return conventions as `YamKinematics`**: `seed(left7, right7, waist3=None)`, `forward_kinematics(left7, right7, waist3=None) -> (left_pos, left_quat_xyzw, right_pos, right_quat_xyzw)` **expressed in the `torso_link` frame**, `inverse_kinematics(left_pos, left_quat_xyzw, right_pos, right_quat_xyzw, *, seeded=False, ...) -> (left7, right7)` (targets in torso frame; waist joints are fixed at the seeded value during the solve via a `mink.PostureTask` on the waist with cost 1e3), `grasp_site_jacobian(side, ...)`. `ARM_DOF = 7`.
- `YamKinematicsIKProvider(kinematics, *, tolerance_m=0.01, dof: int = 6)`: replace the module-level `ARM_DOF` uses with `self._dof`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, mujoco
from pathlib import Path
from cap_harness.g1_real.kinematics import G1Kinematics
from cap_harness.g1_real.config import load_g1_station_config

MODEL = Path("src/cap_harness/g1_real/description/g1_upper_body.xml")

def test_model_has_no_legs_and_both_grasp_sites():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert not any("hip" in n or "knee" in n or "ankle" in n for n in names)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site") >= 0

def test_fk_at_home_is_in_front_of_the_torso_and_mirrored():
    cfg = load_g1_station_config("g1-lab"); kin = G1Kinematics(MODEL)
    lp, _, rp, _ = kin.forward_kinematics(cfg.arms["left"].home_joints, cfg.arms["right"].home_joints)
    assert lp[0] > 0.15 and rp[0] > 0.15            # forward of torso_link
    assert abs(lp[1] + rp[1]) < 1e-6 and lp[1] > 0  # left arm on +Y

def test_ik_reaches_a_displaced_home_pose():
    cfg = load_g1_station_config("g1-lab"); kin = G1Kinematics(MODEL)
    home_l, home_r = cfg.arms["left"].home_joints, cfg.arms["right"].home_joints
    lp, lq, rp, rq = kin.forward_kinematics(home_l, home_r)
    kin.seed(home_l, home_r)
    l7, r7 = kin.inverse_kinematics(lp, lq, rp + np.array([0.05, 0, 0.05]), rq, seeded=True)
    _, _, rp2, _ = kin.forward_kinematics(l7, r7)
    assert np.linalg.norm(rp2 - (rp + [0.05, 0, 0.05])) < 0.01
    assert np.allclose(l7, home_l, atol=0.02)  # untouched arm holds

def test_ik_provider_accepts_seven_joint_state():
    from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider
    from cap_harness.contracts import Pose, RobotState
    cfg = load_g1_station_config("g1-lab"); kin = G1Kinematics(MODEL)
    home = {s: cfg.arms[s].home_joints for s in ("left", "right")}
    lp, lq, rp, rq = kin.forward_kinematics(home["left"], home["right"])
    def pose(p, q): return Pose(position=p, quaternion_wxyz=np.array([q[3], q[0], q[1], q[2]]), frame="torso")
    state = RobotState(joint_positions=home, joint_velocities={s: np.zeros(7) for s in home},
        end_effector_poses={"left": pose(lp, lq), "right": pose(rp, rq)}, gripper_positions={"left": 1.0, "right": 1.0},
        base_frame="torso", embodiment="g1_real")
    result = YamKinematicsIKProvider(kin, dof=7).solve_ik(pose(rp + [0.03, 0, 0], rq), state, arm="right")
    assert result.ok and result.joint_positions.shape == (7,)
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_g1_real_kinematics.py -v` → FileNotFoundError / ImportError.

- [ ] **Step 3: Write `scripts/build_g1_models.py` and run it; commit the two generated files.** ElementTree is enough; keep the hands' joints (mink needs them to exist, they are held by posture cost).

- [ ] **Step 4: Implement `kinematics.py`** (torso frame: compose `inv(T_world_torso) @ T_world_site` for FK; map targets to world with `T_world_torso` before `FrameTask.set_target`). Parameterize `YamKinematicsIKProvider` by `dof`.

- [ ] **Step 5: Run `pytest tests/test_g1_real_kinematics.py tests/test_yam_kinematics_provider.py -v`** → PASS (YAM provider still defaults to 6).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_g1_models.py src/cap_harness/g1_real/description src/cap_harness/g1_real/kinematics.py src/cap_harness/providers/yam_kinematics/client.py tests/test_g1_real_kinematics.py
git commit -m "feat(g1_real): torso-rooted G1 model and mink kinematics"
```

---

### Task 6: `G1Env`, the 50 Hz controller, keep-alive, engage, and the codec state functions

**Files:**
- Create: `src/cap_harness/g1_real/env.py`
- Create: `src/cap_harness/g1_real/control/__init__.py`, `src/cap_harness/g1_real/control/controller.py`
- Modify: `src/cap_harness/g1_real/codec.py` (fill `robot_state`, `observation`, `arm_command_targets`)
- Modify: `src/cap_harness/yam_real/action.py:118-126,145-160` (`dof = getattr(env, "dof", ARM_DOF)`)
- Modify: `src/cap_harness/yam_real/codec.py:170-195` (`observation(env, *, task_context=None, robot_state_fn=robot_state)`)
- Test: `tests/test_g1_real_env.py`

**Interfaces:**
- `class G1Env(config: G1StationConfig, body: G1BodyClient, *, camera=None, kinematics: G1Kinematics | None = None)`; attributes `config`, `kin`, `dof = 7`, `body`; methods mirroring `RealYamEnv`: `get_observations(side) -> {"joint_pos"(7), "joint_vel"(7), "gripper_pos"(1), "hand_q"(7), "ee_pos"(3), "ee_quat"(4, xyzw, torso frame)}`, `read_camera()`, `read_aux_camera(role)`, `aux_camera_roles`, `camera_aliases`, `camera_frame(alias)`, `render_rgb(alias)`, `execute_action_batch(batch, *, command_hz=None, start_interp_s=0.0, settle_s=0.2, playback_speed=1.0) -> dict` (the chokepoint; honours `batch.meta["hand_joints"] = {side: (N,7)}` as the hand column override, otherwise derives hands from the gripper scalar via `codec.hand_pose`), `set_gripper(side, position, *, duration_s=None) -> dict`, `set_hand_joints(side, joints7, *, duration_s=None) -> dict` (builds a two-waypoint batch holding arms and setting `meta["hand_joints"]`), `go_home(*, duration_s=3.0, keep_grippers=True) -> dict`, `engage() -> dict` (idempotent; raises `RuntimeError("feedback stale")` if `body.feedback_age_s() > config.sonic.feedback_stale_s`; otherwise seeds the hold command from measured state, streams it for `engage_hold_s` at 50 Hz, calls `body.send_start()`, starts the keep-alive), `keep_alive_running: bool`, `control_contract() -> dict` (`version: "g1-sonic-v1-planner-stream"`), `close()`, `_last_action_input`, `_action_input_seq` (recorder contract, same as YAM).
- `control/controller.py: execute_upper_body_trajectory(env, timestamps, left, right, left_hand, right_hand, *, command_hz, start_interp_s=0.0, settle_s=0.2, playback_speed=1.0) -> dict`: resamples like `yam_real.control.controller` (import `_sample`, `_raise_if_stopped` from it), clamps each emitted q17 to `±config.controller.max_joint_step_rad` from the previously emitted q17, and calls `env._send_tick(q17, dq17, lh, rh)` which checks `body.feedback_age_s()` before every publish and returns `{"success": False, "reason": "feedback stale"}` on violation. After the timestamped phase it keeps ticking at `command_hz` until the emitted (clamped) target equals the final waypoint (catch-up phase, bounded by `ceil(max|Δq| / max_joint_step_rad) + 1` ticks), then holds for `settle_s`; the result dict reports `catch_up_ticks`.
- `codec.robot_state(env)`: 7-joint arms, `gripper_positions` from `hand_scalar`, `base_frame "torso"`, `embodiment "g1_real"`. `codec.observation(env, *, task_context=None)` delegates to `yam_real.codec.observation(env, task_context=..., robot_state_fn=robot_state)`.

- [ ] **Step 1: Write the failing tests** (all with `FakeBodyClient`; `SyntheticCamera` from `yam_real.cameras` as `camera`)

```python
import numpy as np, time, pytest
from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.g1_real import codec
from cap_harness.yam_real.action import YamActionBatch
from cap_harness.yam_real.cameras import SyntheticCamera

@pytest.fixture
def env():
    cfg = load_g1_station_config("g1-lab")
    fake = FakeBodyClient(tau_s=0.01, initial_q17=codec.upper_body_from_arms(cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints))
    e = G1Env(cfg, fake, camera=SyntheticCamera()); e.engage(); yield e; e.close()

def test_step_clamp_stretches_a_jump(env):
    start = env.get_observations("right")["joint_pos"].copy(); target = start.copy(); target[0] += 0.5
    env.execute_action_batch(YamActionBatch.joint_abs([0.0], [env.get_observations("left")["joint_pos"]], [target], source="test"), settle_s=0.0)
    steps = np.diff([c[0][4] for c in env.body.commands])  # right shoulder_pitch is SONIC-17 index 4
    assert np.max(np.abs(steps)) <= 0.03 + 1e-9 and len(env.body.commands) >= 16

def test_keep_alive_resends_last_target_between_batches(env):
    n = len(env.body.commands); time.sleep(0.25)
    assert len(env.body.commands) - n >= 8 and env.keep_alive_running

def test_stale_feedback_aborts_a_batch(env):
    env.body.stale = True
    result = env.execute_action_batch(YamActionBatch.joint_abs([0.0, 0.5], [np.zeros((2, 7))], [np.zeros((2, 7))], source="test"))
    assert result["success"] is False and "stale" in result["reason"]

def test_gripper_scalar_round_trips_through_the_hand(env):
    env.set_gripper("right", 0.3); time.sleep(0.1)
    assert abs(env.get_observations("right")["gripper_pos"][0] - 0.3) < 0.05
    sent = env.body.commands[-1][3]
    np.testing.assert_allclose(sent, codec.hand_pose(0.3, env.config.hands["right"].open, env.config.hands["right"].closed), atol=1e-6)

def test_joint_only_batch_holds_the_measured_hand(env):
    env.set_gripper("right", 0.2); time.sleep(0.1)
    env.execute_action_batch(YamActionBatch.joint_abs([0.0], [env.get_observations("left")["joint_pos"]], [env.get_observations("right")["joint_pos"]], source="test"), settle_s=0.0)
    assert abs(codec.hand_scalar(env.body.commands[-1][3], env.config.hands["right"].open, env.config.hands["right"].closed) - 0.2) < 0.05

def test_set_hand_joints_overrides_scalar_mapping(env):
    q = np.array([0.1, 0.2, 0.3, -0.4, -0.5, -0.6, -0.7])
    env.set_hand_joints("left", q); np.testing.assert_allclose(env.body.commands[-1][2], q, atol=1e-6)

def test_engage_is_idempotent_and_refuses_stale_feedback():
    cfg = load_g1_station_config("g1-lab"); fake = FakeBodyClient(); e = G1Env(cfg, fake, camera=SyntheticCamera())
    fake.stale = True
    with pytest.raises(RuntimeError): e.engage()
    assert fake.start_count == 0 and not e.keep_alive_running
    fake.stale = False; e.engage(); e.engage()
    assert fake.start_count == 1; e.close()

def test_robot_state_is_seven_joint_torso_frame(env):
    state = codec.robot_state(env)
    assert state.embodiment == "g1_real" and state.base_frame == "torso" and state.joint_positions["left"].shape == (7,)
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_g1_real_env.py -v` → ImportError.

- [ ] **Step 3: Parameterize `yam_real/action.py` by `env.dof`** (three places: `_as_rows` minimum width, `[:, :dof]` slices, joint7 gripper detection `> dof`). Add `robot_state_fn` to `yam_real.codec.observation`. Run `pytest tests/test_yam_real_codec.py tests/test_yam_real_adapter.py -q` → still PASS.

- [ ] **Step 4: Implement `controller.py`, `env.py`, and fill `codec.robot_state/observation/arm_command_targets`.** One `threading.RLock` guards every publish; `execute_action_batch` sets `self._streaming = True` so the keep-alive thread skips its tick; the keep-alive sends the last `(q17, dq17=0, lh, rh)` every 20 ms once `engage()` has run.

- [ ] **Step 5: Run tests → PASS; commit**

```bash
git add src/cap_harness/g1_real src/cap_harness/yam_real/action.py src/cap_harness/yam_real/codec.py tests/test_g1_real_env.py
git commit -m "feat(g1_real): plant with 50 Hz planner stream, keep-alive and engage"
```

---

### Task 7: Codec-injected adapter, `G1RealAdapter`, task registry, hand extension tools

**Files:**
- Modify: `src/cap_harness/yam_real/adapter.py` (replace module-level `codec.`/`ARM_DOF`/`ARMS` uses with `self._codec`, `self.dof`, `self.arms`; `__init__` gains `codec=yam_codec`; `embodiment`/`registry_prefix` become instance attributes set from `codec.EMBODIMENT`)
- Create: `src/cap_harness/g1_real/adapter.py`, `src/cap_harness/g1_real/registry.py`
- Modify: `src/cap_harness/api.py:1567-1577` (add `("set_hand_joints", "control")`, `("get_hand_state", "state")` to the extension tuple)
- Test: `tests/test_g1_real_adapter.py`

**Interfaces:**
- `g1_real/registry.py`: `G1_REAL_TASKS = (YamRealTaskMetadata("observe_station", "Observe the G1 station.", family="g1_real", scene_reset="none"), YamRealTaskMetadata("reach_home", ..., family="g1_real"), YamRealTaskMetadata("joint_motion", ..., family="g1_real"))`; `class G1RealTaskRegistry(YamRealTaskRegistry)` defaulting to `G1_REAL_TASKS`.
- `class G1RealAdapter(YamRealAdapter)`: `__init__(env, *, registry=None, task_context=None, allow_physical_motion=False, run_observer=None)` passes `codec=cap_harness.g1_real.codec`; overrides `get_planning_context()` → `RobotPlanningContext(embodiment="g1_real", model="g1_real", joint_names={side: config.arms[side].joint_names}, base_transforms=identity, end_effector_links={side: f"{side}_grasp"})`, `get_task_metadata()`, `get_controller_metadata()` (14-wide bounds, contract version `g1-sonic-v1-planner-stream`), and adds `set_hand_joints(arm: str, joints: list[float]) -> ExecutionResult` (interlock → `env.set_hand_joints`) and `get_hand_state() -> dict[str, list[float]]` (`{"left": [...7], "right": [...7]}`).
- Existing `YamRealAdapter` behaviour is unchanged for YAM (all `tests/test_yam_real_*` pass).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, pytest
from cap_harness.contracts import Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.g1_real.adapter import G1RealAdapter
from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.g1_real import codec
from cap_harness.yam_real.cameras import SyntheticCamera

@pytest.fixture
def adapter():
    cfg = load_g1_station_config("g1-lab")
    home = codec.upper_body_from_arms(cfg.waist_targets, cfg.arms["left"].home_joints, cfg.arms["right"].home_joints)
    env = G1Env(cfg, FakeBodyClient(tau_s=0.01, initial_q17=home), camera=SyntheticCamera()); env.engage()
    a = G1RealAdapter(env, allow_physical_motion=True); yield a; a.close()

def test_identity(adapter):
    assert adapter.embodiment == "g1_real" and adapter.dof == 7 and adapter.registry_prefix == "g1_real"
    ctx = adapter.get_planning_context(); assert ctx.model == "g1_real" and ctx.end_effector_links["right"] == "right_grasp"

def test_unauthorized_adapter_refuses_hands_and_motion():
    cfg = load_g1_station_config("g1-lab"); env = G1Env(cfg, FakeBodyClient(), camera=SyntheticCamera())
    a = G1RealAdapter(env)
    assert a.set_hand_joints("left", [0.0]*7).error.code == ErrorCode.SAFETY_INTERLOCK
    assert a.go_home().error.code == ErrorCode.SAFETY_INTERLOCK and env.body.commands == []

def test_go_home_reports_home_verified(adapter):
    result = adapter.go_home(); assert result.ok and result.diagnostics["home_verified"]

def test_trajectory_residual_fails_when_the_body_lags(adapter):
    adapter.native_env.body.tau_s = 5.0  # arm barely moves
    state = adapter.get_robot_state(); target = state.joint_positions["right"] + np.array([0.4, 0, 0, 0, 0, 0, 0])
    traj = Trajectory(joint_positions=np.linspace(state.joint_positions["right"], target, 25), dt_s=0.02,
        joint_names=list(adapter.native_env.config.arms["right"].joint_names), planner="test", collision_aware=False,
        expected_start=state, arm="right", embodiment="g1_real")
    result = adapter.execute_trajectory(traj)
    assert not result.ok and result.error.code == ErrorCode.EXECUTION_FAILED and "did not track" in result.error.message

def test_hand_tools_round_trip(adapter):
    q = [0.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    assert adapter.set_hand_joints("right", q).ok
    import time; time.sleep(0.1)
    np.testing.assert_allclose(adapter.get_hand_state()["right"], q, atol=0.05)

def test_hand_tools_are_registered_as_g1_extensions(adapter):
    from cap_harness.api import CapApi
    from cap_harness.registry import G1_REAL_PUBLIC_TOOL_NAMES, ToolRegistry
    names = {s.name for s in CapApi(adapter, default_camera="ego_view").register_tools(
        ToolRegistry(public_extension_allowlist=G1_REAL_PUBLIC_TOOL_NAMES)).all_specs(public_only=True)}
    assert {"g1_real.set_hand_joints", "g1_real.get_hand_state", "g1_real.get_controller_metadata"} <= names
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_g1_real_adapter.py -v` → ImportError.

- [ ] **Step 3: Refactor `YamRealAdapter` to codec injection.** Run `pytest tests/test_yam_real_adapter.py tests/test_yam_real_interlock.py tests/test_yam_real_plant_refusal.py tests/test_yam_real_recorded_action.py tests/test_yam_integration.py -q` → PASS before continuing.

- [ ] **Step 4: Implement `g1_real/registry.py`, `g1_real/adapter.py`; extend the `api.py` extension tuple.**

- [ ] **Step 5: Run `pytest tests -q` → PASS; commit**

```bash
git add src/cap_harness/yam_real/adapter.py src/cap_harness/g1_real/adapter.py src/cap_harness/g1_real/registry.py src/cap_harness/api.py tests/test_g1_real_adapter.py
git commit -m "feat(g1_real): adapter with hand extension tools via codec-injected YAM adapter"
```

---

### Task 8: Episode recorder width follows the plant

**Files:**
- Modify: `src/cap_harness/yam_real/recorder.py:41-55,150-160,185-210`
- Test: `tests/test_g1_real_recorder.py`

**Interfaces:**
- `YamEpisodeRecorder.start(env, meta)` reads `dof = getattr(env, "dof", ARM_DOF)` once and uses it for every `reshape` and channel width; the docstring table says `(N, dof)` / `(N, dof+1)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np, time
from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.yam_real.cameras import SyntheticCamera
from cap_harness.yam_real.recorder import YamEpisodeRecorder

def test_recorder_writes_seven_joint_arrays_for_g1(tmp_path):
    cfg = load_g1_station_config("g1-lab"); env = G1Env(cfg, FakeBodyClient(), camera=SyntheticCamera()); env.engage()
    rec = YamEpisodeRecorder(tmp_path / "episode", video_dir=tmp_path / "videos", fps=50, cameras=(), task="t")
    rec.start(env, meta={}); env.go_home(duration_s=0.3); time.sleep(0.2); rec.finalize(); env.close()
    assert np.load(tmp_path / "episode" / "left-joint_pos.npy").shape[1] == 7
    assert np.load(tmp_path / "episode" / "action-left-pos.npy").shape[1] == 8
```

- [ ] **Step 2: Run → FAIL** (`cannot reshape array of size 7 into shape (6,)`).
- [ ] **Step 3: Parameterize the recorder by `dof`.**
- [ ] **Step 4: Run `pytest tests/test_g1_real_recorder.py tests/test_yam_episode_recorder.py -v` → PASS; commit** `git commit -am "feat(recorder): joint width follows the plant"` (add the new test file).

---

### Task 9: `cap-harness run --benchmark g1_real` and example programs

**Files:**
- Modify: `src/cap_harness/run.py:86-87,116-124,151,171,248-258,327-345,355`
- Modify: `src/cap_harness/cli.py:144-152`
- Create: `src/cap_harness/g1_real/station.py`
- Create: `examples/g1_real/observe_station.py`, `examples/g1_real/go_home.py`, `examples/g1_real/joint_motion.py` (copies of the `examples/yam_real/` programs with seven-joint targets)
- Test: `tests/test_g1_real_run.py`

**Interfaces:**
- `g1_real/station.py`: `build_real_station(station="g1-lab", *, config_root=None, enable_camera=True) -> G1Env` (SonicZmqBodyClient + `ComposedCameraRgbd` from Task 12; until then `camera=None`), `build_fake_station(station="g1-lab", *, config_root=None, tau_s=0.02) -> G1Env` (FakeBodyClient + SyntheticCamera, engaged).
- `run.py`: `REAL_BENCHMARKS = {"yam_real", "g1_real"}`; the three `benchmark == "yam_real"` conditionals become `benchmark in REAL_BENCHMARKS` and the station/kinematics/default-camera lookups dispatch through a small table `{"yam_real": (build_real_station, YamKinematics, YamRealAdapter, YAM_REAL_PUBLIC_TOOL_NAMES, YamRealTaskRegistry), "g1_real": (...)}`; the IK provider is built with `YamKinematicsIKProvider(kin, dof=adapter.dof)`; `run_program` (the existing entry point, returning `RunOutcome(run_dir, program_ok, task_success, protocol_success, termination_reason)`) gains `station_backend: str = "real"` so tests can select `build_fake_station`.
- `cli.py`: `--benchmark` choices add `g1_real`; `--station` default stays `yam-example`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from cap_harness.run import run_program

def test_run_program_on_fake_g1_station(tmp_path):
    outcome = run_program(benchmark="g1_real", suite="joint_motion", task_id=0, seed=1,
        program_path=Path("examples/g1_real/joint_motion.py"), output_root=tmp_path,
        station="g1-lab", allow_motion=True, station_backend="fake")
    assert outcome.program_ok and outcome.termination_reason == "program_completed"
    assert any(tmp_path.rglob("right-joint_pos.npy"))
```

(Use the existing `tests/test_run.py` fixtures for monkeypatching SAM3/cuRobo clients if `run_episode` constructs them eagerly.)

- [ ] **Step 2: Run → FAIL** (`benchmark must be ...`).
- [ ] **Step 3: Implement the dispatch table, `station.py`, the examples, and the CLI choice.**
- [ ] **Step 4: Run `pytest tests/test_g1_real_run.py tests/test_run.py tests/test_cli.py -v` → PASS; commit** `git commit -m "feat(g1_real): run and CLI wiring with fake station backend"`.

---

### Task 10: System 2 server, MCP bridge, engage flag, camera-only Viser

**Files:**
- Modify: `src/cap_harness/agent/server.py:37-80,90-170`
- Modify: `src/cap_harness/agent/mcp.py:195-205`
- Modify: `src/cap_harness/agent/visualizer.py:35-60,77-105`
- Modify: `scripts/start_system2.sh` (no change needed; verify it passes args)
- Test: `tests/test_g1_real_system2.py`

**Interfaces:**
- `server.build_runtime(env, *, adapter_type, public_tool_names, family, allow_motion=False, episode_recorder=None, event_sink=None, tool_result_sink=None, **api_options)`; `build_yam_runtime` becomes a thin wrapper kept for `tests/test_system2_runtime.py`.
- `main` gains `--benchmark {yam_real,g1_real}` (default `yam_real`), `--engage` (G1 only; calls `env.engage()` after the runtime is built and only when `--allow-motion`), `--fake` (G1 analogue of `--sim`: `build_fake_station`). For `g1_real` the IK provider is `YamKinematicsIKProvider(G1Kinematics(env.config.model_xml), dof=7)`, `default_camera=env.config.camera.role`, `CuRoboProvider(control_period_s=1/50)`.
- `mcp.py`: `_DIRECT` and `_READ_ONLY` are built from `{"get_robot_state", "localize_object", "move_to_joints", "move_to_pose", "move_synchronized", "set_gripper"}` plus every registered public extension spec whose name ends with `.get_controller_metadata`, `.set_hand_joints`, `.get_hand_state`; the `*_hand_*` and `*get_controller_metadata` extensions of the active embodiment appear as `cap_g1_real__set_hand_joints` etc. (existing `.`→`__` rule).
- `visualizer.System2Visualizer(session, *, allow_motion, host, port, urdf_path: Path | None = YAM_STATION_URDF, engage: Callable[[], dict] | None = None)`: when `urdf_path is None` the robot model is not created and `_refresh_robot` returns False without hiding anything else; when `engage` is given, a button "Engage SONIC" calls it (disabled when `allow_motion` is False).

- [ ] **Step 1: Write the failing tests**

```python
from cap_harness.agent.server import build_runtime
from cap_harness.agent.mcp import CapMcpBridge
from cap_harness.g1_real.adapter import G1RealAdapter
from cap_harness.g1_real.station import build_fake_station
from cap_harness.registry import G1_REAL_PUBLIC_TOOL_NAMES

def test_g1_mcp_exposes_hand_tools_and_seven_joint_state():
    env = build_fake_station()
    adapter, runtime = build_runtime(env, adapter_type=G1RealAdapter, public_tool_names=G1_REAL_PUBLIC_TOOL_NAMES, family="g1_real", allow_motion=True)
    try:
        bridge = CapMcpBridge(runtime.session); names = {t.name for t in bridge.tool_specs()}
        assert {"cap_g1_real__set_hand_joints", "cap_g1_real__get_hand_state", "cap_g1_real__get_controller_metadata", "cap_move_to_joints"} <= names
        state = bridge.call("cap_get_robot_state", {}).structuredContent["result"]
        assert len(state["joint_positions"]["left"]) == 7
        hand = bridge.call("cap_g1_real__set_hand_joints", {"arm": "left", "joints": [0.0]*7})
        assert hand.structuredContent["ok"] is True
    finally:
        runtime.close()

def test_server_main_rejects_engage_without_motion(capsys):
    from cap_harness.agent.server import main
    assert main(["--benchmark", "g1_real", "--station", "g1-lab", "--fake", "--engage", "--no-viser", "--port", "0"]) == 2
```

- [ ] **Step 2: Run → FAIL** (`ImportError: build_runtime`).
- [ ] **Step 3: Implement the server/bridge/visualizer changes.** `main` returns 2 with a message when `--engage` is passed without `--allow-motion`.
- [ ] **Step 4: Run `pytest tests/test_g1_real_system2.py tests/test_system2_runtime.py tests/test_system2_visualizer.py tests/test_system2_launcher.py tests/test_mcp_pose_contract.py -v` → PASS; commit** `git commit -m "feat(system2): g1_real runtime, operator engage path, camera-only viser"`.

---

### Task 11: cuRobo `g1_real` model

**Files:**
- Create: `src/cap_harness/g1_real/description/curobo/g1_dual_arm.yml`
- Modify: `src/cap_harness/providers/curobo/service.py:104-200,267-271`
- Modify: `pyproject.toml:155-158` (package `g1_real/description/**/*.{xml,urdf,yml}`)
- Test: `tests/test_g1_real_curobo_config.py`

**Interfaces:**
- `_g1_real_robot_config() -> str` (lru_cached like `_yam_real_robot_config`): loads the v2-schema YAML, rewrites `urdf_path` to the absolute packaged `g1_dual_arm.urdf`, writes a temp file. YAML content: `robot_cfg.kinematics` with `format_version: 2.0`, `base_link: torso_link`, `tool_frames: [left_grasp, right_grasp]`, `cspace.joint_names` = the 14 arm joints (left 7 then right 7, Unitree order), `default_joint_position` = home, `collision_spheres` for `torso_link`, `head_link`, each arm link and each `*_wrist_yaw_link` (hand envelope: one 0.06 m sphere at the grasp offset), `self_collision_ignore` for adjacent links, `self_collision_buffer` 0.005.
- `_robot_config(request)` returns it when `request.model == "g1_real"`.

- [ ] **Step 1: Write the failing test**

```python
import yaml
from pathlib import Path
from cap_harness.providers.curobo import service

def test_g1_real_config_is_v2_torso_rooted_with_both_grasp_frames():
    path = Path(service._g1_real_robot_config()); cfg = yaml.safe_load(path.read_text())
    kin = cfg["robot_cfg"]["kinematics"]
    assert kin["format_version"] == 2.0 and kin["base_link"] == "torso_link"
    assert kin["tool_frames"] == ["left_grasp", "right_grasp"] and len(kin["cspace"]["joint_names"]) == 14
    assert Path(kin["urdf_path"]).is_file()
    assert service._g1_real_robot_config() == str(path)  # cached
```

- [ ] **Step 2: Run → FAIL** (`AttributeError`).
- [ ] **Step 3: Author the YAML and the service branch.**
- [ ] **Step 4: Run `pytest tests/test_g1_real_curobo_config.py tests/test_providers.py -v` → PASS.**
- [ ] **Step 5: Service smoke (needs `.venv-curobo`):** `scripts/supervise_services.sh --providers curobo --profile "$ASPIRE_PROFILE"` then, from `.venv-system2`, a 20-line script that builds `RobotPlanningContext` as `G1RealAdapter.get_planning_context()` does and calls `CuRoboProvider(base_url="http://127.0.0.1:8118", control_period_s=0.02).solve_ik(...)` for the right arm's own home pose (from `G1Kinematics.forward_kinematics`). Expected: `ok=True`. Then `plan_to_pose` to home + `[0.05, 0, 0.05]`: `ok=True`, `len(trajectory.joint_positions) <= 120`. Record both outputs in the commit message.
- [ ] **Step 6: Commit** `git commit -m "feat(curobo): g1_real dual-arm model rooted at torso_link"`.

---

### Task 12: RGB-D over the robot's camera server

**Files (fork, `~/work/GR00T-WholeBodyControl`, separate commit there):**
- Modify: `gear_sonic/camera/drivers/realsense.py:84-155`
- Modify: `gear_sonic/camera/sensor_server.py:106-146` (`ImageMessageSchema` gains `intrinsics: dict[str, dict] = field(default_factory=dict)`, serialized/deserialized pass-through)
- Modify: `gear_sonic/camera/composed_camera.py:460-472` (merge `intrinsics` like `timestamps`)
- Test: `gear_sonic/tests/test_realsense_depth_message.py`

**Files (ASPIRE):**
- Create: `src/cap_harness/g1_real/cameras.py`
- Modify: `src/cap_harness/g1_real/station.py` (`build_real_station` opens `ComposedCameraRgbd`)
- Test: `tests/test_g1_real_cameras.py`

**Interfaces:**
- Fork: with `enable_depth`, `RealSenseSensor.read()` applies `rs.align(rs.stream.color)` and returns `{"timestamps": {...}, "images": {mount: rgb, f"{mount}_depth": depth_uint16}, "intrinsics": {mount: {"fx", "fy", "ppx", "ppy", "width", "height", "depth_scale"}}}` (`depth_scale` from `device.first_depth_sensor().get_depth_scale()`).
- ASPIRE: `class ComposedCameraRgbd(host: str, port: int, *, role: str = "ego_view", depth_scale_m: float = 0.001)` with `read() -> CameraFrame | None` (`yam_real.cameras.CameraFrame`; `depth_m = depth_uint16 * message depth_scale (fallback depth_scale_m)`; `intrinsics` 3×3 from the message; `timestamp_s` = receive monotonic time) and `close()`; a background thread keeps the latest message (CONFLATE SUB, msgpack-numpy decode). When the message carries no depth, `read()` returns a frame whose `depth_m` is an all-NaN array and sets `.depth_missing = True`; `codec.observation` then builds the `CameraObservation` normally and `CapApi.localize_object` returns its existing typed failure for non-finite depth.

- [ ] **Step 1 (fork): write the failing schema test**

```python
import numpy as np
from gear_sonic.camera.sensor_server import ImageMessageSchema

def test_intrinsics_survive_serialization():
    msg = ImageMessageSchema(timestamps={"ego_view": 1.0}, images={"ego_view": np.zeros((4, 4, 3), np.uint8)},
                             intrinsics={"ego_view": {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0, "width": 640, "height": 480, "depth_scale": 0.001}})
    out = ImageMessageSchema.deserialize(msg.serialize())
    assert out.intrinsics["ego_view"]["fx"] == 600.0
```

- [ ] **Step 2 (fork): run → FAIL; implement the three edits; run `pytest gear_sonic/tests/test_realsense_depth_message.py -v` → PASS; commit in the fork:** `git commit -m "feat(camera): aligned RealSense depth with published intrinsics"`.

- [ ] **Step 3 (ASPIRE): write the failing client tests**

```python
import msgpack, msgpack_numpy as m, numpy as np, time, zmq
from cap_harness.g1_real.cameras import ComposedCameraRgbd

def _publish(port, payload):
    pub = zmq.Context.instance().socket(zmq.PUB); pub.bind(f"tcp://127.0.0.1:{port}"); time.sleep(0.3)
    pub.send(msgpack.packb(payload, default=m.encode, use_bin_type=True)); return pub

def test_client_builds_metric_frame():
    cam = ComposedCameraRgbd("127.0.0.1", 5730, role="ego_view")
    pub = _publish(5730, {"timestamps": {"ego_view": 1.0}, "images": {"ego_view": np.zeros((4, 4, 3), np.uint8), "ego_view_depth": np.full((4, 4), 1500, np.uint16)},
                          "intrinsics": {"ego_view": {"fx": 600.0, "fy": 600.0, "ppx": 2.0, "ppy": 2.0, "width": 4, "height": 4, "depth_scale": 0.001}}})
    time.sleep(0.3); frame = cam.read()
    assert frame.depth_m[0, 0] == 1.5 and frame.intrinsics[0, 0] == 600.0 and not cam.depth_missing
    cam.close(); pub.close()

def test_missing_depth_is_flagged_not_fatal():
    cam = ComposedCameraRgbd("127.0.0.1", 5731)
    pub = _publish(5731, {"timestamps": {"ego_view": 1.0}, "images": {"ego_view": np.zeros((4, 4, 3), np.uint8)}})
    time.sleep(0.3); frame = cam.read()
    assert frame is not None and np.isnan(frame.depth_m).all() and cam.depth_missing
    cam.close(); pub.close()
```

- [ ] **Step 4: Run → FAIL; implement `cameras.py`; wire `build_real_station`; run `pytest tests/test_g1_real_cameras.py -v` → PASS; commit** `git commit -m "feat(g1_real): RGB-D client for the robot camera server"`.

---

### Task 13: Operator runbook, agent skill delta, bootstrap

**Files:**
- Create: `docs/g1-real.md`
- Modify: `docs/system2.md` (one paragraph linking to `docs/g1-real.md` under "What the two launchers do")
- Modify: `.agents/skills/cap-tool-usage/SKILL.md` (append a "G1 station" section: seven joints per arm, base frame `torso`, `cap_set_gripper` is open/closed only, `cap_g1_real__set_hand_joints` for finger articulation, `cap_g1_real__get_hand_state`)
- Create: `examples/g1_real/bottle_pick/TASK_CONTEXT.md`
- Modify: `scripts/start_system2_agent.sh` (no code change; confirm it forwards task text)
- Test: `tests/test_g1_real_docs.py`

- [ ] **Step 1: Write the test** asserting `docs/g1-real.md` contains the strings `--input-type zmq_manager --output-type zmq --zmq-host 192.168.0.62 real`, `--realsense-enable-depth`, `--engage`, and `5557`; and that `cap-tool-usage/SKILL.md` mentions `cap_g1_real__set_hand_joints`.
- [ ] **Step 2: Write `docs/g1-real.md`** with the §8 table and start order from the spec, the exact PC2 commands (`tmux` camera session with `--realsense-enable-depth --port 5555`, `scripts/preflight.sh`, `deploy.sh`), the workstation commands (`supervise_services.sh --providers sam3,curobo`, `scripts/start_system2.sh --benchmark g1_real --station g1-lab --station-config-root <root> --allow-motion --engage --sam3-url ... --curobo-url ...`, `scripts/start_system2_agent.sh --allow-motion`), the preflight gates (feedback < 100 ms, hands open, home residual ≤ 0.03 rad, camera age < 500 ms with depth, `/healthz` ok, `planner_reference_active == 1`), the stop procedure (`O` on PC2 / PICO A+B+X+Y, then MCP), and the port-5556 contention note.
- [ ] **Step 3: Write the skill delta and `TASK_CONTEXT.md`** (bottle pick-and-place: localize with `cap_localize_object` in `ego_view`, approach from above in `torso` frame, close with `cap_g1_real__set_hand_joints` using the profile `hand_closed` as the starting closure, verify lift from the returned image and wrist joint load, place, release, report; do not call Home while holding).
- [ ] **Step 4: Run `pytest tests/test_g1_real_docs.py -v` → PASS; commit** `git commit -m "docs(g1_real): operator runbook, skill delta, bottle task context"`.

---

### Task 14: Sim2sim integration against SONIC's own MuJoCo loop

**Files:** none created; this task is a verification gate recorded in `docs/g1-real.md` under "Off-robot check".

- [ ] **Step 1: Start the SONIC sim** (fork, two terminals): `source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py`; then from `gear_sonic_deploy/`: `./deploy.sh --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 sim` and wait for `Init done`. Press `9` in the MuJoCo window to drop the robot.
- [ ] **Step 2: Local profile:** copy `g1-lab/station.yaml` to `/tmp/g1-sim/g1-sim/station.yaml`, set `station: g1-sim`, `feedback_host: 127.0.0.1`, `camera.host: 127.0.0.1`, keep `command_bind: tcp://*:5556`.
- [ ] **Step 3: Run** `.venv-system2/bin/cap-harness run --benchmark g1_real --station g1-sim --station-config-root /tmp/g1-sim --suite joint_motion --task-id 0 --program examples/g1_real/joint_motion.py --allow-motion --output-root outputs/g1-sim`. Expected: the sim robot engages, homes, performs the bounded motion; `outputs/g1-sim/.../episode/raw/right-joint_pos.npy` has 7 columns; no `feedback stale` in the log.
- [ ] **Step 4: Run System 2:** `scripts/start_system2.sh --benchmark g1_real --station g1-sim --station-config-root /tmp/g1-sim --allow-motion --engage --curobo-url http://127.0.0.1:8118` (SAM3 omitted; camera absent → `observe` reports the camera missing). From another terminal call MCP `cap_program_go_home` and then `cap_move_to_pose` for the right arm to home + 5 cm forward with `strategy={"ik_solver":"mink","trajectory_planner":"curobo"}` (via `scripts/start_system2_agent.sh --allow-motion exec "..."` or a 15-line MCP client script). Expected: `home_verified: true`; move result `ok: true`, `joint_residual_rad.right < 0.10`.
- [ ] **Step 5: Record the exact commands and observed values in `docs/g1-real.md`; commit** `git commit -m "docs(g1_real): sim2sim verification record"`.

---

### Task 15: On-robot tracking acceptance (spec §9)

**Files:**
- Create: `scripts/g1_tracking_acceptance.py`
- Test: `tests/test_g1_tracking_acceptance.py` (runs the analysis functions on synthetic logs; the script's robot path is operator-run)

**Interfaces:**
- `scripts/g1_tracking_acceptance.py --station g1-lab --station-config-root <root> --phase {hold,segments,loaded}` builds `build_real_station`, engages, and: `hold` streams the measured pose for 5 s and reports `max |body_q − target|` per arm (pass ≤ 0.05 rad); `segments` runs ten 0.3 m straight-line `move_to_pose` segments at 0.1 m/s through `CapApi.move_to_pose` and reports final EE position/orientation error from `G1Kinematics.forward_kinematics` of measured joints (pass ≤ 1.5 cm and ≤ 5°, no residual failure); `loaded` repeats `hold` after the operator confirms a 300 g object is held. Pure functions `hold_error(targets: np.ndarray, measured: np.ndarray) -> dict[str, float]` and `segment_error(target_pose, measured_pose) -> tuple[float, float]` are unit-tested; the script prints `PASS`/`FAIL` per phase and writes `outputs/g1-acceptance/<timestamp>.json`.

- [ ] **Step 1: Write the unit tests for the two pure functions** (hold error of a 0.04 rad offset is 0.04 and passes; 0.06 fails; a 2 cm position miss fails `segment_error`).
- [ ] **Step 2: Run → FAIL; implement; run → PASS; commit** `git commit -m "feat(g1_real): on-robot tracking acceptance script"`.
- [ ] **Step 3 (operator, real robot): run the three phases after the runbook preflight.** Record the JSON in `docs/g1-real.md`. If any phase fails: adjust arm `STIFFNESS_*`/`DAMPING_*` in the fork's `policy_parameters.hpp`, rebuild on PC2 (`just build`), rerun once. If still failing, open the decoupled-WBC fallback as a new plan (spec §9); the body-client seam from Task 4 is the only thing that changes.
