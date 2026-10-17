# ASPIRE on Unitree G1 with GEAR-SONIC: design

Status: draft for operator approval. Nothing below is built yet.
Author: Claude (Fable 5.1), from a read of this repository, of upstream
`NVlabs/GR00T-WholeBodyControl` at `45c03fa` (2026-10-13), and of the user's
fork at `~/work/GR00T-WholeBodyControl` (`vr3pt-cleanup-minimal-official`,
HEAD `7c6d710`, 105 commits ahead of upstream plus uncommitted edits).

## 1. Intent

Run ASPIRE System 2 (coding agent + CAP MCP tools + SAM3 + cuRobo) on the
user's real Unitree G1 instead of the bimanual YAM station. GEAR-SONIC v1.1
keeps the robot balanced; ASPIRE drives both 7-DoF arms and the Dex3 hands
through the same Cartesian / joint / gripper tool contract it uses on YAM, so
the agent, skills, recipes and recording pipeline carry over unchanged.

The user's two options are both kept open behind one seam:

- **SONIC planner streaming** (`deploy.sh --input-type zmq_manager`): the
  planner holds the legs in `idle`; ASPIRE streams 17 upper-body joint
  targets on the ZMQ `planner` topic at 50 Hz. **Primary path.** It is the
  path the user's GR00T N1.7 VLA evaluation already uses on this robot.
- **Decoupled WBC** (`decoupled_wbc/`): RL lower body, open-loop interpolated
  PD targets for the upper body over ROS 2. **Fallback only**, chosen if the
  §9 tracking test fails. The user's current deployment does not run it.

## 2. Goals, non-goals, anti-goals

Goals

- `cap_move_to_pose`, `cap_move_synchronized`, `cap_move_to_joints`,
  `cap_set_gripper`, `cap_localize_object`, `observe`, `cap_program_go_home`
  behave on the G1 exactly as the existing skills describe them.
- Same evidence model: typed results, joint residual check after every
  trajectory, images on every terminal result, motion interlock, episode
  recording with a versioned control contract.
- Off-robot test path that exercises the real ZMQ wire format against
  SONIC's own MuJoCo sim2sim loop. No humanoid physics inside ASPIRE.
- Coexist with the user's existing tooling: same deploy binary, same
  feedback telemetry, same camera server, same Dex3 hand calibration.

Non-goals (v1)

- Walking, squatting or any locomotion during a task (planner mode 0).
- Dexterous in-hand manipulation; hands are a scalar gripper (0 closed,
  1 open).
- Inspire hands. The user's Inspire track is a separate, unmerged DDS path
  (`rt/inspire_hand/ctrl/*`); the hand driver seam in D5 leaves room for it.
- Replacing SONIC's or the user's safety procedures (`preflight.sh`,
  keyboard `O`, PICO A+B+X+Y).

Anti-goals

- A second code path to the motors besides `execute_action_batch`.
- Reporting success when the arm did not arrive.
- Forking the user's bridge/camera code into ASPIRE. ASPIRE vendors only
  the two small upstream wire modules and imports nothing XR-specific.

## 3. Facts the design rests on

ASPIRE (`src/cap_harness/`)

- Real-robot stack is `yam_real/`: `env.py` (one actuation chokepoint
  `execute_action_batch`), `control/controller.py` (level-1 wall-clock
  resampler streaming joint targets at `command_stream_hz`), `adapter.py`
  (interlock, 0.10 rad residual check, distance-scaled homing), `codec.py`,
  `kinematics.py` (mink over a MuJoCo model with `{side}_grasp_site`),
  `station.py`, `config.py` (profile + hashed calibration bundle).
- Joint dimension is enforced per embodiment in
  `contracts.JOINT_DIMENSIONS = {behavior: 7, libero: 7, robosuite: 7, yam_real: 6}`;
  the cuRobo service keys its model on the same literal
  (`model: Literal["panda", "dual_panda", "yam_real"]`).
- `yam_real` is referenced in a countable set of shared places: `run.py`,
  `cli.py`, `agent/server.py`, `agent/mcp.py` (`yam_real.get_controller_metadata`),
  `agent/visualizer.py` (URDF path), `registry.py` (allowlist, prefixes).
  The 6-DoF assumption lives only inside `yam_real/`.
- The cuRobo submodule is not checked out in this clone; upstream cuRobo
  ships no G1 model.

SONIC deploy (upstream, unchanged by the user in these respects)

- `zmq_manager` subscribes to `command`, `planner`, `pose` topics on one SUB
  socket. The publisher **binds** (PUB) and deploy connects with
  `--zmq-host <publisher-ip>`. Wire format:
  `[topic][1280-byte NUL-padded JSON header][little-endian payload]`;
  builders in `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`.
- `planner` message: `mode` i32, `movement` f32[3], `facing` f32[3],
  `speed`, `height`, optional `upper_body_position` f32[17],
  `upper_body_velocity` f32[17], `left/right_hand_joints` f32[7]. With
  `upper_body_position` present, the deploy loop overwrites the 17
  upper-body entries of every motion-reference frame before the policy sees
  them. The policy, not a PD loop, tracks the arms.
- 17 upper-body order: `[waist_yaw, waist_roll, waist_pitch, L/R shoulder_pitch,
  L/R shoulder_roll, L/R shoulder_yaw, L/R elbow, L/R wrist_roll,
  L/R wrist_pitch, L/R wrist_yaw]`, left/right interleaved.
- Planner messages older than 1 s reset locomotion to IDLE **and clear the
  upper-body / hand targets**: the arms return to the planner's idle pose.
  Switching `planner` on/off triggers a safety reset.
- Control loop 50 Hz, planner ~10 Hz, wxyz quaternions, Z-up, X forward.
- Feedback `--output-type zmq --zmq-out-port 5557`, topic `g1_debug`,
  msgpack map every tick: `body_q[29]`, `body_dq[29]` in **MuJoCo order**,
  `base_quat`, `left/right_hand_q[7]`, `body_q_target[29]`, `last_action[29]`.
- Hands: the C++ stack drives Dex3 only; `--disable-dex3-hands` exists in
  the user's fork.

The user's stack (`~/work/GR00T-WholeBodyControl`)

- Topology in use: workstation `192.168.0.62` binds the SONIC command PUB on
  **5556** and runs PICO manager / VLA client / data exporter. Robot PC2
  (`unitree@192.168.0.223`, onboard Orin) runs `deploy.sh` (connects to
  `.62:5556`), publishes feedback on **5557** and the composed camera server
  on **5555** (`gear_sonic.camera.composed_camera --ego-view-camera realsense`).
  GR00T PolicyServer runs on a separate GPU host `192.168.75.173`.
  Only one process can bind 5556 on the workstation: ASPIRE replaces the
  PICO manager / XR bridge for the duration of a session.
- Deploy fork adds a VR_3PT jump filter (irrelevant to joint streaming),
  `--disable-dex3-hands`, `preflight.sh` before real runs, and feedback keys
  `reference_heading_quat` / `planner_reference_active` consumed by
  `gear_sonic/utils/inference/planner_heading_frame.py`.
- Reusable Python already proven on this robot:
  `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py` (wire builders, upstream),
  `gear_sonic/utils/data_collection/zmq_state_subscriber.py` (`ZMQStateSubscriber`,
  conflated SUB on 5557, upstream),
  `gear_sonic/utils/teleop/bridge_planner_publisher.py` (`PlannerCommand`,
  hand step limiter 0.25 rad/msg),
  `gear_sonic/utils/teleop/xr_upperbody_bridge.py` (`UpperBodyFilter`
  0.03 rad/msg joint step limit, `G1_UPPER_BODY_JOINT_INDICES`
  `[12,13,14,15,22,16,23,17,24,18,25,19,26,20,27,21,28]` = MuJoCo-order
  indices of the 17 upper-body joints, `G1_STANDING_UPPER_BODY` idle pose
  `[0,0,0, 0.2,0.2, 0.2,-0.2, 0,0, 0.6,0.6, 0×6]`, `anchor_frame_planner_heading`),
  `gear_sonic/utils/inference/standing_reset.py` (`StandingReset`: feedback-paced
  ramp to standing, ≤0.5 rad/s, ≤0.15 rad lead over measured).
- Camera: `gear_sonic/camera/drivers/realsense.py` publishes JPEG colour at
  15 Hz and, with `realsense_enable_depth`, raw uint16 depth, via msgpack on
  the composed-camera PUB. It does **not** align depth to colour and does
  **not** publish intrinsics.
- No G1 forward kinematics exists in the inference path; proprio is joint
  values only.
- First task family on this robot: tabletop pick-and-place (bottle, cube,
  trash) at standing height. Matches ASPIRE's cube/bottle recipes.

## 4. Constraints

- The pelvis sways while SONIC balances and there is no base-position
  feedback, so nothing may be expressed in a world frame.
- ASPIRE components stay on the workstation (and optionally the GPU host
  for SAM3/cuRobo); the only robot-side change is a small camera-driver
  patch in the user's repo.
- G1 reach is ~0.55 m from the shoulder; the table must sit in that band.
- ASPIRE must not keep SONIC's `planner` topic silent for more than ~1 s
  while a task is active.

## 5. Architecture (decided)

```
agent ── MCP ──> CAP session ──> G1RealAdapter ──> G1Env.execute_action_batch
                                                     │ level-1 @50 Hz: 17 joints + 2×7 hand joints
                                                     ▼
                              G1BodyClient (port) ── SonicZmqBodyClient ── PUB :5556 (workstation) ──> deploy.sh zmq_manager @ PC2
                                                     ▲                     SUB  PC2:5557 g1_debug <── body_q / hand_q / heading
                                                     └── (fallback) DecoupledWbcRosClient ── ROS 2 ControlPolicy/upper_body_pose
RGB-D: PC2 composed_camera (patched: aligned depth + intrinsics) ── SUB PC2:5555 ──> ASPIRE ComposedCameraRgbd
IK: mink over MuJoCo G1 torso+arms+Dex3 model     Planning: cuRobo service model "g1_real" (14 DoF, fixed base at torso_link)
```

D1. **Embodiment `g1_real`, 7 joints per arm**, Unitree order per side
    `[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll,
    wrist_pitch, wrist_yaw]`. Arms `left`/`right`; `arm_aliases` in the profile.

D2. **Base frame `torso` (`torso_link`).** Waist targets are a fixed vector
    (default zeros) sent in every tick, so torso→shoulder is rigid during a
    task and the head camera extrinsic is constant. Pelvis sway cancels.

D3. **`G1BodyClient` port with `SonicZmqBodyClient` first.** Surface:
    `get_observations() -> {left_joint_pos[7], right_joint_pos[7], waist[3],
    left_hand_q[7], right_hand_q[7], base_quat[4], heading_yaw, timestamp_s}`
    and `command_upper_body(q17, dq17, left_hand7, right_hand7)`.
    Implementation vendors two upstream files verbatim with provenance
    headers (`zmq_planner_sender.py`, `zmq_state_subscriber.py`) and copies
    the four constants above from the user's bridge. It does not import the
    XR bridge. Order conversion MuJoCo→SONIC-17 uses
    `G1_UPPER_BODY_JOINT_INDICES`; `facing` is anchored to the measured
    heading every tick (`anchor_frame_planner_heading` logic, ~10 lines)
    so an idle planner never turns the robot.

D4. **Level-1 streams at 50 Hz** through the chokepoint with the YAM
    controller's shape (wall-clock resampling, start interpolation, settle
    hold). A per-message joint step clamp of 0.03 rad mirrors the user's
    `UpperBodyFilter`; ASPIRE's own trajectories are far slower. Between
    batches a **keep-alive thread re-sends the last target at 50 Hz** so the
    planner timeout cannot fire. Feedback older than 0.5 s refuses the next
    command with `EXECUTION_FAILED`.

D5. **Dex3 hands are commanded joint-by-joint by the agent (user decision).**
    No hand primitives and no pinch calibration. Two embodiment extension
    tools, registered under the `g1_real.` prefix and allowlisted for MCP:
    `g1_real.set_hand_joints(arm, joints)` (7 radians, clamped to the Dex3
    limits in the profile, step-limited at 0.25 rad per message like the
    user's hand limiter) and `g1_real.get_hand_state()` (measured 7-vectors
    per hand from `left/right_hand_q`). The shared `set_gripper(p)` stays
    only as a trivial convenience: linear between the profile's `hand_open`
    (zeros) and `hand_closed` (full-close pose, same values as SONIC's own
    test script) with `p` reported by projection onto that segment. Hand
    targets ride in the same planner message as the arms (one writer).
    An Inspire driver later implements the same two-method hand seam.
    v1 is right-arm primary with the left hand held at `hand_open`
    (user confirmed). ASPIRE's recording stays the ASPIRE action
    encoding (7 joints + scalar per arm); feeding the GR00T dataset path is
    not a goal (user decision).

D6. **IK is mink** over a MuJoCo model of torso + both arms + Dex3 derived
    from the user's `gear_sonic_deploy/g1/g1_29dof_with_hand.xml` (legs
    removed, body fixed at `torso_link`, `left/right_grasp_site` at the
    fingertip contact point). `YamKinematics` is generalized (joint slices and site names
    from config), not copied.

D7. **cuRobo model `g1_real`**: 14-DoF dual-arm config generated from
    upstream `gear_sonic/data/robots/g1/g1_dual_arm.urdf` (already rooted at
    the waist with waist joints fixed), with collision spheres for torso,
    head, arms and hands plus a self-collision ignore map. Registered in
    `providers/curobo/service.py` next to `yam_real`.

D8. **Camera: patch the user's RealSense driver, not a new publisher.**
    Add `rs.align(rs.stream.color)` when depth is enabled and publish
    `intrinsics` (fx, fy, cx, cy, width, height) once per message under a
    new key; ~15 lines in `gear_sonic/camera/drivers/realsense.py`, behind
    the existing `realsense_enable_depth` flag so the VLA path is untouched.
    ASPIRE adds `ComposedCameraRgbd` (msgpack SUB on PC2:**5555**, user
    confirmed; same frame object as `yam_real/cameras.py`). The user
    approved patching the driver in the fork.
    **Extrinsic `torso_from_camera`:** none exists today (user). The YAM
    calibration tool needs portal arm clients and hand-guided ChArUco
    sweeps, neither of which exists on the G1, so v1 does not port it. v1
    uses the URDF `d435_joint` pose as the nominal extrinsic plus a one-shot
    translation refinement: localize one object at a tape-measured torso
    position and store the offset in the bundle. The Phase 5 gate is
    therefore 2 cm. A proper hand-eye tool (board on the Dex3, sweep via the
    planner stream, FK from the torso model) is deferred.

D9. **Off-robot test = SONIC sim2sim** (`run_sim_loop.py` + `deploy.sh
    --input-type zmq_manager sim`, same two ports). Unit tests use a fake
    body client with first-order lag so the residual check can fail.

D10. **Home = `G1_STANDING_UPPER_BODY` + open hands**, ramped with the YAM
     adapter's distance-scaled homing at ≤0.35 rad/s, which is inside the
     user's `StandingReset` envelope (≤0.5 rad/s, ≤0.15 rad lead). `reset()`
     homes. `home_verified` uses the 0.03 rad diagnostic tolerance.

D11. **Start is an operator-only ASPIRE action; stop stays with SONIC.**
     In `zmq_manager` mode the deploy binary has no `]` key: control starts
     only from a ZMQ `command{start=1, planner=1}` message
     (`zmq_manager.hpp` lines 245-250), and only one process can bind 5556.
     So ASPIRE's server exposes an **operator-only** engage path (a
     `--engage` CLI flag on `start_system2.sh` and a Viser button, never an
     agent tool) that: waits for fresh 5557 feedback, publishes a
     heading-anchored hold planner message at the measured upper-body pose
     for ≥ 200 ms (the manager requires a planner message < 100 ms old at
     mode entry), then sends `command{start=1, planner=1}`. This mirrors the
     user's `build_start_control_messages` in the XR bridge and avoids the
     ≤ 1 s idle snap-back a PUB handover would cause. The keep-alive takes
     over immediately after. ASPIRE never sends `stop`; keyboard `O` on PC2
     and PICO A+B+X+Y remain the e-stops, and `allow_physical_motion` gates
     every publish including engage.

D12. **Hosts.** Workstation `.62`: CAP MCP, Viser, ASPIRE PUB 5556.
     PC2 `.223`: SONIC deploy, feedback 5557, camera 5555. SAM3 and cuRobo:
     workstation `.62` if it has VRAM headroom, otherwise the
     `192.168.75.173` or `.174` GPU host (user). Both are plain HTTP
     services; profile fields `sam3_url`, `curobo_url`.

## 6. What stays unchanged

Everything the agent sees: AGENTS.md rules, `cap-tool-usage`, the pose
contract, `strategy={"ik_solver":"mink","trajectory_planner":"curobo"}`,
job/observe semantics, recording layout. Visible differences: seven joint
values per arm, base frame `torso`, hand state as a scalar.

## 7. Implementation plan

Phase 1. Contracts and plumbing (ASPIRE only)
- `JOINT_DIMENSIONS["g1_real"] = 7`; `registry.py` prefix + public tool
  names; `run.py` / `cli.py` accept `--benchmark g1_real`; `agent/server.py`
  builds the runtime from an embodiment switch; `agent/mcp.py` derives
  `<prefix>.get_controller_metadata` from the adapter; visualizer URDF path
  from the profile.
- Check: current suite green; new structural test asserts `g1_real`
  resolves everywhere `yam_real` does.

Phase 2. `cap_harness/g1_real/`
- `config.py` (profile: ZMQ hosts/ports/topics, joint limits per arm, waist
  targets, home, hand open/closed poses, rates, camera endpoint, calibration
  bundle, `sam3_url`/`curobo_url`), `codec.py` (MuJoCo ↔ SONIC-17 ↔ per-arm
  Unitree order; quaternion order), `sonic/` (two vendored upstream files),
  `body_client.py` (`G1BodyClient`, `SonicZmqBodyClient`, `FakeBodyClient`),
  `env.py` (chokepoint + keep-alive), `control/controller.py` (50 Hz
  level-1 with step clamp), `kinematics.py`, `adapter.py` (incl. the
  `g1_real.set_hand_joints` / `get_hand_state` extensions), `registry.py`,
  `station.py`, `cameras.py` (`ComposedCameraRgbd`); `agent/server.py`
  gains the operator-only engage path (D11).
- Check: `tests/test_g1_real_*`: order round-trips against the
  `G1_ISAACLAB_TO_MUJOCO_DOF` table, interlock, residual failure with the
  lagging fake, keep-alive cadence, wire bytes decoded by a Python port of
  the C++ header parser, gripper scalar ↔ pose round-trip.

Phase 3. cuRobo `g1_real` model + segmentation spheres; service literal;
`cap-harness doctor --providers curobo`.
- Check: IK solves the measured pose; a plan between two chest-front poses
  is collision-aware and ≤120 waypoints at 1/50 s.

Phase 4. Sim2sim integration on the workstation
- `run_sim_loop.py` + `deploy.sh --input-type zmq_manager sim` (user's fork);
  `cap-harness run --benchmark g1_real --program examples/g1_real/joint_motion.py --allow-motion`;
  then System 2 server against the same sim.
- Check: `cap_program_go_home` → `home_verified`; `cap_move_to_pose` to a
  chest-front target with residual < 0.10 rad.

Phase 5. Perception
- Patch `realsense.py` in the user's fork (aligned depth + intrinsics);
  restart the PC2 camera tmux session with `--realsense-enable-depth`;
  calibration bundle for `torso_from_camera`.
- Check: `cap_localize_object` on a bottle returns a torso-frame centroid
  within 1 cm of a tape measurement.

Phase 6. Operator and agent material
- `docs/g1-real.md` runbook, station profile `g1-lab` with the real
  hosts/ports, `examples/g1_real/{observe_station,go_home,joint_motion}.py`,
  a `TASK_CONTEXT.md` for bottle pick-and-place, one-paragraph delta in
  `cap-tool-usage` (7 joints, hand scalar, base `torso`).

Phase 7. On-robot acceptance (§9), then the first agent task.

## 8. Deployment plan

| Service | Host | Port | Direction |
| --- | --- | --- | --- |
| ASPIRE SONIC command PUB (`command`,`planner`) | workstation .62 | 5556 | deploy SUB connects |
| SONIC feedback PUB `g1_debug` | PC2 .223 | 5557 | ASPIRE SUB |
| Composed camera PUB (RGB + aligned depth) | PC2 .223 | 5555 | ASPIRE SUB |
| SAM3 | .62 or .75.173 | 8114 | ASPIRE HTTP |
| cuRobo | .62 or .75.173 | 8118 | ASPIRE HTTP |
| CAP MCP / Viser | .62 | 8222 / 8080 | agent / operator |

Start order: (1) stop any PICO manager / XR bridge on `.62` (5556 must be
free); (2) on PC2: camera session with depth enabled; `deploy.sh
--input-type zmq_manager --output-type zmq --zmq-host 192.168.0.62 real`
after `preflight.sh`, wait for `Init done`; (3) `supervise_services.sh
--providers sam3,curobo`; (4) `start_system2.sh --benchmark g1_real
--station g1-lab --allow-motion ...` (binds 5556; no messages yet);
(5) operator runs the engage path (D11) and confirms the robot is standing
in PLANNER idle with fresh feedback; the keep-alive is now live;
(6) `start_system2_agent.sh --allow-motion`: homes once, reports READY.

Stop: place or release held objects, then `O` on PC2 (or PICO A+B+X+Y),
then stop MCP. Keyboard `O` remains the e-stop; ASPIRE shutdown is not.

Preflight gates before any agent task: feedback age < 100 ms, both hands at
open pose, home residual ≤ 0.03 rad, camera age < 500 ms with depth present,
SAM3 and cuRobo `/healthz` ok, `planner_reference_active == 1`.

## 9. Acceptance test that decides the backend

On the real robot, SONIC path:

1. Hold a fixed upper-body target 5 s; compare `body_q` to target.
   Pass: max per-joint error ≤ 0.05 rad on both arms.
2. Ten 0.3 m straight-line `move_to_pose` segments at 0.1 m/s; EE error from
   FK of measured joints. Pass: ≤ 1.5 cm, ≤ 5°, no 0.10 rad residual failure.
3. Hold a 300 g bottle 30 s; repeat test 1.

Tuning knobs are narrower than upstream docs suggest: the user's fork
removed `motor_gain_scaling.*`, the deploy binary has no `--motor-kp-scale`
flag, arm kp/kd are compile-time constants in `policy_parameters.hpp`, and
no `low_latency` checkpoint is on disk (only `release` and `sonic_v1_1`).
So "one round of tuning" means editing the arm kp/kd constants and
rebuilding on PC2, which the user confirmed is in scope. If that round
fails, implement `DecoupledWbcRosClient` and rerun; nothing else changes.

## 10. User decisions (2026-10-17)

1. Hands: no hand primitives or pinch calibration; the agent commands Dex3
   joints directly (D5).
2. Camera: PC2 port 5555; patching the fork's RealSense driver is approved.
3. A deploy rebuild on PC2 for arm kp/kd tuning is in scope (§9).
4. No torso→D435 extrinsic exists; v1 uses the URDF pose plus one-point
   refinement (D8).
5. Recording stays the original ASPIRE encoding; GR00T dataset
   compatibility is not a goal.
6. v1 is right-arm primary with the left hand open; bimanual later.
7. SAM3 and cuRobo on `.62` if VRAM allows, else the `.75.173`/`.174` GPU
   host (D12).
8. Table at ~0.75–0.85 m, bottle pick-and-place first.

## 11. Left to the builder

- Exact derivation of the MuJoCo torso+arms+Dex3 model and grasp-site pose.
- Sphere radii and self-collision ignore map for the cuRobo model.
- `upper_body_velocity`: finite differences of the streamed targets
  (start there; zeros are the alternative).
- Keep-alive thread ownership (env vs body client).
- Whether the calibration bundle format is reused verbatim or gets a
  `torso` variant.

## 12. Risks

- **Policy-tracked arms.** Precision is unproven for ASPIRE-style
  manipulation; §9 decides.
- **Planner timeout snap-back.** Losing the keep-alive returns the arms to
  idle within 1 s; mitigated by D4, the staleness refusal and the gates.
- **Port 5556 contention** with the user's PICO manager; the runbook makes
  stopping it a step.
- **Reach.** Many YAM recipes assume a larger workspace; verify placement
  with `cap_localize_object` before planning.
- **Dex3 closure is the agent's problem.** With joint-level hand control
  the agent must choose finger poses per object; the first recipe should
  record a few working closures as text guidance, not code.
