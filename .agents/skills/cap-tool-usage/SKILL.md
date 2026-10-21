---
name: cap-tool-usage
description: Localize objects and compose low-level CAP MCP actions using observations, Cartesian or synchronized motion, joint targets, and single-arm gripper control.
---

# CAP Tool Usage

Follow [AGENTS.md](../../../AGENTS.md). Use the native `mcp__cap__*` interface
attached to the ASPIRE CAP session. Keep that session as the only environment and
camera owner; do not create another robot or camera runtime during a physical task.

For a plainly short-horizon task, execute the shortest dependency-ordered MCP sequence immediately.
Skip source inspection, a full planning report, and redundant checks.

## ASPIRE action contract

Native MCP schemas are authoritative. The low-level action tools are:

| Tool | Target |
| --- | --- |
| `cap_move_to_pose` | `target_pose`, with `arm="left"` or `"right"` |
| `cap_move_synchronized` | `targets={"left": pose, "right": pose}` for jointly planned motion |
| `cap_move_to_joints` | `target` containing six joint angles in radians, with `arm` |
| `cap_set_gripper` | `position` from 0 closed to 1 open, with `arm` |

An absolute pose includes `position: [x, y, z]` in metres, `frame`, and exactly one
of `rpy_deg: [roll, pitch, yaw]` or `quaternion_wxyz: [w, x, y, z]`. RPY uses degrees
and fixed-axis XYZ: `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)` in the declared frame.
Returned poses use WXYZ quaternions. Read measured arm poses with `cap_get_robot_state`.

For Cartesian or synchronized pose motion with cuRobo configured, select
`strategy={"ik_solver":"mink","trajectory_planner":"curobo"}`. Never disable
collision checking to force a failed plan through. Do not use direct joint motion
as implicit IK or to bypass a failed collision-aware Cartesian plan.

For a layer turn, derive successive absolute pose targets from the measured cube
axis and EEF pose. An offset EEF needs both its position and orientation rotated
about that axis. Verify contact and actual sticker motion from returned evidence.
Use separate `cap_set_gripper` calls for intended grip changes. Prefer synchronized
motion only when both targets are grounded and neither depends on the other's feedback.

## Object-approach tool chain

Use the MCP schemas as authoritative. The expected discoverable tools are:

`cap_localize_object` → `cap_move_to_pose` → optional `observe`

`cap_localize_object` performs one-shot SAM3 plus shared RGB-D localization. `cap_move_to_pose`
performs its own IK and motion planning. Observation and localization are read-only; only action
tools actuate the robot.

Read current arm state when a target depends on the measured EEF pose. Form a
full pose from grounded object geometry and the intended grasp orientation; an
object's estimated orientation is not automatically a suitable gripper orientation.

Use the runtime's available camera names; their names and count are not fixed.
`observe` and `get_job` accept an optional `cameras` list. World-frame localization
requires a calibrated RGB-D view. Check the installed pads before applying the
cube guide's soft UMI contact guidance.

## Pose evidence

1. Discover the current tool schemas from CAP MCP; do not infer arguments from this summary.
2. When object-relative metric geometry is needed, call `cap_localize_object` in a calibrated RGB-D view. Reuse that
   result when its identity, frame, geometry, and diagnostics are valid. Repeat only after concrete
   invalid or contradictory evidence, or after motion changed the scene; do not sample twice merely
   to estimate stability. Reject unreliable estimates, not the entire task: another view or a bounded
   correction grounded in measured EEF state and visible alignment may remain feasible.
3. Refresh grounding that motion invalidated. A new localization call is needed when the next action
   depends on object geometry that fresh returned evidence does not establish.

## Shared rules

- A job receipt is not a terminal result; handle execution and completion as described in AGENTS.md.
  Process terminal status, errors, state, and images before the next dependent action.
- Use successful measured postconditions as evidence of the metric effects they actually verify.
  Continue directly when they resolve the next decision. Call `observe` when a required relation is absent,
  stale, ambiguous, or contradicted; never use it as a routine acknowledgement of successful motion.
- Action feedback includes images by default. Reuse fresh returned evidence; numeric motion
  success alone does not verify grasp, placement, or task completion.
- When IK succeeds and motion executes, large measured tracking errors near an object, the table,
  or the other gripper can indicate contact. Use returned views to distinguish obstruction from
  free-space tracking error; do not push farther or loosen tolerance to get past suspected contact.
  An IK rejection by itself is not evidence of contact.
- For a supporting arm, retain the intended hold reference across calls; do not silently replace it
  with each drifted measurement. Plan corrections from the achieved state with contact and clearance
  checked. Distinguish a changed reference from failure to track an unchanged one before attributing
  drift to hardware or declaring the task blocked.
- Perception tools acquire their own fresh observation. Do not call `observe` first unless visual
  inspection is needed to select or disambiguate the subject or target.
- Ground geometry from a fresh official observation before a metric action.
- Require evidence sufficient for the next action's precision and contact risk, not a uniformly
  precise scene model. An endpoint miss measures that attempt, not the best achievable accuracy;
  missing camera extrinsics prevent world-frame localization, not use of its RGB view for alignment.
- Prefer one collision-aware motion whose returned terminal state verifies the requested effect.
- Preserve the current gripper through metric motion unless a change is intended. Gripper commands are
  actions, not acquisition probes: use them only when the gripper-state transition is the grounded
  subgoal, and verify support before releasing a load.
- Use `cap_set_gripper` for one arm. Position `0.0` is fully closed, `1.0` is fully open,
  and intermediate values are proportional. One successful call completes the transition;
  do not repeat it when the returned gripper state already
  verifies the target.

## G1 station

Applies when the capability graph shows `g1_real` (operator runbook: `docs/g1-real.md`).

- Each arm has seven joints in Unitree order: shoulder_pitch, shoulder_roll,
  shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw. Joint targets are radians.
- The base frame is `torso` (`torso_link`); give pose targets in it.
- Reach is short, about 0.55 m from the shoulder. Targets outside the joint limits are
  refused with "outside the configured limits"; reposition the target, not the tolerance.
- **Grip (Dex3):** `cap_close_gripper(arm)` closes the hand in steps and stops when
  the fingers are blocked by an object, then squeezes lightly; it reports
  `contact` (False means it closed on nothing) and the `gripper_position` it left,
  which later arm moves hold. `cap_open_gripper(arm)` opens the hand fully. Prefer
  these to `cap_set_gripper(0.0)`, which drives to full close and keeps pushing into
  the object. Contact is not proof of a grasp: lift a little and check the image.
- `cap_set_gripper` maps its scalar between open (all zeros) and the profile's
  full-close pose. For finger articulation call
  `cap_g1_real__set_hand_joints(arm, joints)`: seven radians in Dex3 order, clamped to
  limits, at most 0.25 rad per message. Read fingers with `cap_g1_real__get_hand_state()`.
- `cap_g1_real__get_controller_metadata` reports contract `g1-sonic-v1-planner-stream`.
- SONIC owns balance and the keep-alive; the operator engages it and the e-stops are
  physical. Do not try to start or stop the controller.
- **cuRobo on the G1** plans around what the head camera's depth shows (the
  robot's own arms are cut out; the floor is dropped). Its plans run about 120 to
  260 waypoints, so pass `max_steps: 300` with `trajectory_planner: "curobo"`. It
  sees only the faces the camera sees: space behind or under an object counts as
  free, and a scene it cannot solve is refused ("could not find a feasible plan"),
  which moves nothing. On a refusal, raise the hand and come back toward the body in
  short `interpolation` moves (see the next item), then plan again. From Home the hand
  sits below the work surface next to it, and cuRobo refuses that narrow start (sim):
  first lift the hand straight up above the surface with `interpolation` (handshake
  orientation `[1, 0, 0, 0]` reaches), then plan with cuRobo.
- **Extra camera views.** A station may list image-only cameras next to `ego_view`
  (the sim has `third_view`, a fixed camera ahead and to the right). They show the
  hand and the object from the side, which is what the head camera loses once the
  hand is in front of the object. Use them to judge contact, grasp and placement.
  They have no 3D: localize with `ego_view` only. Motion results include every
  camera; `observe` returns only `ego_view` unless asked, so call
  `observe(cameras=["ego_view", "third_view"])`.
- **Posture and the legs.** `cap_g1_real__get_posture()` reports, in the torso frame,
  each arm's shoulder, elbow, wrist, palm center and palm normal, the hips, knees and
  ankles, and each forearm's clearance to the legs. Arm trajectories that would bring
  a forearm or hand within 3 cm of the legs are refused ("nothing moved"), unless
  they move away from them; pick a target higher or farther out.
- **Leaving a work surface.** Neither the planner nor Home knows about tables,
  boxes or the robot's own legs. After releasing or touching something, back the
  hand away from the object, raise it 10 to 15 cm above the surface, and bring it
  back toward the body past the surface's near edge before any large move. Never
  lower the hand while it is over a surface except to place or touch. Home does
  this on its own from a raised hand (lift, back over the home position, straight
  down); a `cap_move_to_pose` to a start pose does not.
- Task recipes: [bottle pick and place](../../../examples/g1_real/bottle_pick/TASK_CONTEXT.md)
  needs Dex3 hands (PC2 `.222`); [palm touch](../../../examples/g1_real/bottle_touch/TASK_CONTEXT.md)
  works on both robots and says what differs per hand. Read the one that matches the
  task; `cap_g1_real__get_controller_metadata` reports the connected `station`.
