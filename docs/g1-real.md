# Real G1 operator runbook

A Unitree G1 is balanced by NVIDIA GEAR-SONIC while ASPIRE streams the two 7-joint
arms and the Dex3 hands over SONIC's ZMQ planner topic. This runbook covers the G1
station; the YAM equivalent is [System 2 operator runbook](system2.md). The agent
side is [CAP tool usage](../.agents/skills/cap-tool-usage/SKILL.md) (section "G1
station") and [the bottle task](../examples/g1_real/bottle_pick/TASK_CONTEXT.md).

Steps marked **Pending operator run** need the robot or the SONIC MuJoCo loop and
have not been run; fill in the result placeholders when you do.

## 1. Topology

| Service | Host | Port | Direction |
| --- | --- | --- | --- |
| ASPIRE SONIC command PUB (`command`, `planner`) | workstation `192.168.0.62` | 5556 | PC2 `deploy.sh` connects |
| SONIC feedback PUB `g1_debug` | PC2 `unitree@192.168.0.223` | 5557 | ASPIRE SUB |
| Composed camera PUB (RGB + aligned depth) | PC2 | 5555 | ASPIRE SUB |
| SAM3 | `192.168.75.174` (`h100_174`), GPU 3, Docker container `jihun-aspire-sam3-174-gpu3-261020` | 8114 | ASPIRE HTTP |
| cuRobo | `192.168.75.174` (`h100_174`), GPU 2, Docker container `jihun-aspire-curobo-174-gpu2-261020` | 8118 | ASPIRE HTTP |
| CAP MCP / Viser | `.62` | 8222 / 8080 | agent / operator |

SAM3 runs from `docker/sam3.Dockerfile` (image `jihun-aspire-sam3:<commit>`, built
on the host from a clone under `~/aspire-sam3/ASPIRE`). The gated weights are a
read-only Hugging Face cache at `~/aspire-sam3/hf`, streamed from the workstation, so
the shared `kube` account holds no token; the container runs offline with
`--restart unless-stopped`. Check it with
`curl --noproxy '*' http://192.168.75.174:8114/healthz`. Verified 2026-10-20 on a
live frame: "water bottle" 0.96 and "black box" 0.91, and the floor back-projects
level at 0.84 m below the torso origin, so the model's camera transform holds.

Depth cleanup in the workstation's camera client (2026-10-20): pixels on depth jumps
(RealSense flying pixels) are dropped, and with `camera.floor_below_z_m` set the floor
is fitted as a plane in every frame and dropped too, so localization and planning
never see it. The System 2 viewer shows the main camera's depth next to its RGB view
(near red, far blue over 0.3 to 1.5 m, black where there is no depth or it was
dropped). The depth view is for the operator; the agent still receives RGB only.

**Main robot (PC2 `192.168.0.222`, `ssh pc2_222`, Dex3), set up 2026-10-20.** Use
`--station g1-222` (same profile as `g1-lab` with the .222 hosts). Its checkout is
NVlabs `087f9ac` with local edits; the camera package now carries the fork's depth
patch (`6a1741c`), and the local camera edits it replaced (device selection, frame
rate, depth switch, all included in the patch except a 30 fps default) are backed up
in `~/aspire-backup-20261020` on .222. Another stack's `g1_head_camera_server.py`
(`/dev/video4`, "egocentric") also serves port 5555 there and holds the RealSense;
stop it before starting ASPIRE's camera server (section 2). The deploy there
publishes `init_base_quat`, which the heading fix uses.

**Choosing the robot.** Both launchers take `--robot 222` (main robot, Dex3) or
`--robot 223` (rubber hands). The server maps it to `--benchmark g1_real --station
g1-222` or `g1-lab`; the agent launcher tells the agent which hands it has and which
recipe applies, and on 223 does not pre-approve the gripper and hand tools:

```bash
scripts/start_system2.sh --robot 222 --allow-motion --viser-port 8081 \
  --sam3-url http://192.168.75.174:8114 --curobo-url http://192.168.75.174:8118
scripts/start_system2_agent.sh --robot 222 --allow-motion
```

**Home retreat (2026-10-20).** Home from a raised hand (more than 5 cm above its
home height) first lifts it 8 cm (4 cm or none if out of reach), brings it back over
its home position at that height, and lowers it straight down, in 2 cm IK waypoints
at 0.10 m/s, before the usual joint move; the result reports `retreat_waypoints`.
A Home from over a work surface therefore takes about 8 s. The straight joint-space
Home it replaces swept the hand down into the box after a bottle placement.

Only one process can bind 5556 on the workstation. **Stop the PICO manager / XR
bridge on `.62` before starting ASPIRE.** If SAM3/cuRobo run on another GPU host,
point `sam3_url` / `curobo_url` there (profile fields, or `--sam3-url` /
`--curobo-url`).

## 2. Start order

1. Stop any PICO manager / XR bridge on `.62`.
2. PC2, camera. PC2's checkout (`~/GR00T-WholeBodyControl`, NVlabs `main` plus
   uncommitted deploy edits that the built binary depends on) carries the fork's
   RealSense depth patch as an overlay of `gear_sonic/camera/` from fork commit
   `6a1741c` (deployed 2026-10-19; `git status` shows the three files modified,
   `git checkout -- gear_sonic/camera` reverts). RealSense init takes about 20 s
   before the port binds:

   ```bash
   tmux new-session -d -s gr00t_camera_5555 "cd ~/GR00T-WholeBodyControl && ~/GR00T-WholeBodyControl/.venv_camera/bin/python -u -m gear_sonic.camera.composed_camera --ego-view-camera realsense --realsense-enable-depth --port 5555 > /tmp/gr00t_camera_5555.log 2>&1"
   tail -f /tmp/gr00t_camera_5555.log   # until "Sensor server running at tcp://*:5555"
   ```

   Verified 2026-10-19 from the workstation: 640x480 RGB, aligned depth (all
   pixels valid, 1.2 to 3.0 m in the lab), fx 606.6 / fy 607.1, about 15 Hz.

3. PC2, deploy; wait for `Init done`:

   ```bash
   cd gear_sonic_deploy && ./deploy.sh --input-type zmq_manager --output-type zmq --zmq-host 192.168.0.62 real
   ```

   (The fork's interactive `scripts/preflight.sh` safety checklist is not in PC2's
   checkout; run through its items by hand: harness, 3 m clear zone, spotter,
   keyboard `O` drilled.)

4. Workstation, providers (skip if running on the other GPU host):

   ```bash
   scripts/supervise_services.sh --providers sam3,curobo --profile "$ASPIRE_PROFILE"
   ```

5. Workstation, CAP MCP. This binds 5556, and with `--engage` runs the engage
   sequence (below) at server startup: it waits for fresh feedback, holds the
   measured pose, and sends `start` once.

   ```bash
   scripts/start_system2.sh --benchmark g1_real --station g1-lab \
     --station-config-root <root> --allow-motion --engage \
     --sam3-url http://127.0.0.1:8114 --curobo-url http://127.0.0.1:8118
   ```

   `--engage` requires `--allow-motion`; the server exits non-zero when the
   feedback is stale. On `.62` port 8080 belongs to a Docker container, so add
   `--viser-port 8081` (or `--no-viser`). Without `--engage` nothing is sent until the operator
   presses the **Engage SONIC** button on the Viser page (http://127.0.0.1:8080,
   camera-only for G1). The same button re-engages after a feedback outage.
6. Operator confirms the robot stands in PLANNER idle with fresh feedback. The
   keep-alive is now live.
7. Attach the agent; it homes once and reports READY:

   ```bash
   scripts/start_system2_agent.sh --allow-motion
   ```

## 3. Engage

**Heading (fixed 2026-10-19).** The first real engage turned the robot 90 deg
left. SONIC's planner reads `facing` in a frame whose yaw zero is the heading
the policy initialised with, and ASPIRE was sending the raw IMU yaw. The body
client now sends the IMU yaw relative to `reference_heading_quat` (fork deploy
builds, while `planner_reference_active`) or `init_base_quat` + `delta_heading`
(older builds such as PC2's), and zero when neither is published, which holds the
planner's own heading. Sim cannot catch this (the sim robot starts at yaw zero),
so the first engage after any change here is done with the robot **hanging,
feet off the floor**: the legs must stay still, not step or turn.

**First engage on a fresh deploy (fixed 2026-10-19).** SONIC publishes no
feedback before `start`, so ASPIRE cannot know where the arms are. It used to
stream the profile home blind before `start`, and SONIC's first control tick
drove the arms straight there from wherever they hung: an abrupt jump, and the
arms were still moving when the first plan ran (`stale_plan`). Now the first
engage sends `start` alone, lets SONIC bring the arms up through its own idle
motion, waits for feedback and for the arms to be still (below 0.1 rad/s for
0.3 s, at most 5 s), then seeds the stream from the measured pose. Expect
SONIC's normal start motion, then nothing from ASPIRE until a command.

**After a PC2 reboot** the camera server is gone (its tmux session and
`/tmp` log do not survive); restart it as in section 2 before any run that
reads images, including the acceptance segments, which build observations.

In `zmq_manager` mode SONIC has no `]` start key. ASPIRE's operator-only engage path
waits for fresh feedback, streams a hold at the measured pose for 0.2 s, sends
`command{start=1, planner=1}` once, then starts the 50 Hz keep-alive. SONIC drops
the arms to its idle pose after 1 s of silence, which is why the keep-alive exists.
Engage runs from `--engage` at startup or the Viser button; the batch runner
(`cap-harness run`) engages on `--allow-motion`. A second engage on a healthy stream
sends nothing (`already_engaged`). After a feedback outage the keep-alive resumes by
holding the measured pose (SONIC idled the arms meanwhile, so the pre-outage target
is not re-sent), and the Viser button runs the full hold plus `start` again. The
agent cannot engage. ASPIRE never sends `stop`.

## 4. Preflight gates

All must hold before an agent task:

- feedback age < 100 ms
- both hands at the open pose
- home residual <= 0.03 rad
- camera age < 500 ms with depth present
- SAM3 and cuRobo `/healthz` ok
- `planner_reference_active == 1`

```bash
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8114/healthz
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8118/healthz
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8222/health
```

## 5. Stop

Place or release held objects first, then press `O` on PC2 (or PICO A+B+X+Y), then
stop MCP. Keyboard `O` and PICO A+B+X+Y remain the e-stops; stopping ASPIRE is not
one, and the arms drop to SONIC's idle pose within 1 s of it exiting.

## 6. Off-robot checks

### cuRobo model

```bash
scripts/supervise_services.sh --providers curobo --profile "$ASPIRE_PROFILE"
```

From `.venv-system2`, build a `RobotPlanningContext` as
`G1RealAdapter.get_planning_context()` does (model `g1_real`, per-arm `joint_names`,
identity bases, `end_effector_links {side: f"{side}_grasp"}`), then call
`CuRoboProvider(base_url="http://127.0.0.1:8118", control_period_s=0.02).solve_ik(...)`
for the right arm's own home pose (from `G1Kinematics.forward_kinematics`). Expect
`ok=True`. Then `plan_to_pose` to the home pose + `[0.05, 0, 0.05]`: expect
`ok=True` and `len(trajectory.joint_positions) <= 120`.

Result (2026-10-20, service in Docker on `h100_174` GPU 2, image built from
`docker/curobo.Dockerfile`): IK for the right arm's home pose `ok` (about 60 s on the
first call while cuRobo compiles its kernels, then 0.02 s); `plan_to_pose` to home +
`[0.05, 0, 0.05]` `ok` with 127 waypoints, more than the default `max_steps` of 120,
so pass `max_steps: 300` with cuRobo. The smoke found that the G1 model had no
`collision_link_names`, so cuRobo loaded no spheres and every plan was unchecked;
fixed in `47a746c` (plus a palm sphere per hand). After the fix, a return from 8 cm
above a closed box to home keeps the fingertip and palm spheres 0.3 to 1.7 cm clear
of it (it went 4 cm inside before); with only the top and near face (what the head
camera sees) the planner refuses rather than cutting through.

### Fake station

```bash
cap-harness run --benchmark g1_real --station g1-lab --station-config-root <root> \
  --suite joint_motion --task-id 0 --program examples/g1_real/joint_motion.py \
  --allow-motion --station-backend fake
scripts/start_system2.sh --benchmark g1_real --station g1-lab \
  --station-config-root <root> --allow-motion --fake
```

### Fake-plant checks (run on the workstation)

Run 2026-10-17 from the worktree root with `PYTHONPATH= VIRTUAL_ENV=` cleared. The
fake plant is an engaged first-order-lag body with a synthetic camera; this exercises
the real CLI and the real MCP server, not SONIC.

Batch runner (exit 0):

```bash
.venv-system2/bin/cap-harness run --benchmark g1_real --station g1-lab --station-backend fake \
  --suite joint_motion --task-id 0 --program examples/g1_real/joint_motion.py \
  --allow-motion --output-root <scratch>/g1-fake-run
```

Observed: `program_ok=True task_success=False termination_reason=program_completed`;
`episode/raw/right-joint_pos.npy` has shape `(29, 7)`.

System 2 server on port 8223, driven by `scripts/g1_fake_mcp_smoke.py`:

```bash
scripts/start_system2.sh --benchmark g1_real --station g1-lab --fake --allow-motion \
  --engage --no-viser --port 8223
.venv-system2/bin/python scripts/g1_fake_mcp_smoke.py --port 8223   # --port defaults to 8222
```

Observed: `/health` returned `{"ok":true,"job":null}`; the tool list holds
`cap_g1_real__get_controller_metadata`, `cap_g1_real__get_hand_state` and
`cap_g1_real__set_hand_joints` beside the `cap_*` motion tools. `cap_program_go_home`:
`home_verified: true`, `joint_residual_rad` left 0.0 / right 1.35e-07. Right
`cap_move_to_pose` to the measured position + 5 cm x (frame `torso`, `mink` +
`interpolation`): `ok: true`, `position_error_m` 4.4e-05, `orientation_error_rad`
1.3e-03, `joint_target_converged: true`. `cap_g1_real__get_hand_state`: all zeros.
`cap_g1_real__set_hand_joints` left with the profile `closed` values: `ok: true`; the
final left `gripper_positions` is 0.0175 (1.0 is open). The script exits 0 and prints `OK`.

Note: on g1_real `move_to_pose` reports task-space errors, not `joint_residual_rad`;
only `cap_program_go_home` does. The sim2sim step therefore gates the move on
`position_error_m` < 0.01, `orientation_error_rad` < 0.10 and `joint_target_converged`
(the smoke script requires the same). The smoke script moves whatever server it is
pointed at: re-run it against the real server only with the operator ready.

### Sim with camera and objects (`scripts/g1_sim.sh`)

One command runs the SONIC MuJoCo sim headless with a black case (0.84 m, top at torso
height) and a 500 ml bottle on it, publishes the head camera's RGB, depth and
intrinsics in the PC2 camera format, starts the SONIC deploy, and with `smoke` also a
sim CAP server and a check list:

```bash
scripts/g1_sim.sh smoke          # up, engage, drop, checks, down (--keep leaves it running)
scripts/g1_sim.sh up             # or by hand: sim + deploy only
scripts/start_system2.sh --robot sim --allow-motion --engage --sam3-url http://192.168.75.174:8114
scripts/g1_sim.sh drop           # drop the robot onto its feet once SONIC holds it
scripts/g1_sim.sh reset          # lay the case and bottle out in front of it (after drop)
scripts/start_system2_agent.sh --robot sim --allow-motion
scripts/g1_sim.sh down
```

The sim lives in `scripts/g1_sim/sim_rgbd.py`, which runs the fork's simulator with its
own Python and leaves the fork unchanged: it swaps the scene at load (case, bottle, and
`head_camera` moved to the real D435 pose and field of view, so the `g1-sim` profile
shares the real camera transform), renders on the GPU through EGL, and toggles the
elastic band on a signal instead of the viewer's `9`. It uses its own ports (camera
5565, command 5566, feedback 5567, MCP 8223, viewer 8083) and can run next to a
real-robot server. Each camera message also carries `sim_truth`, the objects' torso-frame
positions, which `scripts/g1_sim/smoke.py` checks localization against.

Result (2026-10-21): all seven checks pass (camera RGB-D and intrinsics, station, Home,
right hand 5 cm up at 8 mm, Dex3 close on nothing and open, SAM3 localizes the bottle
3.6 cm from its true center). Building it surfaced a lost `start` when engage runs
right at server startup (ZMQ slow joiner); the body client now waits 1 s after binding
and the first engage resends `start` every 2 s until feedback arrives.

### Sim experiments (`scripts/g1_sim/experiments.py`)

Run against `scripts/g1_sim.sh up` (no CAP server on the sim's port). Verdicts come
from the sim's truth (object poses, robot-object contacts on every physics step).
Results, 2026-10-21:

| Experiment | Result |
| --- | --- |
| `engage` (fresh deploy, then drop) | standing hold still (0.10 rad/s peak) and within SONIC's offset (0.135 rad). While hanging on the band SONIC swings the arms up to 9 rad/s, so the first engage's wait for still arms runs its full 5 s: on the harness keep the feet on the ground or drop promptly. |
| `heading` (`G1_SIM_YAW_DEG=60`) | heading holds (+4 deg) through engage and drop; with the pre-fix facing (`--legacy-heading`) the sim turns the robot 60 deg back, reproducing the robot's turn-on-engage |
| `retreat` | from 10 cm over the case the old straight Home hits it (fingers and wrist, 16-19 N); the retreating Home does not (10.7 s instead of 3.3 s) |
| `grip` | palm contact at the palm center, contact-aware close, lift: with 0.08 squeeze the 0.5 kg bottle slipped; with 0.15 3 of 3 lifted it 3.7-5.1 cm and set it down upright. Re-measured after agent runs 5 and 6 slipped on the first lift (bottle at x 0.36, y -0.06, index finger base first, contact at closure ~0.6), 4 runs each: 0.15 lifted 0/4 (0.5-2.3 cm), 0.25 4/4 (4.0-4.7 cm), 0.35 4/4 (4.4-5.0 cm), all set down upright. 0.35 is now the default |
| `outage` (feedback cut 2 s while SONIC runs) | motion refused as stale; the arms move 0.08 rad when feedback returns; re-engage moves them 0.035 rad; motion works again |
| `curobo` (live depth scene) | from Home cuRobo refuses (the hand starts below the case top beside its face); after an interpolation lift every leg plans and runs with no case contact. Plans right after a long move were refused as stale under the 0.02 rad start check; `move_to_pose` now re-plans |
| `lean` | reaching 0.25-0.48 m out tilts the torso 1-2 deg, but SONIC shifts the stance sideways: a fixed object moves up to 4.5 cm in the torso frame |
| `grip --bottle-xy` sweep | bottle placed over x 0.34-0.52, y -0.18-0.06 (torso frame, case top), 3 runs per reachable point. x >= 0.46 is out of reach for the handshake palm site (`mink` misses by 2-9 cm); x 0.40, y -0.18 misses by 1.2-1.4 cm. Over the 8 reachable points: touch 19/24, close on contact 19/24, lift >= 3 cm 8/24, lift and set down upright 6/24. Best band y -0.10 (4/6). The first contact is almost always the index finger base, the hand ends 1-4 cm short of its target toward the body, and the close then holds the bottle by the fingers, which slips on lift. The placement also drifts up to 10 cm in the torso frame between engages (stance and yaw). With `--past-contact 0.03` (3 cm more after first contact): lift >= 3 cm 11/24, lift and set down upright 10/24, but 4/24 runs pushed the bottle off the case (none before). It helps on the right (y -0.18 to -0.10 at x 0.34: 5/6 full) and topples the bottle near y -0.02. |
| acceptance (`--station g1-sim`) | hold 0.11 rad; segments 5 of 10 within 15 mm, the returns to the start pose plateau near 32 mm (SONIC tracking; real robot: 0.11 rad, 38 mm) |

Agent runs in sim (same task: touch, grasp, lift 5 cm, put back, release, Home):
1. The hand hid the bottle from `ego_view`, the first contact was a finger base at
   1-2 N the agent could not see, and its approach steps pushed the bottle 3 cm and off
   the case. The sim deploy publishes no hand torques, so a third-person camera
   (`camera.aux_roles`, `third_view` in sim) was added and the recipes say to judge
   contact from it and stop when no view shows it.
2. With `third_view`: the fingertips struck the case top (18-21 N) and the middle
   finger tipped the bottle. Posture tool and leg guard added afterwards.
3. With the posture tool and leg guard: the bottle stayed upright on the case, the
   only contacts were 3 N (a right finger base on the bottle, fingertips on the case)
   and an 11 N left index fingertip on the case. The agent reported the touch as
   unverified and stopped: cuRobo refused two approach goals as `scene_collision` (the
   bottle and case are one depth obstacle, so a palm goal against the bottle collides),
   and `mink` missed a 1 cm tolerance by 1.9 cm. The leg guard never fired. The torso
   pitched about 9 deg during startup Home and then held, which alone moves an object
   0.35 m ahead by 5-6 cm in the torso frame: localize after Home, not before.
   `watch --summary` now also prints the bottle's world drift and the torso's yaw.
4. With the reach map in the recipes (pick, lift 5 cm, put back, release, Home; bottle at
   torso x 0.36, y -0.06): the agent never reached the bottle. Its staging move (right
   site at 0.24, -0.244, 0.20, handshake) first exceeded `max_steps` (171 > 120). With
   `max_steps` 300, the first pass ended 3 cm off while SONIC yawed the torso 14 deg.
   The correction round then ended 45.5 cm off: the torso lurched 5 cm, turned to -22 deg
   yaw, the uncommanded left hand struck the case (up to 58 N), and the right hand
   finished over the head (z 0.48). Home was then refused, because the retreat IK put the
   right wrist pitch at the model limit (-1.61443), just past the profile's -1.6144.
   Retreat rows are now clipped to the profile limits. The correction round diverged
   because the staging target sits at the right elbow's limit (-1.047), where mink's
   solution is very sensitive: the round's 2.6 cm and 9 deg re-command moved the
   solution 1.6 rad (shoulder pitch 1.0 to 2.6), whatever the seed. `move_to_pose` now
   refuses a correction round whose IK solution is more than 0.5 rad from the previous
   command and returns the earlier round's result with `correction_joint_jump_rad`.
   Replayed in the sim: run 4's staging (jump 1.54 rad) and the grip staging pose
   (0.59 rad) stop after the first round at 2-3 cm; ordinary rounds (5 cm up, 3 cm
   forward, hand turned -30 and -45 deg) moved joints 0.06-0.2 rad and still ran. The
   bottle stayed upright but slid 5 cm. SAM3 returned a 0.22 m box (score 0.04) that the
   recipe says to reject; the agent never used it.
5. With the correction guard (same task, bottle at torso x 0.36, y -0.04): **done**. Sim
   truth: the bottle rose 5.1 cm, ended upright 0.7 cm from its start in the world, and
   tilted at most 0.96 upright. The first close (site at 0.372, -0.062, 0.086) slipped on
   the lift; the agent opened, went 2 cm lower (0.36, -0.05, 0.065), closed again, and
   the lift held. 31 tool calls, 87k tokens, about 4 min. Open points:
   - Staging move (0.229, -0.248, 0.16): its second correction round went 1.1 to 7.6 cm
     while SONIC yawed the torso 13 deg; the stall check stopped it, without a swing. The
     guard did not fire, since the joints jumped less than 0.5 rad.
   - The torso-frame bottle position drifted from y -0.09 to 0.00 over five localizations
     (the torso yawed 33 to 22 to 29 deg) while the bottle stayed put in the world; the
     agent re-localized before each approach, which kept it on target.
   - The uncommanded left hand rested on the case (1-5 N) from the fourth move on and
     pressed it at 33 N during the final Home (left index finger base).
   - Startup Home ended unverified (residuals 0.059 / 0.057 rad against 0.03).
6. With `sonic.hold_heading` (bottle at torso x 0.36, y -0.06): **done**, and the left hand
   never touched the case. The body client used to send SONIC the current heading as
   `facing` on every tick, so nothing turned the robot back and each turn an arm reach
   caused became the new command (yaw spans 40 deg in run 4, 20 deg in run 5). The left
   hand, whose home pose is about 6 cm from the case's near-left corner, swings about
   5 cm per 10 deg of yaw; in run 5 Home pressed it into the case at 33 N. With
   `hold_heading: true` the client latches the heading at engage (again on each engage,
   dropped while the heading is unknown, and re-latched rather than commanding a turn
   back of more than 0.5 rad). Run 5's reach sequence twice plus Home: yaw span 23 deg
   and 4.3 deg net drift without, 11.5 deg and 0.3 deg with. Run 6 itself: yaw span
   9.3 deg, no left-hand contact, bottle up 5.8 cm, back upright 1.5 cm from its start;
   the first close slid again and the second held. `hold_heading` is on in `g1-sim`
   only; enable it on a real profile after a check with the robot hanging.
7. With grip squeeze 0.35 (bottle at torso x 0.37, y -0.06): the first close and lift
   held (bottle up 5.6 cm; runs 5 and 6 slipped here). The task then failed. Reading a
   7.3 cm measured hand rise, the agent "corrected" the height 0.8 cm down, the bottle
   came back down onto the case and out of the grip, a second close near the neck and a
   diagonal move (6 cm forward, 5 cm up) tipped it off the case to the floor beside the
   left foot. The agent rejected a localization that found the hand, reported the task
   incomplete and went Home. Why the 0.8 cm move lost the grip is not yet known.

### Sim2sim (SONIC MuJoCo loop)

Run on the workstation on 2026-10-18 (headless: the MuJoCo viewer on an Xvfb
display, `9` injected with pynput; fork at `6a1741c`, ASPIRE at the commit that
records this). Steps 1 to 5 done; step 6 still pending.

Two things the gate corrected in ASPIRE, both now committed:

- SONIC publishes feedback only in its CONTROL state, so the first `engage()`
  cannot wait for feedback before `start`. It now holds the profile home on the
  stream, sends `start`, waits for feedback (5 s), then seeds the chain from
  the measured pose. A later `engage()` with feedback present keeps the old
  order. Both orders were exercised against the real deploy binary.
- Missing depth is encoded as zero (no return), not NaN: `CameraObservation`
  rejects NaN and every observation, including `go_home`'s, failed to build.

Camera note: the fork's `.mp4` replay driver publishes an empty image dict
through the composed server (it returns `color_image`, which the server does
not merge); the gate used a 20-line synthetic publisher in the composed
schema instead. Not needed on PC2, where the RealSense driver runs.

1. Fork terminal 1: `source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py`
   (headless: `DISPLAY=:99 ... --enable-onscreen` under `Xvfb :99`; `9` is a
   GLFW key callback only, so the viewer must exist to drop the robot).
2. Fork terminal 2, from `gear_sonic_deploy/`:
   `./deploy.sh --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 sim`;
   wait for `Init done`, then press `9` in the MuJoCo window **after** the
   policy is started (ASPIRE's engage sends `start`; dropping first lets the
   robot fall). Results below were taken with the robot dropped; on the band
   the home residuals were roughly twice as large.
3. Copy the `g1-lab` profile and set `station: g1-sim`, `feedback_host: 127.0.0.1`,
   `camera.host: 127.0.0.1`; keep the same `command_bind`.
4. Run the `cap-harness run` command above (without `--station-backend fake`; the
   batch runner engages on `--allow-motion`, so no separate engage step) and
   `scripts/start_system2.sh --benchmark g1_real --station g1-sim --station-config-root <copy-root> --allow-motion --engage`
   against that profile.
5. Expect `home_verified: true` and a right-arm `move_to_pose` with
   `position_error_m < 0.01`, `orientation_error_rad < 0.10` and
   `joint_target_converged: true`.
6. Kill feedback for 2 s (pause `deploy.sh` output or unplug), restore, press
   Engage SONIC in Viser; expected: arms ramp back from where they are, no jump
   (this also checks that a second `command{start=1, planner=1}` to the running
   `zmq_manager` is accepted); record the result.

Result (steps 1 to 5, 2026-10-18): the whole chain works end to end; the
tracking gate does **not** pass. Engage switched SONIC to CONTROL and enabled
its planner; `cap_program_go_home` streamed and returned `ok: true` with
`home_verified: false`; the Dex3 hands tracked `cap_g1_real__set_hand_joints`
to within about 0.05 rad (closed pose measured `[0.01, 0.01, 1.71, -1.52, -1.71, -1.52, -1.71]`).
Arm tracking, robot dropped, per-joint `measured - home` in rad:

| Checkpoint | left (sh_p, sh_r, sh_y, elbow, wr_r, wr_p, wr_y) | right | max |
| --- | --- | --- | --- |
| `release` | 0.001, -0.005, 0.046, -0.032, 0.033, -0.091, -0.035 | -0.036, 0.007, -0.033, 0.018, 0.050, -0.111, 0.016 | 0.11 (wrist pitch) |
| `sonic_v1_1` | 0.027, 0.015, 0.021, -0.090, 0.028, 0.005, 0.007 | -0.004, -0.047, 0.058, -0.025, 0.091, -0.034, -0.021 | 0.09 |

Right-arm `move_to_pose` (mink + interpolation), end-effector error against
the goal, `sonic_v1_1`: +3 cm forward settles at about 1.0 cm (0.7 to 1.7 cm,
oscillating), -3 cm retract at 2.0 cm, +5 cm up at 2.6 cm; the error is flat
from 1 s to 5 s after the command, so it is a steady bias of the learned
controller, not slow convergence. With `release`, +3 cm forward passed
(0.7 cm) and the retract failed (3.0 cm). Verdict: the spec section 9
criteria (<= 0.05 rad hold, <= 1.5 cm) are not met in sim with either
checkpoint; remedies are the arm kp/kd tuning on PC2 or the decoupled-WBC
fallback (section 7), or a bounded correction of the command in ASPIRE.

Correction rounds (same rig, `sonic_v1_1`, robot dropped, 2026-10-18): the
`controller.pose_correction_rounds` profile knob (3 on `g1-lab`) makes
`move_to_pose` re-plan to the previous command plus the measured residual, up
to that many rounds, verifying each against the original goal; it stops on
success, on a stall (a round that does not beat 75% of the previous error) or
at a 5 cm per-axis offset cap, and reports `correction_rounds` and
`correction_offset_m`. With it, every move passed the 1 cm tolerance:

| Right-arm move (torso frame) | rounds | final EE error | offset from goal (x, y, z) | wall time |
| --- | --- | --- | --- | --- |
| +3 cm forward | 1 | 0.6 cm | -0.3, +1.3, -0.7 cm | 2.2 s |
| -3 cm retract | 0 | 0.6 cm | none | 1.2 s |
| +5 cm up | 2 | 0.9 cm | -2.2, -1.5, -1.1 cm | 3.8 s |
| +10 cm forward | 0 | 0.7 cm | none | 1.4 s |

The error stayed under 1 cm for the 5 s after each return (0.3 to 0.9 cm,
oscillating). The bias is not along the move axis alone (the "up" move needed
a sideways and downward offset), so the task-space loop is the right place
for it rather than a per-axis gain. The raw single-shot bias varies between
runs by about 1 cm, which is why the retract needed no round here and 2.0 cm
earlier. Cost: about 1.1 s per round.

Result (step 6): ______ (pending; the re-engage order after an outage was
exercised once by a server restart with SONIC already in CONTROL, and the
second `command{start=1, planner=1}` was accepted without a jump).

## 7. Acceptance test

Pending operator run. `scripts/g1_tracking_acceptance.py` (Task 15) automates it.

| Test | Pass criterion | Result |
| --- | --- | --- |
| Hold the upper-body target 5 s | max per-joint error <= 0.05 rad, both arms | FAIL 2026-10-19: max 0.11 rad (shoulder yaw, wrist roll), steady after ~2.5 s |
| Ten 0.15 m vertical segments, up from the start pose then back (default `--segment-m 0.15`; the earlier torso-X retract put the hand beside the thigh and it hit the leg); speed is logged per segment, not enforced, because the interpolation planner ignores `time_dilation_factor` | final EE error <= 1.5 cm and <= 5 deg; no 0.10 rad residual failure | FAIL 2026-10-20 (torso-X version, hand touching the leg): worst 37.9 mm, 3 of 10 within 1.5 cm after 1-2 correction rounds |
| Hold a 300 g object | repeat the 5 s hold, error <= 0.05 rad | ______ |

**Operator decision 2026-10-20:** an end-effector error of about 4 cm is accepted
for now. It comes from SONIC's arm tracking, not from ASPIRE, and the session goal
is the ASPIRE-SONIC integration; tracking (kp/kd on PC2) is revisited later.

`move_to_pose` already runs up to `controller.pose_correction_rounds` corrective
rounds (section 6, sim2sim); the segment test reports the rounds each segment
used. If tracking still fails, tune by editing the arm kp/kd constants in the
fork's `policy_parameters.hpp` and rebuilding on PC2 (no runtime gain flag
exists); higher gains make each round count for more. If that
round fails, the fallback is the decoupled-WBC body client, which is not built.
