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
| SAM3 | `.62`, else `192.168.75.173` / `.174` | 8114 | ASPIRE HTTP |
| cuRobo | `.62`, else `192.168.75.173` / `.174` | 8118 | ASPIRE HTTP |
| CAP MCP / Viser | `.62` | 8222 / 8080 | agent / operator |

Only one process can bind 5556 on the workstation. **Stop the PICO manager / XR
bridge on `.62` before starting ASPIRE.** If SAM3/cuRobo run on another GPU host,
point `sam3_url` / `curobo_url` there (profile fields, or `--sam3-url` /
`--curobo-url`).

## 2. Start order

1. Stop any PICO manager / XR bridge on `.62`.
2. PC2, camera (the fork's RealSense driver patch under `gear_sonic/camera/` must
   already be deployed to PC2's checkout):

   ```bash
   tmux new-session -d -s gr00t_camera_5555 "cd <fork> && exec <fork>/.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --realsense-enable-depth --port 5555"
   ```

3. PC2, deploy; wait for `Init done`:

   ```bash
   cd gear_sonic_deploy && scripts/preflight.sh && ./deploy.sh --input-type zmq_manager --output-type zmq --zmq-host 192.168.0.62 real
   ```

4. Workstation, providers (skip if running on the other GPU host):

   ```bash
   scripts/supervise_services.sh --providers sam3,curobo --profile "$ASPIRE_PROFILE"
   ```

5. Workstation, CAP MCP. This binds 5556 and sends no messages yet:

   ```bash
   scripts/start_system2.sh --benchmark g1_real --station g1-lab \
     --station-config-root <root> --allow-motion --engage \
     --sam3-url http://127.0.0.1:8114 --curobo-url http://127.0.0.1:8118
   ```

   `--engage` requires `--allow-motion`. The Viser page (http://127.0.0.1:8080) is
   camera-only for G1 and has an **Engage SONIC** button.
6. Operator engages (below) and confirms the robot stands in PLANNER idle with
   fresh feedback. The keep-alive is now live.
7. Attach the agent; it homes once and reports READY:

   ```bash
   scripts/start_system2_agent.sh --allow-motion
   ```

## 3. Engage

In `zmq_manager` mode SONIC has no `]` start key. ASPIRE's operator-only engage path
waits for fresh feedback, streams a hold at the measured pose for 0.2 s, sends
`command{start=1, planner=1}` once, then starts the 50 Hz keep-alive. SONIC drops
the arms to its idle pose after 1 s of silence, which is why the keep-alive exists.
Engage runs from `--engage` at startup or the Viser button. The agent cannot engage.
ASPIRE never sends `stop`.

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

Pending operator run. Result: ______

### Fake station

```bash
cap-harness run --benchmark g1_real --station g1-lab --station-config-root <root> \
  --suite joint_motion --task-id 0 --program examples/g1_real/joint_motion.py \
  --allow-motion --station-backend fake
scripts/start_system2.sh --benchmark g1_real --station g1-lab \
  --station-config-root <root> --allow-motion --fake
```

### Sim2sim (SONIC MuJoCo loop)

Pending operator run (checklist for Task 14).

1. Fork terminal 1: `source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py`
2. Fork terminal 2, from `gear_sonic_deploy/`:
   `./deploy.sh --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 sim`;
   wait for `Init done`, then press `9` in the MuJoCo window.
3. Copy the `g1-lab` profile and set `station: g1-sim`, `feedback_host: 127.0.0.1`,
   `camera.host: 127.0.0.1`; keep the same `command_bind`.
4. Run the `cap-harness run` command above (without `--station-backend fake`) and
   `scripts/start_system2.sh --benchmark g1_real --station g1-sim --station-config-root <copy-root> --allow-motion --engage`
   against that profile.
5. Expect `home_verified: true` and a right-arm `move_to_pose` with
   `joint_residual_rad.right < 0.10`.

Result: ______

## 7. Acceptance test

Pending operator run. `scripts/g1_tracking_acceptance.py` (Task 15) automates it.

| Test | Pass criterion | Result |
| --- | --- | --- |
| Hold the upper-body target 5 s | max per-joint error <= 0.05 rad, both arms | ______ |
| Ten 0.3 m `move_to_pose` segments at 0.1 m/s | final EE error <= 1.5 cm and <= 5 deg; no 0.10 rad residual failure | ______ |
| Hold a 300 g object | repeat the 5 s hold, error <= 0.05 rad | ______ |

If tracking fails, tune by editing the arm kp/kd constants in the fork's
`policy_parameters.hpp` and rebuilding on PC2 (no runtime gain flag exists). If that
round fails, the fallback is the decoupled-WBC body client, which is not built.
