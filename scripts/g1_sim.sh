#!/usr/bin/env bash
# SONIC MuJoCo sim of the G1 for ASPIRE, on this workstation.
#
#   scripts/g1_sim.sh up      start the sim (headless, RGB-D on 5565) and the SONIC deploy
#                             (command 5566, feedback 5567); waits for both
#   scripts/g1_sim.sh drop    toggle the elastic band (drop the robot after engage)
#   scripts/g1_sim.sh reset [X Y]
#                             place the case and the bottle in front of the robot as it
#                             stands now (do this after drop; it steps when it lands);
#                             X Y puts the bottle there in the torso frame (default 0.36 -0.06)
#   scripts/g1_sim.sh status  sessions, ports, last log lines
#   scripts/g1_sim.sh down    stop everything this script started
#   scripts/g1_sim.sh smoke [--keep]
#                             up, a CAP server (--robot sim, engaged), drop, the checks in
#                             scripts/g1_sim/smoke.py, then down (unless --keep)
#
# Then, for an agent session: scripts/start_system2.sh --robot sim --allow-motion --engage
# (MCP 8223, viewer 8083), scripts/g1_sim.sh drop, scripts/start_system2_agent.sh --robot sim
# --allow-motion. The sim ports never overlap a real-robot server (5555-5557, 8222).
# Overrides: GR00T_WBC_ROOT, TensorRT_ROOT, G1_SIM_CHECKPOINT (sonic_v1_1 or release),
# G1_SIM_YAW_DEG (the robot's initial heading).
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_ROOT="${GR00T_WBC_ROOT:-$HOME/work/GR00T-WholeBodyControl}"
TRT_ROOT="${TensorRT_ROOT:-$HOME/TensorRT/TensorRT-10.13.0.35}"
CHECKPOINT="${G1_SIM_CHECKPOINT:-sonic_v1_1}"
RUN_DIR="${G1_SIM_RUN_DIR:-/tmp/${USER:-aspire}/aspire-g1-sim}"
SIM=aspire-g1sim DEPLOY=aspire-g1sim-deploy CAP=aspire-g1sim-cap
CAMERA_PORT=5565 COMMAND_PORT=5566 FEEDBACK_PORT=5567 MCP_PORT=8223
# tmux sessions inherit its server's environment; a snap terminal's EGL vendor dirs hide
# NVIDIA's EGL and ROS's libddsc breaks the sim's CycloneDDS.
CLEAN_ENV="env -u __EGL_VENDOR_LIBRARY_DIRS -u LD_LIBRARY_PATH"
mkdir -p "$RUN_DIR"

port_bound() { ss -ltn | grep -q ":$1 "; }
running() { tmux has-session -t "$1" 2>/dev/null; }
wait_for() {  # wait_for <seconds> <description> <command...>
  local deadline=$((SECONDS + $1)) what="$2"; shift 2
  until "$@"; do
    ((SECONDS < deadline)) || { printf 'timed out waiting for %s\n' "$what" >&2; return 1; }
    sleep 1
  done
}
sim_signal() {
  [[ -f "$RUN_DIR/sim.pid" ]] && kill "-$1" "$(cat "$RUN_DIR/sim.pid")" \
    || { printf 'sim is not running\n' >&2; exit 1; }
}

up() {
  if running "$SIM"; then printf 'sim already up\n'; return 0; fi
  for port in $CAMERA_PORT $FEEDBACK_PORT; do
    ! port_bound "$port" || { printf 'port %s is in use\n' "$port" >&2; exit 1; }
  done
  [[ -x "$WBC_ROOT/.venv_sim/bin/python" ]] || { printf 'no sim venv in %s\n' "$WBC_ROOT" >&2; exit 1; }
  # A previous run's "Init Done" would satisfy the readiness wait before the new deploy
  # has even truncated its log.
  rm -f "$RUN_DIR"/{sim,deploy,cap}.log "$RUN_DIR/sim.pid"
  tmux new-session -d -s "$SIM" "cd '$WBC_ROOT' && MUJOCO_GL=egl PYTHONPATH= exec $CLEAN_ENV .venv_sim/bin/python \
'$ASPIRE_ROOT/scripts/g1_sim/sim_rgbd.py' --port $CAMERA_PORT --pid-file '$RUN_DIR/sim.pid' \
--layout-file '$RUN_DIR/layout.json' --yaw-deg ${G1_SIM_YAW_DEG:-0} > '$RUN_DIR/sim.log' 2>&1"
  wait_for 60 "the sim camera on $CAMERA_PORT" port_bound "$CAMERA_PORT"
  local policy="policy/$CHECKPOINT"
  tmux new-session -d -s "$DEPLOY" "cd '$WBC_ROOT/gear_sonic_deploy' && unset __EGL_VENDOR_LIBRARY_DIRS LD_LIBRARY_PATH && export TensorRT_ROOT='$TRT_ROOT' \
&& source scripts/setup_env.sh && exec ./target/release/g1_deploy_onnx_ref lo $policy/model_decoder.onnx \
reference/example/ --obs-config $policy/observation_config.yaml --encoder-file $policy/model_encoder.onnx \
--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type zmq \
--zmq-host 127.0.0.1 --zmq-port $COMMAND_PORT --zmq-out-port $FEEDBACK_PORT --disable-crc-check \
> '$RUN_DIR/deploy.log' 2>&1"
  wait_for 180 "the SONIC deploy (Init Done)" grep -qs "Init Done" "$RUN_DIR/deploy.log"
  printf 'sim up: RGB-D %s, SONIC command %s / feedback %s, logs in %s\n' \
    "$CAMERA_PORT" "$COMMAND_PORT" "$FEEDBACK_PORT" "$RUN_DIR"
}

down() {
  for session in "$CAP" "$DEPLOY" "$SIM"; do
    if running "$session"; then tmux kill-session -t "$session"; fi
  done
  rm -f "$RUN_DIR/sim.pid"
  ports_free() { ! port_bound $CAMERA_PORT && ! port_bound $FEEDBACK_PORT && ! port_bound $MCP_PORT; }
  wait_for 20 "the sim ports to close" ports_free
  printf 'sim down\n'
}

status() {
  for session in "$SIM" "$DEPLOY" "$CAP"; do
    printf '%-22s %s\n' "$session" "$(running "$session" && echo running || echo stopped)"
  done
  for port in $CAMERA_PORT $COMMAND_PORT $FEEDBACK_PORT $MCP_PORT; do
    printf 'port %-5s %s\n' "$port" "$(port_bound "$port" && echo bound || echo free)"
  done
  for log in sim deploy cap; do
    [[ -f "$RUN_DIR/$log.log" ]] && { printf -- '--- %s.log\n' "$log"; tail -3 "$RUN_DIR/$log.log"; }
  done
  return 0
}

smoke() {
  local keep=0
  [[ "${1:-}" == "--keep" ]] && keep=1
  # its drop toggles the band: on a sim whose robot already stands it would lift it again
  ! running "$SIM" || { printf 'sim already up; run %s down first\n' "$0" >&2; exit 1; }
  up
  local providers=()
  if curl --noproxy '*' -sf -m 3 http://192.168.75.174:8114/healthz >/dev/null; then
    providers+=(--sam3-url http://192.168.75.174:8114)
  fi
  ! port_bound "$MCP_PORT" || { printf 'port %s is in use\n' "$MCP_PORT" >&2; exit 1; }
  tmux new-session -d -s "$CAP" "cd '$ASPIRE_ROOT' && exec scripts/start_system2.sh --robot sim \
--allow-motion --engage --no-viser ${providers[*]} > '$RUN_DIR/cap.log' 2>&1"
  wait_for 120 "the sim CAP server on $MCP_PORT" curl --noproxy '*' -sf -m 2 "http://127.0.0.1:$MCP_PORT/health" -o /dev/null
  sleep 3
  sim_signal USR1  # drop the robot onto its feet now that SONIC holds it
  sleep 10          # it steps to catch itself; let it settle
  sim_signal USR2   # then lay the case and the bottle out in front of it
  sleep 2
  local status=0
  env -u PYTHONPATH "$ASPIRE_ROOT/.venv-system2/bin/python" "$ASPIRE_ROOT/scripts/g1_sim/smoke.py" \
    --mcp-port "$MCP_PORT" --camera-port "$CAMERA_PORT" || status=$?
  ((keep)) || down
  return "$status"
}

case "${1:-}" in
  up) up ;;
  drop) sim_signal USR1 ;;
  reset)
    if (($# == 3)); then printf '{"bottle_xy": [%s, %s]}\n' "$2" "$3" > "$RUN_DIR/layout.json"
    else rm -f "$RUN_DIR/layout.json"; fi
    sim_signal USR2 ;;
  down) down ;;
  status) status ;;
  smoke) shift; smoke "$@" ;;
  *) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
