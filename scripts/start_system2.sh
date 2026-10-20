#!/usr/bin/env bash
# Start the repository's persistent MCP runtime; callers select the station.
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASPIRE_PYTHON="$ASPIRE_ROOT/.venv-system2/bin/python"
[[ -x "$ASPIRE_PYTHON" ]] || { printf 'Run scripts/bootstrap_system2.sh first.\n' >&2; exit 1; }
cd "$ASPIRE_ROOT"
export OPENBLAS_NUM_THREADS=1
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
# --robot selects a G1: 222 is the main robot (Dex3 hands), 223 the rubber-hand one.
ASPIRE_ARGS=()
while (($#)); do
  case "$1" in
    --robot)
      case "${2:-}" in
        222) ASPIRE_ARGS+=(--benchmark g1_real --station g1-222) ;;
        223) ASPIRE_ARGS+=(--benchmark g1_real --station g1-lab) ;;
        *) printf -- '--robot must be 222 (Dex3 hands) or 223 (rubber hands)\n' >&2; exit 2 ;;
      esac
      shift 2 ;;
    *) ASPIRE_ARGS+=("$1"); shift ;;
  esac
done
exec env -u PYTHONPATH "$ASPIRE_PYTHON" -m cap_harness.agent "${ASPIRE_ARGS[@]}"
