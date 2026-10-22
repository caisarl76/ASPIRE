#!/usr/bin/env bash
# Robot check for sonic.hold_heading on g1-222, robot hanging (docs/g1-real.md, section 3).
#   server  CAP server on a temporary copy of g1-222 with hold_heading on; engages at startup
#   watch   prints the heading SONIC's facing is read in, from the feedback, at 5 Hz
# The repo profile stays untouched (hold_heading off) until this check passes.
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/tmp/jihun/g1-heading-check
case "${1:-}" in
  server)
    mkdir -p "$ROOT/g1-222"
    sed 's/^sonic:$/sonic:\n  hold_heading: true/' \
      "$ASPIRE_ROOT/src/cap_harness/configs/g1_real/g1-222/station.yaml" > "$ROOT/g1-222/station.yaml"
    grep -q '^  hold_heading: true$' "$ROOT/g1-222/station.yaml"
    exec "$ASPIRE_ROOT/scripts/start_system2.sh" --robot 222 --station-config-root "$ROOT" \
      --allow-motion --engage --viser-port 8081
    ;;
  watch)
    cd "$ASPIRE_ROOT"
    exec env -u PYTHONPATH -u VIRTUAL_ENV .venv-system2/bin/python - <<'PY'
import math, time
from cap_harness.g1_real.body_client import SonicZmqBodyClient
from cap_harness.g1_real.sonic.zmq_state_subscriber import ZMQStateSubscriber
sub = ZMQStateSubscriber("192.168.0.222", 5557, "g1_debug")
first = None
while True:
    msg = sub.get_msg()
    if msg is not None:
        h = SonicZmqBodyClient._known_heading(msg)
        if h is None:
            print("heading unknown (no init_base_quat / reference_heading_quat): facing 0, nothing held")
        else:
            first = h if first is None else first
            d = math.degrees(math.atan2(math.sin(h - first), math.cos(h - first)))
            print(f"heading {math.degrees(h):+7.1f} deg   from the first reading {d:+6.1f} deg")
    time.sleep(0.2)
PY
    ;;
  *) printf 'usage: %s server|watch\n' "$0" >&2; exit 2 ;;
esac
