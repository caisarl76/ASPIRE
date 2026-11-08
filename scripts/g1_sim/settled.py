"""Wait until the sim robot stands settled, from the sim's own truth; exit 0 if it does in time.

    settled.py [--port 5575] [--timeout 15] [--hold 2] [--landed]

Settled: both soles on the floor, pelvis tilt under 3 deg and pelvis height within 1.5 cm of
its clean stand (0.786 m), held for ``--hold`` s with the pelvis still. ``--landed`` asks only
for both soles down and the pelvis still: the robot has landed after the drop.
"""

from __future__ import annotations

import argparse
import time

import msgpack
import msgpack_numpy
import zmq

STAND_Z, Z_TOL, MAX_TILT, STILL_M = 0.786, 0.015, 3.0, 0.002  # STILL_M: pelvis z change per sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--port", type=int, default=5575)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--landed", action="store_true")
    args = parser.parse_args()
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.subscribe(b"")
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    deadline, since, last_z, truth = time.monotonic() + args.timeout, None, None, {}
    while time.monotonic() < deadline:
        try:
            truth = msgpack.unpackb(sock.recv(), object_hook=msgpack_numpy.decode, raw=False)["sim_truth"]
        except zmq.Again:
            continue
        z, tilt = truth["pelvis_z"], truth["pelvis_tilt_deg"]
        ok = all(truth["feet_on_floor"]) and last_z is not None and abs(z - last_z) < STILL_M
        if not args.landed:
            ok = ok and tilt < MAX_TILT and abs(z - STAND_Z) < Z_TOL
        last_z = z
        since = (since or time.monotonic()) if ok else None
        if since and time.monotonic() - since >= args.hold:
            break
        time.sleep(0.1)
    status = "landed" if args.landed else "settled"
    ok = since is not None and time.monotonic() - since >= args.hold
    print(f"{status if ok else 'not ' + status}: pelvis {truth.get('pelvis_z')} m, tilt "
          f"{truth.get('pelvis_tilt_deg')} deg, feet {truth.get('feet_on_floor')}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
