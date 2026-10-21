"""Checks against a running G1 sim and its CAP server (scripts/g1_sim.sh smoke runs this).

WARNING: this moves the robot it is pointed at; the defaults are the sim's ports.
Exit 0 when every check passes (a SAM3 check is skipped when SAM3 is not configured).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import msgpack
import msgpack_numpy
import numpy as np
import zmq

from cap_harness.g1_real.cameras import ComposedCameraRgbd

LOCALIZE_TOLERANCE_M = 0.05


async def _call(session: ClientSession, name: str, args: dict | None = None) -> dict:
    result = await session.call_tool(name, args or {})
    return json.loads(result.content[0].text)


def _terminal(value: dict) -> dict:
    """The tool's own result, whether returned directly or through a job."""
    job = value.get("job") or {}
    return (job.get("result") or value.get("result") or {}) if job else value.get("result", value)


def _sim_truth(port: int) -> dict:
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.subscribe(b"")
    sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        message = msgpack.unpackb(sock.recv(), object_hook=msgpack_numpy.decode, raw=False)
    finally:
        sock.close()
    return message.get("sim_truth", {})


def check_camera(port: int) -> tuple[bool, str]:
    camera = ComposedCameraRgbd("127.0.0.1", port, role="ego_view")
    try:
        deadline = time.monotonic() + 10.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = camera.read()
            time.sleep(0.1)
        if frame is None:
            return False, "no frame in 10 s"
        valid = float(np.mean(frame.depth_m > 0))
        ok = valid > 0.5 and not camera.intrinsics_missing
        return (
            ok,
            f"rgb {frame.rgb.shape[1]}x{frame.rgb.shape[0]}, depth {valid:.0%} valid, fx {frame.intrinsics[0, 0]:.0f}",
        )
    finally:
        camera.close()


async def run(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    def record(name: str, ok: bool | None, detail: str) -> None:
        rows.append((name, "SKIP" if ok is None else "PASS" if ok else "FAIL", detail))

    ok, detail = check_camera(args.camera_port)
    record("camera RGB-D + intrinsics", ok, detail)
    url = f"http://127.0.0.1:{args.mcp_port}/mcp/"
    async with (
        streamablehttp_client(url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        meta = _terminal(await _call(session, "cap_g1_real__get_controller_metadata"))
        station = meta.get("station")
        record("station", station == "g1-sim", f"station {station!r}")

        home = _terminal(await _call(session, "cap_program_go_home"))
        record(
            "go_home",
            bool(home.get("ok")),
            json.dumps(home.get("result", {}).get("diagnostics", {}).get("joint_residual_rad")),
        )

        state = _terminal(await _call(session, "cap_get_robot_state"))
        pose = state["end_effector_poses"]["right"]
        target = (np.asarray(pose["position"]) + [0.0, 0.0, 0.05]).tolist()
        move = _terminal(
            await _call(
                session,
                "cap_move_to_pose",
                {
                    "arm": "right",
                    "max_steps": 300,
                    "target_pose": {
                        "position": target,
                        "quaternion_wxyz": pose["quaternion_wxyz"],
                        "frame": "torso",
                    },
                    "strategy": {"ik_solver": "mink", "trajectory_planner": "interpolation"},
                },
            )
        )
        # a failed move's diagnostics carry only the correction rounds; its error has the rest
        details = {
            **((move.get("error") or {}).get("details") or {}),
            **(move.get("diagnostics") or {}),
        }
        error = details.get("position_error_m")
        failure = move.get("error") or {}
        record(
            "move right hand 5 cm up",
            error is not None and error <= 0.03,
            f"error {error if error is None else round(error * 1000)} mm, rounds {details.get('correction_rounds', 0)}"
            + (f" [{failure.get('code')}: {failure.get('message')}]" if failure else ""),
        )

        close = _terminal(await _call(session, "cap_close_gripper", {"arm": "right"}))
        diagnostics = close.get("diagnostics", {})
        record(
            "close right hand (empty)",
            bool(close.get("ok")),
            f"{diagnostics.get('completion')} closure {diagnostics.get('closure')}",
        )
        opened = _terminal(await _call(session, "cap_open_gripper", {"arm": "right"}))
        record("open right hand", bool(opened.get("ok")), "")

        if "cap_localize_object" not in tools:
            record("localize water bottle (SAM3)", None, "SAM3 not configured")
        else:
            truth = _sim_truth(args.camera_port).get("aspire_bottle")
            found = _terminal(
                await _call(
                    session,
                    "cap_localize_object",
                    {"query": "water bottle", "camera_name": "ego_view", "target_frame": "torso"},
                )
            )
            position = ((found.get("geometry") or {}).get("pose") or {}).get("position")
            if position is None or truth is None:
                record("localize water bottle (SAM3)", False, f"ok={found.get('ok')} truth={truth}")
            else:
                miss = float(np.linalg.norm(np.asarray(position) - np.asarray(truth)))
                record(
                    "localize water bottle (SAM3)",
                    miss <= LOCALIZE_TOLERANCE_M,
                    f"{miss * 100:.1f} cm from the sim's bottle",
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-port", type=int, default=8223)
    parser.add_argument("--camera-port", type=int, default=5565)
    args = parser.parse_args()
    rows = asyncio.run(run(args))
    width = max(len(name) for name, _, _ in rows)
    for name, verdict, detail in rows:
        print(f"{verdict:4s}  {name:{width}s}  {detail}")
    return 0 if all(verdict != "FAIL" for _, verdict, _ in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
