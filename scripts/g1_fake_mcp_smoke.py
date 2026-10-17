"""Off-robot MCP smoke for the g1_real System 2 server.

Start the server with ``--fake`` (``scripts/start_system2.sh --benchmark g1_real
--station g1-lab --fake --allow-motion --engage --no-viser --port 8223``), then run
``.venv-system2/bin/python scripts/g1_fake_mcp_smoke.py --port 8223``.
It homes, nudges the right hand 5 cm forward in the torso frame, closes the left
hand, and exits non-zero on any unexpected result. The same script can be pointed at
the real server later (it moves the real robot: only run it with the operator ready).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import yaml

PROFILE = (
    Path(__file__).resolve().parents[1] / "src/cap_harness/configs/g1_real/g1-lab/station.yaml"
)
HAND_TOOLS = {
    "cap_g1_real__get_hand_state",
    "cap_g1_real__set_hand_joints",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def find(data, key):
    """First value stored under ``key`` anywhere in nested job/result dicts."""
    if isinstance(data, dict):
        if key in data:
            return data[key]
        data = list(data.values())
    if isinstance(data, list):
        for item in data:
            if (hit := find(item, key)) is not None:
                return hit
    return None


async def call(session: ClientSession, name: str, args: dict) -> dict:
    result = await session.call_tool(name, args)
    text = next(c.text for c in result.content if c.type == "text")
    if result.isError:
        fail(f"{name} returned an error: {text[:500]}")
    data = json.loads(text)
    print(f"{name} -> {json.dumps(data, default=str)[:700]}")
    return data


async def main(port: int) -> None:
    url = f"http://127.0.0.1:{port}/mcp/"
    async with streamablehttp_client(url) as (r, w, _), ClientSession(r, w) as session:
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        print("tools:", sorted(names))
        if not names >= HAND_TOOLS or "cap_move_to_pose" not in names:
            fail("missing tools")
        await call(session, "get_capability_graph", {})
        home = await call(session, "cap_program_go_home", {})
        if find(home, "home_verified") is not True:
            fail("home_verified is not true")
        state = await call(session, "cap_get_robot_state", {})
        right = state["result"]["end_effector_poses"]["right"]
        x, y, z = right["position"]
        move = await call(
            session,
            "cap_move_to_pose",
            {
                "arm": "right",
                "target_pose": {
                    "position": [x + 0.05, y, z],
                    "quaternion_wxyz": right["quaternion_wxyz"],
                    "frame": "torso",
                },
                "strategy": {"ik_solver": "mink", "trajectory_planner": "interpolation"},
            },
        )
        # g1_real move_to_pose reports task-space errors, not joint_residual_rad (that is go_home's).
        pos_err = find(move, "position_error_m")
        rot_err = find(move, "orientation_error_rad")
        converged = find(move, "joint_target_converged")
        if find(move, "ok") is not True or not converged or pos_err is None or rot_err is None:
            fail("move_to_pose not ok/converged")
        if pos_err >= 0.01 or rot_err >= 0.10:
            fail(f"move_to_pose errors too large: {pos_err} m, {rot_err} rad")
        await call(session, "cap_g1_real__get_hand_state", {})
        closed = yaml.safe_load(PROFILE.read_text())["hands"]["left"]["closed"]
        await call(session, "cap_g1_real__set_hand_joints", {"arm": "left", "joints": closed})
        final = await call(session, "cap_get_robot_state", {})
        grip = final["result"]["gripper_positions"]["left"]
        print("left gripper_positions:", grip)
        if grip >= 0.5:
            fail("left gripper did not close")
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8222)
    asyncio.run(main(ap.parse_args().port))
