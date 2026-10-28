"""Run the System 2 agent on one task N times in the G1 sim and judge each run by sim truth.

Each run starts from a fresh sim (`g1_sim.sh down`, then `smoke --keep`, which also starts the
CAP server and drops the robot), lays out the case and bottle, records the sim truth, runs
`start_system2_agent.sh --robot sim --allow-motion exec <task>` and scores the bottle:

  lifted     it rose at least 3 cm at some point
  stands     it ends upright (cos tilt >= 0.9) on the case (within 2 cm of its start height)
  placement  how far it ends from its start, horizontally, in the world
  success    lifted, stands, and placement <= --max-placement-m

    .venv-system2/bin/python scripts/g1_sim/agent_batch.py --runs 5 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASK = (
    "Pick up the water bottle with the right hand, lift it 5 cm, put it back where it was, "
    "release it, then go Home."
)


def sh(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, **kw)


def score(truth: Path, agent_log: Path, max_placement_m: float) -> dict:
    rows = [json.loads(line) for line in truth.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "aspire_bottle_world" in r]
    start, end = rows[0]["aspire_bottle_world"], rows[-1]["aspire_bottle_world"]
    rise = max(r["aspire_bottle_world"][2] for r in rows) - start[2]
    upright = rows[-1].get("aspire_bottle_upright", 1.0)
    placement = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    stands = upright >= 0.9 and abs(end[2] - start[2]) < 0.02
    log = agent_log.read_text(errors="replace")
    tokens = re.search(r"tokens used\n([\d,]+)", log)
    return {
        "lifted": rise >= 0.03,
        "stands": stands,
        "placement_m": round(placement, 4),
        "rise_m": round(rise, 4),
        "min_upright": round(min(r.get("aspire_bottle_upright", 1.0) for r in rows), 3),
        "success": rise >= 0.03 and stands and placement <= max_placement_m,
        "tool_calls": len(re.findall(r"mcp: cap/\S+ started", log)),
        "tokens": int(tokens.group(1).replace(",", "")) if tokens else None,
    }


def run_once(
    index: int,
    out: Path,
    task: str,
    timeout_s: float,
    max_placement_m: float,
    bottle_xy,
    agent_options: list[str],
) -> dict:
    run = out / f"run{index:02d}"
    run.mkdir(parents=True, exist_ok=True)
    sh("scripts/g1_sim.sh", "down")
    smoke = sh("scripts/g1_sim.sh", "smoke", "--keep")
    (run / "smoke.log").write_text(smoke.stdout + smoke.stderr)
    if smoke.returncode != 0:
        return {"run": index, "error": "smoke failed"}
    sh("scripts/g1_sim.sh", "reset", *(str(v) for v in bottle_xy or ()))
    time.sleep(3.0)
    python = str(ROOT / ".venv-system2/bin/python")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    watch = subprocess.Popen(
        [python, "scripts/g1_sim/experiments.py", "watch", "--out", str(run / "watch.jsonl")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    started = time.monotonic()
    agent_env = {k: v for k, v in env.items() if k != "__EGL_VENDOR_LIBRARY_DIRS"}
    with (run / "agent.log").open("w") as log:
        try:
            subprocess.run(
                [
                    "scripts/start_system2_agent.sh",
                    "--robot",
                    "sim",
                    "--allow-motion",
                    *agent_options,
                    "exec",
                    task,
                ],
                cwd=ROOT,
                env=agent_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
    duration = time.monotonic() - started
    watch.terminate()  # every line is flushed as it is written
    watch.wait(timeout=10)
    if "Selected model is at capacity" in (run / "agent.log").read_text(errors="replace"):
        return {"run": index, "error": "model at capacity"}
    result = {"run": index, "duration_s": round(duration), "timed_out": timed_out}
    result.update(score(run / "watch.jsonl", run / "agent.log", max_placement_m))
    (run / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--max-placement-m", type=float, default=0.04)
    parser.add_argument(
        "--bottle-xy", type=float, nargs=2, help="bottle position on the case (torso frame)"
    )
    parser.add_argument("--model", help="codex model (default: ~/.codex/config.toml)")
    parser.add_argument("--effort", help="codex reasoning effort (default: the config's)")
    args = parser.parse_args()
    results = []
    for index in range(1, args.runs + 1):
        options = [*(["--model", args.model] if args.model else []),
                   *(["--effort", args.effort] if args.effort else [])]
        result = run_once(
            index,
            args.out,
            args.task,
            args.timeout_s,
            args.max_placement_m,
            args.bottle_xy,
            options,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    sh("scripts/g1_sim.sh", "down")
    done = [r for r in results if "error" not in r]
    print(
        f"\n{sum(r['success'] for r in done)}/{len(done)} succeeded "
        f"(lifted {sum(r['lifted'] for r in done)}, stands {sum(r['stands'] for r in done)}); "
        f"placement {sorted(round(r['placement_m'] * 100, 1) for r in done)} cm; "
        f"{len(results) - len(done)} infrastructure failures"
    )
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
