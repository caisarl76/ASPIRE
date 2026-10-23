"""Timed agent trials on the real G1, scored by the operator (the robot has no sim truth).

Start the robot's CAP server first (scripts/start_system2.sh --robot 222 --allow-motion
--engage). Each trial: the operator places the bottle and presses Enter, the agent runs
(`start_system2_agent.sh --robot 222 --allow-motion [--model M --effort E] exec <task>`), and
the operator scores what happened. The script records wall time, tool calls, tokens and the
bottle skill's REPORT lines from the codex session.

    .venv-system2/bin/python scripts/g1_real_agent_trials.py --label fast \\
        --model gpt-5.6-luna --effort low --trials 5 --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = Path.home() / ".codex" / "sessions"
TASK = (
    "Pick up the water bottle with the right hand, lift it 5 cm, put it back where it was, "
    "release it, then go Home."
)


def skill_reports(since: float) -> list[dict]:
    """REPORT lines of the bottle skill in codex sessions written since ``since``."""
    reports = []
    for session in SESSIONS.glob("*/*/*/*.jsonl"):
        if session.stat().st_mtime < since:
            continue
        for line in session.read_text(errors="replace").splitlines():
            if '"REPORT ' not in line and "REPORT {" not in line:
                continue
            record = json.loads(line).get("payload", {})
            output = record.get("output")
            for item in output if isinstance(output, list) else []:
                text = item.get("text", "")
                if text.startswith("REPORT "):
                    reports.append(json.loads(text[len("REPORT "):]))
    return reports


def ask(prompt: str) -> str:
    return input(prompt).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--label", required=True, help="name of this setup, e.g. baseline or fast")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robot", default="222")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    out = args.out / args.label
    out.mkdir(parents=True, exist_ok=True)
    options = [*(["--model", args.model] if args.model else []),
               *(["--effort", args.effort] if args.effort else [])]
    results = []
    for trial in range(1, args.trials + 1):
        ask(f"\n[{args.label} {trial}/{args.trials}] place the bottle at its spot, robot clear; Enter to run ")
        log_path = out / f"trial{trial:02d}.log"
        started_wall, started = time.time(), time.monotonic()
        with log_path.open("w") as log:
            try:
                subprocess.run(
                    ["scripts/start_system2_agent.sh", "--robot", args.robot, "--allow-motion",
                     *options, "exec", args.task],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.timeout_s,
                )
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
        duration = time.monotonic() - started
        text = log_path.read_text(errors="replace")
        tokens = re.search(r"tokens used\n([\d,]+)", text)
        print(f"  {duration:.0f} s; agent's last lines:\n    " + "\n    ".join(text.strip().splitlines()[-4:]))
        result = {
            "trial": trial,
            "duration_s": round(duration),
            "timed_out": timed_out,
            "tool_calls": len(re.findall(r"mcp: cap/\S+ started", text)),
            "tokens": int(tokens.group(1).replace(",", "")) if tokens else None,
            "skill_reports": skill_reports(started_wall),
            "lifted": ask("  did the bottle visibly rise at least 3 cm? [y/n] ").lower().startswith("y"),
            "stands": ask("  does it stand upright on the case now? [y/n] ").lower().startswith("y"),
            "placement_cm": float(ask("  distance from its start mark, cm (estimate): ") or "nan"),
            "note": ask("  note (optional): "),
        }
        result["success"] = result["lifted"] and result["stands"] and result["placement_cm"] <= 4.0
        (out / f"trial{trial:02d}.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    ok = [r for r in results if r["success"]]
    times = sorted(r["duration_s"] for r in results)
    summary = {"label": args.label, "model": args.model, "effort": args.effort,
               "succeeded": f"{len(ok)}/{len(results)}", "duration_s": times,
               "tokens": sorted(r["tokens"] or 0 for r in results), "trials": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{args.label}: {len(ok)}/{len(results)} succeeded, times {times} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
