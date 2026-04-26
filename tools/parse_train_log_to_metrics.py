"""
Parse `train_llm` / `train_run.log` style stdout into `training_metrics.json`.

Lines matched:
  Scenario 1849 | Loop 1: PENALTY (Reward: -4.0)
  Scenario 1999 | Loop 3: SUCCESS (Reward: 7.42)

Skips: HuggingFace warnings, DEBUG blocks, and non-matching lines.

Usage:
  python tools/parse_train_log_to_metrics.py --input train_run.log
  type train_run.log | python tools/parse_train_log_to_metrics.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, List

LINE_RE = re.compile(
    r"^Scenario (\d+) \| Loop (\d+): (PENALTY|SUCCESS) \(Reward: "
    r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?)\)\s*$"
)


def parse_log_lines(text: str) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    mode = "heuristic"
    for line in text.splitlines():
        s = line.strip()
        if "STARTING PHASE 2: ONLINE RL (COUNCIL)" in line:
            mode = "council"
        if "STARTING PHASE 2: ONLINE RL (HEURISTIC)" in line:
            mode = "heuristic"
        m = LINE_RE.match(s)
        if not m:
            continue
        scenario, loop, outcome, reward_s = m.groups()
        success = outcome.upper() == "SUCCESS"
        rows.append(
            {
                "scenario": int(scenario),
                "loop": int(loop),
                "mode": mode,
                "reward": float(reward_s),
                "loss": None,
                "success": success,
            }
        )
    if not rows:
        raise ValueError("No 'Scenario N | Loop M: ...' lines found.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert training log text to training_metrics.json (Reward/Success/Scenario rows)."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="-",
        help="Log file (default: stdin).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="training_metrics.json",
        help="Output path (default: training_metrics.json).",
    )
    args = parser.parse_args()

    if args.input in ("-", None):
        text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    rows = parse_log_lines(text)

    out_path = args.output
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    import os

    os.replace(tmp, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
