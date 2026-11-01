
import sys

import knobs

from datetime import datetime
from pathlib import Path

from agents import Run
from video_processor import Spinner


knobs.apply()


# -----------------------------------------
# YOUR CODE: run phase 5, the agents, on the output of phases 1 to 4
# usage: python run_agents.py [--no-gpu] [--force]
# It saves after every segment, so after a power or internet cut just run it again.
# -----------------------------------------

run = Run(use_gpu="--no-gpu" not in sys.argv)

with Spinner("Starting the agents") as spinner:

    def on_step(done, total):
        spinner.text = f"Agents on segment {min(done + 1, total)}/{total}"
        spinner.log(f"[{done}/{total}] {run.results[-1]['summary']}")

    summary = run.run(force="--force" in sys.argv, on_step=on_step)

print()
print(summary["final"])

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log = log_dir / f"agent-summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
log.write_text(summary["final"] + "\n", encoding="utf-8")

print(f"\nLog saved: {log}")
print("Saved: processed/agent/summary.json")
