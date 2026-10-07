"""Record experiment progress and warnings without controlling jobs."""

import json
import math
import os
import sys
import time
from pathlib import Path


def read_progress(run):
    path = run / "monitors/file/metrics.jsonl"
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        offset = max(0, stream.tell() - 2_000_000)
        stream.seek(offset)
        if offset:
            stream.readline()
        lines = stream.read().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


def main():
    job, *names = sys.argv[1:]
    if os.environ.get("SLURM_JOB_ID") != job or len(names) != 2:
        raise ValueError("Run inside the paired allocation with its job ID and both run directories")
    runs = [Path(name) for name in names]
    started = time.time()
    last_update = {run: started for run in runs}
    ready_at = {}
    completed = set()
    log = Path("experiments/score-centering/results/monitor") / f"guard-{job}.jsonl"
    while True:
        now = time.time()
        reasons = []
        for run in runs:
            for row in read_progress(run):
                if "watcher/policy_version" in row:
                    ready_at.setdefault(run, row["time"])
                if "optim/grad_norm" in row:
                    last_update[run] = max(last_update[run], row["time"])
                    if not math.isfinite(row["optim/grad_norm"]):
                        reasons.append(f"{run.name}: nonfinite gradient")
                    if row.get("step", 0) >= 400:
                        completed.add(run)
            if run in completed:
                continue
            if run in ready_at:
                idle = now - max(last_update[run], ready_at[run])
                if idle > 1200:
                    reasons.append(f"{run.name}: no optimizer update for {idle:.0f}s after serving became ready")
            elif now - started > 2700:
                reasons.append(f"{run.name}: startup exceeded 45 minutes")
        record = {
            "time": now,
            "job": job,
            "warnings": reasons,
            "last_update": {str(run): stamp for run, stamp in last_update.items()},
        }
        with log.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if reasons:
            print(json.dumps(record), flush=True)
        if len(completed) == len(runs):
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
