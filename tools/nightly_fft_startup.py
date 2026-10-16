"""Verify first hosted training progress, not merely allocation or submission."""

import argparse
import json
import math
import signal
import subprocess
import time
from pathlib import Path

ACTIVE_STATUSES = {"QUEUED", "PENDING", "CREATING", "PULLING_IMAGE", "RUNNING", "TERMINATING"}


def cli_json(args, remaining):
    result = subprocess.run(["prime", "train", *args], capture_output=True, text=True, timeout=remaining)
    if result.returncode:
        raise ValueError("Training lookup failed; raw response withheld")
    return json.loads(result.stdout)


def inspect_run(record, remaining, read=cli_json):
    deadline = time.monotonic() + remaining
    run = read(["get", record["run_id"], "--output", "json"], remaining)["run"]
    if not isinstance(run, dict) or run["id"] != record["run_id"] or run["name"] != record["name"]:
        raise ValueError("Hosted identity mismatch")
    status = run["status"]
    if status in {"FAILED", "STOPPED"}:
        return status, None
    if status not in ACTIVE_STATUSES | {"COMPLETED"}:
        raise ValueError("Unknown hosted status")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Observation budget expired")
    progress = read(["progress", record["run_id"]], remaining)
    if not isinstance(progress, dict):
        raise ValueError("Invalid progress response")
    step = progress.get("latest_step")
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("Invalid training progress")
    return status, step


def monitor(path, budget, interval, lookup=inspect_run, clock=time.monotonic, wait=time.sleep):
    record = json.loads(path.read_text())
    deadline = clock() + budget
    last_status = None

    def finish(outcome, message, code):
        record["outcome"] = outcome
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(message)
        if code:
            print(f"Inspect: prime train get {record['run_id']} --output json")
            print("No stop request was sent; queued/running training may continue.")
        return code

    try:
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                return finish(
                    "startup_unverified",
                    f"Observation expired (last status: {last_status}); queue wait is not trainer startup failure.",
                    1,
                )
            status, step = lookup(record, remaining)
            if clock() >= deadline:
                return finish("startup_unverified", "Observation expired before startup could be verified.", 1)
            last_status = status
            record["status"] = status
            record["latest_step"] = step
            if status in {"FAILED", "STOPPED"}:
                return finish("training_failed", f"Training ended {status} before startup acceptance.", 1)
            if step is not None and step >= 1 and status in {"RUNNING", "COMPLETED"}:
                return finish(
                    "started",
                    f"Observed training progress step {step}; startup smoke passed, not convergence/full training.",
                    0,
                )
            if status == "COMPLETED":
                return finish(
                    "startup_unverified", "COMPLETED without first training progress; startup not verified.", 1
                )
            record["outcome"] = "observing_startup"
            path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"Waiting for first training progress: {status}", flush=True)
            wait(min(interval, max(0, deadline - clock())))
    except KeyboardInterrupt:
        return finish("observation_detached", "Observation interrupted; startup not verified.", 1)
    except (ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        return finish("startup_unverified", "Could not verify hosted startup; raw response withheld.", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, required=True)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.timeout_seconds, args.poll_seconds)):
        parser.error("Positive finite timeout and interval required")

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    return monitor(args.result, args.timeout_seconds, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
