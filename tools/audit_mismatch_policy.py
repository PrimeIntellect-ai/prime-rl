"""Verify recorded generation, shipment, and trainer versions for a mismatch run."""

import argparse
import json
from collections import Counter
from pathlib import Path

from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk


def records(directory: Path):
    for number in sorted(chunk_numbers(directory)):
        with open_chunk(directory, number) as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                yield json.loads(line)


def audit(run_dir: Path) -> dict:
    base = run_dir / "monitors/file/traces"
    policies = {
        trace["id"]: episode["run"]["work"].get("policy")
        for episode in records(base / "stream")
        if episode.get("run", {}).get("work", {}).get("type") == "train"
        for trace in episode["traces"]
    }
    ships = {}
    for record in records(base / "annotations/orch"):
        if "ship" in record.get("info", {}):
            ships[record["trace_id"]] = record["info"]["ship"]["step"]
    trained = {}
    for record in records(base / "annotations/trainer"):
        if any(
            any(value is not None for value in branch.get("trainer_logprobs", []))
            for branch in record.get("branches", [])
        ):
            trained[record["trace_id"]] = record.get("info", {}).get("trainer")
    violations = []
    missing_trainer_versions = 0
    for trace_id, trainer in trained.items():
        step = ships.get(trace_id)
        policy = policies.get(trace_id)
        if step is None or policy != {"start": step - 1, "end": step - 1}:
            violations.append({"trace_id": trace_id, "ship_step": step, "policy": policy})
        if trainer is None:
            missing_trainer_versions += 1
        elif step is None or trainer != {"step": step, "policy_version": step - 1}:
            violations.append({"trace_id": trace_id, "ship_step": step, "trainer": trainer})
    return {
        "trained_traces": len(trained),
        "per_step": dict(sorted(Counter(ships.get(trace_id, -1) for trace_id in trained).items())),
        "missing_trainer_versions": missing_trainer_versions,
        "violations": violations,
        "verified": bool(trained) and not violations and not missing_trainer_versions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    (args.run_dir / "policy-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verified"] else 1)
