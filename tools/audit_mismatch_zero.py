"""Audit exact sampled logprob bits, policy versions, and complete step coverage."""

import argparse
import json
from pathlib import Path

import torch
import verifiers.v1 as vf
from audit_mismatch_policy import audit as audit_policy
from audit_mismatch_policy import records


def audit(run_dir: Path) -> dict:
    resolved = run_dir / "configs/latest/resolved"
    trainer = json.loads((resolved / "trainer.json").read_text())
    orchestrator = json.loads((resolved / "orchestrator.json").read_text())
    policy = audit_policy(run_dir)
    rows = [json.loads(line) for line in (run_dir / "monitors/file/metrics.jsonl").open()]
    metrics = [row for row in rows if "mismatch_k3_stable/all/mean" in row]
    expected_steps = list(range(1, trainer["max_steps"] + 1))
    actual_steps = [row["step"] for row in metrics]
    update_rows = [row for row in rows if "optim/router_probe_changed_elements" in row]
    update_probe = {
        "available": bool(update_rows),
        "complete": [row["step"] for row in update_rows] == expected_steps,
        "changed_steps": [row["step"] for row in update_rows if row["optim/router_probe_changed_elements"] > 0],
        "nonfinite_elements": sum(row["optim/router_probe_nonfinite_elements"] for row in update_rows),
    }
    update_probe["scored_updated_policy_steps"] = [
        step + 1 for step in update_probe["changed_steps"] if step + 1 in actual_steps
    ]
    update_probe["updated_policy_required"] = bool(update_rows) and trainer["optim"]["lr"] > 0
    update_probe["updated_policy_scored"] = bool(update_probe["scored_updated_policy_steps"])
    metric_keys = (
        "logprob_abs_error/all/max",
        "logprob_bit_mismatch/all/mean",
        "logprob_nonfinite/all/mean",
        "mismatch_k3_stable/all/mean",
    )
    metrics_zero = bool(metrics) and all(row.get(key) == 0 for row in metrics for key in metric_keys)
    full_precision = all(config["monitors"]["file"]["float_decimals"] is None for config in (trainer, orchestrator))
    base = run_dir / "monitors/file/traces"
    annotations = {
        (record["trace_id"], branch["index"]): branch["trainer_logprobs"]
        for record in records(base / "annotations/trainer")
        for branch in record.get("branches", [])
        if any(value is not None for value in branch.get("trainer_logprobs", []))
    }
    seen = set()
    sampled_tokens = bit_mismatches = nonfinite_pairs = long_position_tokens = 0
    max_sequence_length = 0
    max_abs_error = 0.0
    for record in records(base / "stream"):
        episode = vf.Episode.model_validate(record)
        for trace in episode.traces:
            for branch in trace.branches:
                key = (trace.id, branch.index)
                if key not in annotations or key in seen:
                    continue
                values = annotations[key]
                if len(values) != len(branch.logprobs):
                    raise ValueError(f"Trainer and sampling streams have different lengths for {key}")
                selected = [i for i, value in enumerate(values) if value is not None]
                train = torch.tensor([values[i] for i in selected], dtype=torch.float32)
                sample = torch.tensor([branch.logprobs[i] for i in selected], dtype=torch.float32)
                finite = torch.isfinite(train) & torch.isfinite(sample)
                bit_mismatches += ((train.view(torch.int32) != sample.view(torch.int32)) | ~finite).sum().item()
                nonfinite_pairs += (~finite).sum().item()
                if finite.any():
                    max_abs_error = max(max_abs_error, (train[finite] - sample[finite]).abs().max().item())
                sampled_tokens += len(selected)
                long_position_tokens += sum(i >= 593 for i in selected)
                max_sequence_length = max(max_sequence_length, len(values))
                seen.add(key)
    verified = (
        policy["verified"]
        and full_precision
        and metrics_zero
        and actual_steps == expected_steps
        and seen == set(annotations)
        and sampled_tokens > 0
        and bit_mismatches == 0
        and nonfinite_pairs == 0
        and (not update_rows or (update_probe["complete"] and update_probe["nonfinite_elements"] == 0))
        and (not update_probe["updated_policy_required"] or update_probe["updated_policy_scored"])
    )
    return {
        "verified": verified,
        "learning_rate": trainer["optim"]["lr"],
        "router_update_probe": update_probe,
        "expected_steps": expected_steps,
        "actual_steps": actual_steps,
        "full_precision_traces": full_precision,
        "metrics_zero": metrics_zero,
        "policy_audit": policy,
        "sampled_tokens": sampled_tokens,
        "bit_mismatches": bit_mismatches,
        "nonfinite_pairs": nonfinite_pairs,
        "max_abs_error": max_abs_error,
        "max_sequence_length": max_sequence_length,
        "sampled_tokens_at_positions_593_and_later": long_position_tokens,
        "missing_branches": len(set(annotations) - seen),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    (args.run_dir / "zero-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verified"] else 1)
