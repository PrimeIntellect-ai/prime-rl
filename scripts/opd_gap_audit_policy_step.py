#!/usr/bin/env python3
"""Fail closed unless one policy-training step has complete, finite artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

ALIGNED_FIELDS = (
    "position_ids",
    "loss_mask",
    "advantages",
    "inference_logprobs",
    "trainer_logprobs",
    "entropy",
    "mismatch_kl",
)
OPTIONAL_ALIGNED_FIELDS = (
    "ref_logprobs",
    "rl_weights",
    "ref_kl_weights",
    "ce_weights",
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SUCCESS_RE = re.compile(
    r"Step (?P<step>\d+) .*? Loss (?P<loss>[-+0-9.eE]+)"
    r" .*? Entropy (?P<entropy>[-+0-9.eE]+)"
    r" .*? Mismatch KL (?P<mismatch_kl>[-+0-9.eE]+)"
    r" .*? Grad\. Norm (?P<grad_norm>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("--expected-rows", type=int, default=8)
    parser.add_argument("--require-reward-variance", action="store_true")
    parser.add_argument("--require-nonzero-advantages", action="store_true")
    return parser.parse_args()


def numeric_reward(record: dict[str, Any]) -> float:
    rewards = record.get("rewards")
    assert isinstance(rewards, dict) and rewards, "rollout has no reward components"
    values = [value for value in rewards.values() if isinstance(value, (int, float))]
    assert values, "rollout has no numeric reward components"
    assert all(math.isfinite(float(value)) for value in values), "rollout reward is non-finite"
    return float(sum(values))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    run_dir = output_dir / "run_default"
    export_dir = run_dir / "token_exports" / f"step_{args.step}"
    rollout_file = run_dir / "rollouts" / f"step_{args.step}" / "train_rollouts.jsonl"
    broadcast_dir = run_dir / "broadcasts" / f"step_{args.step + 1}"
    trainer_log = output_dir / "logs" / "trainer.log"

    assert (export_dir / "STABLE").is_file(), f"missing stable token export: {export_dir}"
    export_files = sorted(export_dir.glob("rank_*.jsonl"))
    assert export_files, f"missing rank exports: {export_dir}"

    rows = 0
    tokens = 0
    nonzero_advantage_tokens = 0
    nonzero_loss_mask_tokens = 0
    for export_file in export_files:
        for line_number, line in enumerate(export_file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = f"{export_file.name}:{line_number}"
            token_ids = record.get("token_ids")
            assert isinstance(token_ids, list) and token_ids, f"{label}: missing token_ids"
            assert record.get("step") == args.step, f"{label}: wrong policy step"
            assert record.get("export_step") == args.step, f"{label}: wrong export step"
            assert record.get("run_id") == "run_default", f"{label}: wrong run_id"
            assert isinstance(record.get("env_name"), str) and record["env_name"], (
                f"{label}: missing env_name"
            )
            seq_len = len(token_ids)
            for field in ALIGNED_FIELDS:
                values = record.get(field)
                assert isinstance(values, list), f"{label}: {field} missing or null"
                assert len(values) == seq_len, (
                    f"{label}: {field} has {len(values)} values, expected {seq_len}"
                )
                assert np.isfinite(np.asarray(values)).all(), (
                    f"{label}: {field} contains non-finite values"
                )
            for field in OPTIONAL_ALIGNED_FIELDS:
                values = record.get(field)
                if values is None:
                    continue
                assert isinstance(values, list) and len(values) == seq_len, (
                    f"{label}: {field} is not aligned"
                )
                numeric = [value for value in values if value is not None]
                assert len(numeric) in (0, seq_len), f"{label}: {field} mixes null and numeric values"
                if numeric:
                    assert np.isfinite(np.asarray(numeric)).all(), (
                        f"{label}: {field} contains non-finite values"
                    )
            rows += 1
            tokens += seq_len
            nonzero_advantage_tokens += sum(float(value) != 0.0 for value in record["advantages"])
            nonzero_loss_mask_tokens += sum(float(value) != 0.0 for value in record["loss_mask"])

    assert rows == args.expected_rows, f"export rows {rows}, expected {args.expected_rows}"
    assert nonzero_loss_mask_tokens > 0, "loss mask has no nonzero tokens"
    if args.require_nonzero_advantages:
        assert nonzero_advantage_tokens > 0, "advantages have no nonzero tokens"

    rollouts = [
        json.loads(line)
        for line in rollout_file.read_text().splitlines()
        if line.strip()
    ]
    assert len(rollouts) >= args.expected_rows, (
        f"rollout rows {len(rollouts)}, expected at least {args.expected_rows}"
    )
    assert len(rollouts) % args.expected_rows == 0, (
        f"rollout rows {len(rollouts)} are not complete groups of {args.expected_rows}"
    )
    assert all(record.get("errors") == [] for record in rollouts), "rollout contains errors"
    assert any(record.get("stop_condition") == "agent_completed" for record in rollouts), (
        "all rollout attempts missed agent_completed"
    )
    rewards = [numeric_reward(record) for record in rollouts]
    if args.require_reward_variance:
        assert len(set(rewards)) > 1, "rollout group has uniform reward"

    assert (broadcast_dir / "STABLE").is_file(), f"missing stable adapter: {broadcast_dir}"
    assert (broadcast_dir / "adapter_config.json").is_file(), "missing adapter config"
    assert (broadcast_dir / "adapter_model.safetensors").is_file(), "missing adapter weights"

    clean_log = ANSI_RE.sub("", trainer_log.read_text(errors="replace"))
    matches = [
        match
        for match in SUCCESS_RE.finditer(clean_log)
        if int(match.group("step")) == args.step
    ]
    assert matches, f"missing trainer SUCCESS line for step {args.step}"
    metrics = {key: float(value) for key, value in matches[-1].groupdict().items() if key != "step"}
    assert all(math.isfinite(value) for value in metrics.values()), "trainer metric is non-finite"
    assert metrics["grad_norm"] > 0.0, "trainer gradient norm is not positive"

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "step": args.step,
                "rows": rows,
                "rollout_rows": len(rollouts),
                "tokens": tokens,
                "rewards": rewards,
                "nonzero_advantage_tokens": nonzero_advantage_tokens,
                "nonzero_loss_mask_tokens": nonzero_loss_mask_tokens,
                "trainer": metrics,
                "stable_adapter": str(broadcast_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
