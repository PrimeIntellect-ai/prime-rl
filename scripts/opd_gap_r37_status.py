#!/usr/bin/env python3
"""Publish machine-readable status for the matched General Agent r37 fleet."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STATUS_DIR = Path("/home/ubuntu/prime-intellect/projects/opd-gap/status")
DATA_DIR = STATUS_DIR / "data"
AUXILIARY_STATE_FILES = (
    Path("/home/ubuntu/opd-gap-r37-eval-pod-attempt1-state.json"),
    Path("/home/ubuntu/opd-gap-r37-eval-pod-state.json"),
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GROUP_RE = re.compile(
    r"Finished group \| env=genagent-train task_idx=(?P<task>\d+) \| "
    r"rollouts=(?P<rollouts>\d+) \(errored=(?P<errored>\d+), filtered=(?P<filtered>\d+)\) "
    r"\| reward=(?P<reward>-?[0-9.]+) \| filters: (?P<filters>.*?)\s+\[train_sink"
)
STEP_RE = re.compile(
    r"SUCCESS.*?Step\s+(?P<step>\d+)\s+\|\s+"
    r"(?P<duration>(?:(?P<hours>\d+)h\s*)?(?:(?P<minutes>\d+)m\s*)?(?:(?P<seconds>\d+)s)?)\s+"
    r"\|\s+Loss\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+Entropy\s+(?P<entropy>[-+0-9.eE]+)\s+"
    r"\|\s+Mismatch KL\s+(?P<mismatch_kl>[-+0-9.eE]+)\s+"
    r"\|\s+Grad\. Norm\s+(?P<grad_norm>[-+0-9.eE]+)\s+"
    r"\|\s+LR\s+(?P<lr>[-+0-9.eE]+)\s+"
    r"\|\s+Throughput\s+(?P<throughput>[0-9.]+)\s+tokens/s\s+"
    r"\|\s+MFU\s+(?P<mfu>[0-9.]+)%\s+\|\s+Peak Mem\.\s+(?P<peak_mem>[0-9.]+)\s+GiB"
)
ROLLOUT_STEP_RE = re.compile(
    r"SUCCESS.*?Step\s+(?P<step>\d+)\s+\|\s+"
    r"(?:(?P<hours>\d+)h\s*)?(?:(?P<minutes>\d+)m\s*)?(?:(?P<seconds>\d+)s)?\s+"
    r"\|\s+Reward\s+(?P<reward>[-+0-9.eE]+)\s+"
    r"\|\s+Trainable\s+(?P<trainable>\d+)/(?P<rollouts>\d+)\s+\((?P<trainable_pct>[0-9.]+)%\)\s+"
    r"\|\s+Turns\s+(?P<turns>[0-9.]+)\s+\|\s+Branches\s+(?P<branches>[0-9.]+)\s+"
    r"\|\s+Max Off-Policy\s+(?P<max_off_policy>\d+)\s+"
    r"\|\s+Error\s+(?P<error_pct>[0-9.]+)%\s+\|\s+Truncation\s+(?P<truncation_pct>[0-9.]+)%"
)
NATIVE_EVAL_RE = re.compile(r".*/step_(?P<step>\d+)/eval_rollouts_genagent-heldout\.jsonl$")
EVAL_SUCCESS_RE = re.compile(
    r"Evaluated genagent-heldout \(Step (?P<step>\d+)\) \| Policy v(?P<policy>\d+)"
)
EVAL_MIXED_RE = re.compile(
    r"Eval genagent-heldout step (?P<step>\d+) had mixed policy versions: "
    r"(?P<versions>\[[^\]]*\])"
)
EXPECTED_HELDOUT_TASKS = (
    "soap_making_t4",
    "incense_crafting_t3",
    "car_show_t3",
    "timber_t3",
    "bbq_competition_t3",
    "textile_mill_t4",
    "vertical_farm_t3",
    "fireworks_show_t4",
    "party_rental_t4",
    "wine_blend_t3",
    "space_debris_t4",
    "freelance_platform_t3",
    "sewing_workshop_t3",
    "chocolate_boutique_t3",
    "science_fair_t4",
    "fulfillment_center_t4",
    "film_archive_t4",
    "fermentation_lab_t3",
    "coffee_cupping_t3",
    "cable_car_t4",
)

RUN_LABELS = {
    "grpo": {
        "label": "GRPO | no hint",
        "display_name": "General Agent | GRPO | hint: none | Qwen3.5-35B-A3B | train 8 variance-qualified | group 8 | 100 steps",
        "objective": "grpo",
        "hint_style": "none",
    },
    "opsd": {
        "label": "OPSD (1-token, top-64) | full validated answer",
        "display_name": "General Agent | OPSD 1-token top-64 | hint: full validated answer | Qwen3.5-35B-A3B | train 8 variance-qualified | group 8 | 100 steps",
        "objective": "opsd",
        "hint_style": "full_validated_answer",
    },
    "rlsd": {
        "label": "RLSD | full validated answer",
        "display_name": "General Agent | RLSD | hint: full validated answer | Qwen3.5-35B-A3B | train 8 variance-qualified | group 8 | 100 steps",
        "objective": "rlsd",
        "hint_style": "full_validated_answer",
    },
    "rlcsd": {
        "label": "RLCSD | successful sibling trajectory",
        "display_name": "General Agent | RLCSD | hint: successful sibling trajectory | Qwen3.5-35B-A3B | train 8 variance-qualified | group 8 | 100 steps",
        "objective": "rlcsd",
        "hint_style": "successful_sibling_trajectory",
    },
}


@dataclass(frozen=True)
class Run:
    arm: str
    placement: str
    output_dir: str
    wandb: str
    job: str | None = None
    hourly_cost_usd: float | None = None
    billing_started_at_utc: str | None = None


RUNS = (
    Run(
        "grpo",
        "ar:gpu-013",
        "/home/tim/prime-rl/outputs-genagent-grpo-p8var8lt-tp4-r37-full100-20260714",
        "https://wandb.ai/primeintellect/opd-gap/runs/a655cddaabc948b7b06731694c14e85c",
        "214",
    ),
    Run(
        "opsd",
        "ar:gpu-014",
        "/home/tim/prime-rl/outputs-genagent-opsd-goldcalls-d64-p8var8lt-tp4-r37-full100-20260714",
        "https://wandb.ai/primeintellect/opd-gap/runs/fc07d4bed1b24c219ff9818d5727d4d4",
        "215",
    ),
    Run(
        "rlsd",
        "prime-pod:a694aaa2a4194bcda06e72a20d7ce9a5:8xA100-80GB",
        "/home/ubuntu/prime-rl/outputs-genagent-rlsd-goldcalls-d64-p8var8lt-tp4-pod-r37-full100-20260714",
        "https://wandb.ai/primeintellect/opd-gap/runs/eddb212255de4f549b639af7690f7855",
        hourly_cost_usd=22.32,
        billing_started_at_utc="2026-07-14T17:08:11+00:00",
    ),
    Run(
        "rlcsd",
        "prime-pod:991e947ff96047e6bea676060899f625:8xH200",
        "/root/prime-rl/outputs-genagent-rlcsd-d64-p8var8lt-tp4-pod-r37-full100-20260714",
        "https://wandb.ai/primeintellect/opd-gap/runs/01632e97fa2c4edbbd61992a041c25c9",
        hourly_cost_usd=28.0,
        billing_started_at_utc="2026-07-14T15:20:11+00:00",
    ),
)


def command(argv: list[str], *, timeout: int = 30) -> str:
    result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)
    return result.stdout


def remote(run: Run, script: str) -> str:
    if run.arm == "rlcsd":
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            "/home/ubuntu/.ssh/primeintellect_ed25519",
            "-p",
            "30414",
            "root@103.196.86.6",
            "bash -lc " + shlex.quote(script),
        ]
    elif run.arm == "rlsd":
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            "/home/ubuntu/.ssh/primeintellect_ed25519",
            "ubuntu@132.145.177.209",
            "bash -lc " + shlex.quote(script),
        ]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "ar", "bash -lc " + shlex.quote(script)]
    return command(argv)


def clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def queue_state(run: Run) -> tuple[str, str]:
    if run.job:
        line = remote(run, f"squeue -h -j {run.job} -o '%T|%R' || true").strip().splitlines()
        if not line or "|" not in line[-1]:
            return "ABSENT", ""
        return tuple(line[-1].split("|", 1))  # type: ignore[return-value]
    pid_file = "/root/rlcsd-r37-full100.pid" if run.arm == "rlcsd" else "/home/ubuntu/rlsd-r37-full100.pid"
    alive = remote(
        run,
        f"test -f {pid_file} && kill -0 $(cat {pid_file}) 2>/dev/null && echo RUNNING || echo ABSENT",
    )
    return (alive.strip().splitlines()[-1] if alive.strip() else "SSH_ERROR"), ""


def stable_max(run: Run, kind: str) -> int | None:
    script = (
        f"find {shlex.quote(run.output_dir)}/run_default/{kind} -mindepth 2 -maxdepth 2 "
        "-type f -name STABLE 2>/dev/null | sed -nE 's#.*/step_([0-9]+)/STABLE#\\1#p' | sort -n | tail -1"
    )
    value = remote(run, script).strip().splitlines()
    return int(value[-1]) if value and value[-1].isdigit() else None


def read_log(run: Run, name: str, pattern: str) -> str:
    path = f"{run.output_dir}/logs/{name}"
    return clean(remote(run, f"grep -E {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null"))


def _native_eval_payload_code(output_dir: str) -> str:
    """Build remote structured parsing code while keeping large node payloads off SSH."""
    code = f"""
import glob, json
payload = {{'posthoc': [], 'native': []}}
for path in glob.glob({str(Path(output_dir) / 'posthoc_evals' / 'step_*' / 'summary.json')!r}):
    try:
        row = json.load(open(path))
    except (OSError, ValueError):
        continue
    payload['posthoc'].append({{'path': path, 'summary': row}})
for path in glob.glob({str(Path(output_dir) / 'run_default' / 'rollouts' / 'step_*' / 'eval_rollouts_genagent-heldout.jsonl')!r}):
    records = []
    malformed_rows = 0
    try:
        lines = open(path).read().splitlines()
    except OSError:
        continue
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            malformed_rows += 1
            continue
        nodes = row.get('nodes') if isinstance(row.get('nodes'), list) else None
        records.append({{
            'id': row.get('id'),
            'task': {{
                'idx': (row.get('task') or {{}}).get('idx'),
                'name': (row.get('task') or {{}}).get('name'),
            }} if isinstance(row.get('task'), dict) else None,
            'rewards': row.get('rewards'),
            'errors': row.get('errors') if 'errors' in row else None,
            'errors_present': 'errors' in row,
            'stop_condition': row.get('stop_condition') if 'stop_condition' in row else None,
            'stop_condition_present': 'stop_condition' in row,
            'nodes': [
                {{'sampled': node.get('sampled'), 'finish_reason': node.get('finish_reason')}}
                for node in (nodes or []) if isinstance(node, dict)
            ] if nodes is not None else None,
            'policy_version': row.get('policy_version', (row.get('info') or {{}}).get('policy_version')),
            'eval_step': row.get('eval_step', (row.get('info') or {{}}).get('eval_step')),
        }})
    payload['native'].append({{'path': path, 'records': records, 'malformed_rows': malformed_rows}})
print(json.dumps(payload))
"""
    return code


def _error_label(error: object) -> str:
    if isinstance(error, dict):
        return str(error.get("type") or error.get("name") or error.get("error") or "unknown")
    return error if isinstance(error, str) else type(error).__name__


def summarize_native_eval(
    records: list[dict[str, object]],
    *,
    step: int,
    source_path: str,
    eval_log: str,
    malformed_rows: int = 0,
) -> dict[str, object]:
    expected_indices = set(range(len(EXPECTED_HELDOUT_TASKS)))
    expected_identities = set(EXPECTED_HELDOUT_TASKS)
    task_indices: list[int] = []
    task_identities: list[str] = []
    policy_versions: list[int] = []
    eval_steps: list[int] = []
    rewards: list[float] = []
    error_rows = 0
    error_types: Counter[str] = Counter()
    stop_conditions: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    turns: list[int] = []
    truncated_rows = 0
    timeout_rows = 0
    all_errors_available = bool(records)
    all_termination_available = bool(records)
    all_nodes_available = bool(records)

    for row in records:
        task = row.get("task")
        if isinstance(task, dict):
            task_idx = task.get("idx")
            task_name = task.get("name")
            if isinstance(task_idx, int):
                task_indices.append(task_idx)
            if isinstance(task_name, str) and task_name:
                task_identities.append(task_name)

        policy_version = row.get("policy_version")
        if isinstance(policy_version, int):
            policy_versions.append(policy_version)
        eval_step = row.get("eval_step")
        if isinstance(eval_step, int):
            eval_steps.append(eval_step)

        reward_map = row.get("rewards")
        solved = reward_map.get("solved") if isinstance(reward_map, dict) else None
        if isinstance(solved, (int, float)) and not isinstance(solved, bool):
            rewards.append(float(solved))

        errors_present = bool(row.get("errors_present", "errors" in row))
        errors = row.get("errors")
        if not errors_present or not isinstance(errors, list):
            all_errors_available = False
        else:
            error_rows += int(bool(errors))
            error_types.update(_error_label(error) for error in errors)

        stop_present = bool(row.get("stop_condition_present", "stop_condition" in row))
        stop_condition = row.get("stop_condition")
        if not stop_present or not isinstance(stop_condition, str):
            all_termination_available = False
        else:
            stop_conditions[stop_condition] += 1
            timeout_rows += int(stop_condition in {"timeout", "harness_timeout"})

        nodes = row.get("nodes")
        if not isinstance(nodes, list):
            all_nodes_available = False
            continue
        row_turns = 0
        row_truncated = stop_condition in {"timeout", "harness_timeout", "max_tokens"}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("sampled"):
                row_turns += 1
            finish_reason = node.get("finish_reason")
            if isinstance(finish_reason, str):
                finish_reasons[finish_reason] += 1
                row_truncated |= finish_reason == "length"
        turns.append(row_turns)
        truncated_rows += int(row_truncated)

    success_policies = {
        int(match["policy"])
        for match in EVAL_SUCCESS_RE.finditer(eval_log)
        if int(match["step"]) == step
    }
    mixed_matches = [match for match in EVAL_MIXED_RE.finditer(eval_log) if int(match["step"]) == step]
    mixed_warning = bool(mixed_matches)
    warned_policy_versions: set[int] = set()
    for match in mixed_matches:
        try:
            warned = json.loads(match["versions"])
        except json.JSONDecodeError:
            continue
        if isinstance(warned, list):
            warned_policy_versions.update(version for version in warned if isinstance(version, int))
    row_policy_set = set(policy_versions)
    row_eval_step_set = set(eval_steps)
    resolved_policy_versions = row_policy_set if len(policy_versions) == len(records) else success_policies
    mixed_evidence_available = bool(policy_versions or success_policies) or mixed_warning
    mixed_policy_versions = (
        mixed_warning or len(row_policy_set) > 1 or len(success_policies) > 1
        if mixed_evidence_available
        else None
    )

    admission_errors: list[str] = []
    if malformed_rows:
        admission_errors.append("malformed_json_rows")
    if len(records) != len(EXPECTED_HELDOUT_TASKS):
        admission_errors.append("incomplete_row_count")
    if len(task_indices) != len(records) or len(set(task_indices)) != len(task_indices):
        admission_errors.append("missing_or_duplicate_task_indices")
    if len(task_identities) != len(records) or len(set(task_identities)) != len(task_identities):
        admission_errors.append("missing_or_duplicate_task_identities")
    if set(task_indices) != expected_indices or set(task_identities) != expected_identities:
        admission_errors.append("heldout_task_set_mismatch")
    if eval_steps and (len(eval_steps) != len(records) or row_eval_step_set != {step}):
        admission_errors.append("eval_step_mismatch")
    elif not eval_steps and not success_policies:
        admission_errors.append("eval_step_not_corroborated")
    if mixed_policy_versions is True:
        admission_errors.append("mixed_policy_versions")
    elif policy_versions and len(policy_versions) != len(records):
        admission_errors.append("policy_version_incomplete")
    elif row_policy_set and success_policies and row_policy_set != success_policies:
        admission_errors.append("policy_version_mismatch")
    elif len(resolved_policy_versions) != 1:
        admission_errors.append("policy_version_not_pinned")

    complete = not any(
        error
        in {
            "malformed_json_rows",
            "incomplete_row_count",
            "missing_or_duplicate_task_indices",
            "missing_or_duplicate_task_identities",
            "heldout_task_set_mismatch",
        }
        for error in admission_errors
    )
    policy_step = (
        next(iter(resolved_policy_versions))
        if mixed_policy_versions is False and len(resolved_policy_versions) == 1
        else None
    )
    all_rewards_available = len(rewards) == len(records) and bool(records)
    return {
        "source": "native_in_process",
        "source_path": source_path,
        "eval_step": step,
        "policy_step": policy_step,
        "policy_versions": sorted(row_policy_set | success_policies | warned_policy_versions),
        "mixed_policy_versions": mixed_policy_versions,
        "expected_rows": len(EXPECTED_HELDOUT_TASKS),
        "completed_rows": len(records),
        "malformed_rows": malformed_rows,
        "complete": complete,
        "admitted": not admission_errors,
        "admission_errors": admission_errors,
        "task_indices": sorted(task_indices),
        "task_identities": sorted(task_identities),
        "reward_mean_all_rows": sum(rewards) / len(rewards) if all_rewards_available else None,
        "positive_reward_rows": sum(reward > 0 for reward in rewards) if all_rewards_available else None,
        "zero_reward_rows": sum(reward == 0 for reward in rewards) if all_rewards_available else None,
        "error_rows": error_rows if all_errors_available else None,
        "error_types": dict(error_types) if all_errors_available else None,
        "timeout_rows": timeout_rows if all_termination_available else None,
        "termination_reasons": dict(stop_conditions) if all_termination_available else None,
        "stop_conditions": dict(stop_conditions) if all_termination_available else None,
        "truncated_rows": truncated_rows if all_termination_available and all_nodes_available else None,
        "turns_mean": sum(turns) / len(turns) if all_nodes_available and turns else None,
        "turns_min": min(turns) if all_nodes_available and turns else None,
        "turns_max": max(turns) if all_nodes_available and turns else None,
        "turns_histogram": dict(Counter(str(value) for value in turns)) if all_nodes_available else None,
        "finish_reasons": dict(finish_reasons) if all_nodes_available else None,
    }


def normalize_posthoc_eval(summary: dict[str, object], source_path: str) -> dict[str, object]:
    row = dict(summary)
    row.pop("task_results", None)
    row["source"] = "posthoc"
    row["source_path"] = source_path
    row["admitted"] = bool(row.get("complete"))
    row["mixed_policy_versions"] = None
    return row


def merge_eval_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Expose at most one result per policy step while retaining source provenance."""
    selected: dict[object, dict[str, object]] = {}
    unkeyed: list[dict[str, object]] = []
    for row in rows:
        if not row.get("admitted"):
            unkeyed.append(row)
            continue
        key = row.get("policy_step")
        if not isinstance(key, int):
            unkeyed.append(row)
            continue
        current = selected.get(key)
        rank = (bool(row.get("admitted")), row.get("source") == "native_in_process")
        current_rank = (
            (bool(current.get("admitted")), current.get("source") == "native_in_process")
            if current
            else (False, False)
        )
        if current is None or rank > current_rank:
            if current is not None:
                row["alternative_sources"] = sorted(
                    {str(current.get("source")), *map(str, current.get("alternative_sources") or [])}
                )
            selected[key] = row
        else:
            current["alternative_sources"] = sorted(
                {str(row.get("source")), *map(str, current.get("alternative_sources") or [])}
            )
    return sorted(
        [*selected.values(), *unkeyed],
        key=lambda row: (int(row.get("eval_step") or row.get("policy_step") or -1), str(row.get("source_path"))),
    )


def read_eval_summaries(run: Run, eval_log: str) -> list[dict[str, object]]:
    code = _native_eval_payload_code(run.output_dir)
    output = remote(run, f"python3 -c {shlex.quote(code)}")
    try:
        value = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return []
    if not isinstance(value, dict):
        return []
    rows: list[dict[str, object]] = []
    for item in value.get("posthoc", []):
        if isinstance(item, dict) and isinstance(item.get("summary"), dict):
            rows.append(normalize_posthoc_eval(item["summary"], str(item.get("path") or "")))
    for item in value.get("native", []):
        if not isinstance(item, dict) or not isinstance(item.get("records"), list):
            continue
        match = NATIVE_EVAL_RE.match(str(item.get("path") or ""))
        if not match:
            continue
        rows.append(
            summarize_native_eval(
                item["records"],
                step=int(match["step"]),
                source_path=str(item["path"]),
                eval_log=eval_log,
                malformed_rows=int(item.get("malformed_rows") or 0),
            )
        )
    return merge_eval_summaries(rows)


def parse_groups(text: str) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    accepted_step = 0
    for line in text.splitlines():
        match = GROUP_RE.search(line)
        if not match:
            continue
        row: dict[str, object] = {
            "group_index": len(groups),
            "task_idx": int(match["task"]),
            "rollouts": int(match["rollouts"]),
            "errored": int(match["errored"]),
            "filtered": int(match["filtered"]),
            "reward": float(match["reward"]),
            "filters": "" if match["filters"] in {"-", "\u2014"} else match["filters"],
        }
        accepted = int(match["filtered"]) < int(match["rollouts"])
        row["accepted"] = accepted
        row["optimizer_step"] = accepted_step if accepted else None
        if accepted:
            accepted_step += 1
        groups.append(row)
    return groups


def parse_steps(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "step": int(match["step"]),
                "duration_s": (
                    int(match["hours"] or 0) * 3600
                    + int(match["minutes"] or 0) * 60
                    + int(match["seconds"] or 0)
                ),
                "loss": float(match["loss"]),
                "entropy": float(match["entropy"]),
                "mismatch_kl": float(match["mismatch_kl"]),
                "grad_norm": float(match["grad_norm"]),
                "lr": float(match["lr"]),
                "throughput_tokens_s": float(match["throughput"]),
                "mfu_pct": float(match["mfu"]),
                "peak_mem_gib": float(match["peak_mem"]),
            }
        )
    return rows


def parse_rollout_steps(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        match = ROLLOUT_STEP_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "step": int(match["step"]),
                "duration_s": (
                    int(match["hours"] or 0) * 3600
                    + int(match["minutes"] or 0) * 60
                    + int(match["seconds"] or 0)
                ),
                "reward": float(match["reward"]),
                "trainable_rollouts": int(match["trainable"]),
                "rollouts": int(match["rollouts"]),
                "trainable_pct": float(match["trainable_pct"]),
                "turns": float(match["turns"]),
                "branches": float(match["branches"]),
                "max_off_policy": int(match["max_off_policy"]),
                "error_pct": float(match["error_pct"]),
                "truncation_pct": float(match["truncation_pct"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue())


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def collect_run(run: Run) -> dict[str, object]:
    state, reason = queue_state(run)
    orchestrator_log = read_log(run, "orchestrator.log", r"Finished group \| env=genagent-train|SUCCESS.*Step")
    eval_log = read_log(
        run,
        "orchestrator.log",
        r"Evaluated genagent-heldout \(Step [0-9]+\) \| Policy v[0-9]+|"
        r"Eval genagent-heldout step [0-9]+ had mixed policy versions:",
    )
    return {
        "state": state,
        "reason": reason,
        "groups": parse_groups(orchestrator_log),
        "rollout_steps": parse_rollout_steps(orchestrator_log),
        "steps": parse_steps(read_log(run, "trainer.log", r"SUCCESS.*Step")),
        "evals": read_eval_summaries(run, eval_log),
        "token_export": stable_max(run, "token_exports"),
        "broadcast": stable_max(run, "broadcasts"),
    }


def resource_cost(run: Run, now: datetime) -> dict[str, object]:
    if run.hourly_cost_usd is None or run.billing_started_at_utc is None:
        return {
            "hourly_cost_usd": None,
            "billing_started_at_utc": None,
            "estimated_accrued_cost_usd": None,
        }
    started = datetime.fromisoformat(run.billing_started_at_utc)
    elapsed_hours = max(0.0, (now - started).total_seconds() / 3600)
    return {
        "hourly_cost_usd": run.hourly_cost_usd,
        "billing_started_at_utc": run.billing_started_at_utc,
        "estimated_accrued_cost_usd": round(elapsed_hours * run.hourly_cost_usd, 2),
    }


def auxiliary_resources(now: datetime) -> list[dict[str, object]]:
    resources: list[dict[str, object]] = []
    for path in AUXILIARY_STATE_FILES:
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        phase = str(state.get("phase") or "unknown")
        started_at = state.get("created_at_utc")
        hourly_cost = state.get("hourly_cost_usd")
        ended_at = state.get("updated_at_utc") if phase.endswith("terminated") else None
        accrued_cost = None
        if isinstance(started_at, str) and isinstance(hourly_cost, (int, float)):
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end = (
                datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
                if ended_at
                else now
            )
            accrued_cost = round(max(0.0, (end - started).total_seconds() / 3600) * hourly_cost, 2)
        resources.append(
            {
                "id": state.get("pod_id"),
                "name": state.get("pod_name"),
                "purpose": "pinned heldout evaluation",
                "phase": phase,
                "detail": state.get("detail"),
                "state_file": str(path),
                "resource_cost": {
                    "hourly_cost_usd": hourly_cost,
                    "billing_started_at_utc": started_at,
                    "billing_ended_at_utc": ended_at,
                    "estimated_accrued_cost_usd": accrued_cost,
                },
            }
        )
    return resources


def publish() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    fleet: dict[str, object] = {
        "schema_version": 3,
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "comparison_label": "General Agent | matched objectives and hint styles | Qwen3.5-35B-A3B",
        "provenance_revision": "r37",
        "config_fingerprint": "7f45e472c5314f70dd7ed60231fa5262b60b45a450c3cd86f6f1cab43ee119df",
        "target_optimizer_steps": 100,
        "auxiliary_resources": auxiliary_resources(now),
        "runs": {},
    }
    with ThreadPoolExecutor(max_workers=len(RUNS)) as pool:
        snapshots = dict(zip(RUNS, pool.map(collect_run, RUNS), strict=True))
    for run in RUNS:
        snapshot = snapshots[run]
        state = str(snapshot["state"])
        reason = str(snapshot["reason"] or "")
        groups = snapshot["groups"]
        rollout_steps = snapshot["rollout_steps"]
        steps = snapshot["steps"]
        evals = snapshot["evals"]
        token_export = snapshot["token_export"]
        broadcast = snapshot["broadcast"]
        accepted = [row for row in groups if row["accepted"]]
        fully_filtered = [row for row in groups if not row["accepted"]]
        fleet["runs"][run.arm] = {  # type: ignore[index]
            **RUN_LABELS[run.arm],
            "environment": "general_agent_v1",
            "model": "Qwen/Qwen3.5-35B-A3B",
            "train_task_count": 8,
            "train_selection": "variance_qualified_low_truncation",
            "group_size": 8,
            "target_optimizer_steps": 100,
            "provenance_revision": "r37",
            "state": state,
            "reason": reason or None,
            "job": int(run.job) if run.job else None,
            "placement": run.placement,
            "resource_cost": resource_cost(run, now),
            "output_dir": run.output_dir,
            "wandb": run.wandb or None,
            "highest_stable_optimizer_step": token_export,
            "completed_optimizer_steps": token_export + 1 if token_export is not None else 0,
            "highest_stable_policy_broadcast": broadcast,
            "completed_groups": len(groups),
            "accepted_groups": len(accepted),
            "fully_filtered_groups": len(fully_filtered),
            "zero_advantage_filtered_rollouts": sum(
                int(row["filtered"]) for row in groups if "zero_advantage" in str(row["filters"])
            ),
            "last_group": groups[-1] if groups else None,
            "last_rollout_metrics": rollout_steps[-1] if rollout_steps else None,
            "last_optimizer_metrics": steps[-1] if steps else None,
            "heldout_evals": evals,
        }
        write_csv(
            DATA_DIR / f"r37-{run.arm}.groups.csv",
            groups,
            ["group_index", "task_idx", "rollouts", "errored", "filtered", "reward", "filters", "accepted", "optimizer_step"],
        )
        write_csv(
            DATA_DIR / f"r37-{run.arm}.steps.csv",
            steps,
            ["step", "duration_s", "loss", "entropy", "mismatch_kl", "grad_norm", "lr", "throughput_tokens_s", "mfu_pct", "peak_mem_gib"],
        )
        write_csv(
            DATA_DIR / f"r37-{run.arm}.rollouts.csv",
            rollout_steps,
            [
                "step",
                "duration_s",
                "reward",
                "trainable_rollouts",
                "rollouts",
                "trainable_pct",
                "turns",
                "branches",
                "max_off_policy",
                "error_pct",
                "truncation_pct",
            ],
        )
    atomic_write(STATUS_DIR / "r37-live-status.json", json.dumps(fleet, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    while True:
        publish()
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
