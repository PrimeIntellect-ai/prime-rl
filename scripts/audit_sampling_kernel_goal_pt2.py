#!/usr/bin/env python3
"""Audit the local evidence for docs/plans/sampling-kernel/goal-pt2.md."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

THROUGHPUT_RE = re.compile(
    r"INFO (?P<stamp>\d\d-\d\d \d\d:\d\d:\d\d) .*"
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%, "
    r"Prefix cache hit rate: (?P<prefix>[0-9.]+)%"
)

DEFAULT_NATIVE_LOG = Path(
    "outputs/sampling-kernel/r69-native-repeat-n175n195/r69-native-repeat-n175n195/logs/inference/node_0.log"
)
DEFAULT_PATCHED_LOG = Path(
    "outputs/sampling-kernel/r68-patched-repeat-n175n195/r68-patched-repeat-n175n195/logs/inference/node_0.log"
)
DEFAULT_R77_INFERENCE_DIR = Path(
    "outputs/sampling-kernel/r77-fixed-top-p-precompile-n175n195/r77-fixed-top-p-precompile-n175n195/logs/inference"
)
DEFAULT_R56_DIR = Path(
    "outputs/sampling-kernel/r56-patched-b16-decodecg-tokenexport-finalguard/"
    "sampling-r56-patched-b16-decodecg-tokenexport-finalguard"
)
DEFAULT_R57_DIR = Path(
    "outputs/sampling-kernel/r57-flashinfer-b16-decodecg-tokenexport/sampling-r57-flashinfer-b16-decodecg-tokenexport"
)
DEFAULT_ALLOWED_HOSTS = "nid011175,nid011195"
DEFAULT_REQUIRED_TRAIN_NODES = 4
DEFAULT_REQUIRED_INFER_REPLICAS = 12
DEFAULT_PRODUCTION_CONFIG = Path("configs/calibration/gpqa_openended_debate_50step_bs512_g16_4t12i_simul_r64.toml")
DEFAULT_TRAINER_TEMPLATE = Path("src/prime_rl/templates/multi_node_rl.sbatch.j2")
DEFAULT_PYPROJECT = Path("pyproject.toml")
DEFAULT_PATCHES_PATH = Path("src/prime_rl/inference/patches.py")
DEFAULT_INFERENCE_TEMPLATES = (
    Path("src/prime_rl/templates/inference.sbatch.j2"),
    Path("src/prime_rl/templates/gpu_layout_rl.sh.j2"),
    Path("src/prime_rl/templates/_launch_rank.sh.j2"),
)
DEFAULT_SAMPLER_SOURCE = Path("src/prime_rl/inference/vllm/flashinfer_sampler.py")
DEFAULT_SAMPLER_CONTRACT_SWEEP = Path("scripts/probe_prime_flashinfer_sampler_contract_sweep.py")
DEFAULT_SAMPLER_CONTRACT_SWEEP_RESULT = Path("docs/plans/sampling-kernel/sampler_contract_sweep_latest.json")
DEFAULT_RENDERER_CLIENT = Path("deps/renderers/renderers/client.py")
DEFAULT_SERVING_TOKENS_TEST = Path("tests/unit/inference/test_serving_tokens.py")

EXPECTED_PRODUCTION_FIELDS: tuple[tuple[tuple[str, ...], Any], ...] = (
    (("model", "name"), "Qwen/Qwen3.5-35B-A3B"),
    (("seq_len",), 32768),
    (("deployment", "type"), "multi_node"),
    (("deployment", "gpus_per_node"), 4),
    (("deployment", "num_train_nodes"), 4),
    (("deployment", "num_infer_replicas"), 12),
    (("trainer", "model", "lora", "rank"), 64),
    (("orchestrator", "batch_size"), 512),
    (("orchestrator", "group_size"), 16),
    (("orchestrator", "target_lag"), 3),
    (("orchestrator", "max_inflight_rollouts"), 2300),
    (("orchestrator", "train", "sampling", "temperature"), 1.0),
    (("orchestrator", "train", "sampling", "top_p"), 0.95),
    (("orchestrator", "train", "sampling", "max_completion_tokens"), 16384),
    (("orchestrator", "train", "sampling", "thinking_token_budget"), 8192),
    (("orchestrator", "train", "sampling", "extra_body", "top_k"), 20),
    (("orchestrator", "train", "sampling", "extra_body", "min_p"), 0.0),
    (("orchestrator", "train", "sampling", "extra_body", "presence_penalty"), 1.5),
    (("inference", "enable_prefix_caching"), True),
    (("inference", "model", "max_model_len"), 32768),
    (("inference", "parallel", "tp"), 4),
    (("inference", "parallel", "dp"), 1),
    (("inference", "vllm_extra", "max_num_seqs"), 256),
    (("inference", "vllm_extra", "max_num_batched_tokens"), 131072),
    (("inference", "finite_topk_sampled_logprob", "enabled"), True),
    (("inference", "finite_topk_sampled_logprob", "tail"), "triton"),
    (("inference", "finite_topk_sampled_logprob", "dense_presence"), True),
    (("inference", "finite_topk_sampled_logprob", "stats_interval"), 1000),
    (("inference", "finite_topk_sampled_logprob", "hit_log_limit"), 6),
    (("inference", "finite_topk_sampled_logprob", "log_fallback"), True),
    (("inference", "finite_topk_sampled_logprob", "precompile_tail"), True),
    (("inference", "finite_topk_sampled_logprob", "precompile_top_k"), 20),
    (("inference", "finite_topk_sampled_logprob", "precompile_top_p"), 0.95),
    (("inference", "finite_topk_sampled_logprob", "precompile_vocab"), 248320),
    (("inference", "finite_topk_sampled_logprob", "precompile_batches"), [1, 128, 256]),
    (("inference", "finite_topk_sampled_logprob", "boundary_tie_guard"), True),
)
EXPECTED_TRAINER_NSYS_SNIPPETS = (
    "PRIME_RL_NSYS_TRAINER",
    'TRAIN_NODE_RANK" -eq 0',
    "$OUTPUT_DIR/nsys/trainer_node${TRAIN_NODE_RANK}",
    "--trace=cuda,nvtx,osrt,cublas,cudnn",
    "--gpu-metrics-set=gh100",
    "--kill=none",
    "PRIME_RL_NSYS_DURATION:+--stop-on-exit=false",
    "uv run --no-sync torchrun",
)
EXPECTED_PLUGIN_SNIPPETS = (
    (DEFAULT_PYPROJECT.name, '[project.entry-points."vllm.general_plugins"]'),
    (DEFAULT_PYPROJECT.name, 'prime_rl = "prime_rl.inference.patches:transformers_v5_compat"'),
    (DEFAULT_PATCHES_PATH.name, "def transformers_v5_compat"),
    (
        DEFAULT_PATCHES_PATH.name,
        "from prime_rl.inference.vllm.flashinfer_sampler import apply_flashinfer_sampled_logprob_patch",
    ),
    (DEFAULT_PATCHES_PATH.name, "apply_flashinfer_sampled_logprob_patch()"),
)
EXPECTED_INFERENCE_TEMPLATE_SNIPPET = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_LOG_DIR"


@dataclass(frozen=True)
class ThroughputPoint:
    generation_tokens_s: float
    prompt_tokens_s: float
    running: int
    waiting: int
    kv_cache_pct: float
    prefix_cache_hit_pct: float


@dataclass(frozen=True)
class FilteredThroughput:
    label: str
    path: str
    total_points: int
    matching_points: int
    first_n: int
    first_n_mean_generation_tokens_s: float | None
    all_matching_mean_generation_tokens_s: float | None


@dataclass(frozen=True)
class ServingGate:
    native: FilteredThroughput
    patched: FilteredThroughput
    first_n_ratio: float | None
    all_matching_ratio: float | None
    pass_gate: bool


@dataclass(frozen=True)
class JitGate:
    directory: str
    files: int
    events: int
    kernel_hist: dict[str, int]
    sampler_tail_events: int
    pass_gate: bool


@dataclass(frozen=True)
class SamplerStatsGate:
    directory: str
    files: int
    min_learner_row_hit_rate: float | None
    learner_fallback_rows: int
    learner_fallback_calls: int
    fallback_reason_by_traffic: dict[str, int]
    pass_gate: bool


@dataclass(frozen=True)
class TokenExportRun:
    label: str
    directory: str
    trainer_finished: bool
    jsonl_files: int
    stable_files: int
    rows: int
    tokens: int
    loss_tokens: int
    shape_mismatches: int
    bad_numeric_values: int
    pass_gate: bool


@dataclass(frozen=True)
class TrainingCanaryGate:
    native: TokenExportRun
    patched: TokenExportRun
    pass_gate: bool


@dataclass(frozen=True)
class ProductionReadinessGate:
    required_train_nodes: int
    required_inference_replicas: int
    required_total_nodes: int
    allocation_hosts: list[str]
    allowed_hosts: list[str]
    allocation_has_required_nodes: bool
    allowed_lane_has_required_nodes: bool
    pass_gate: bool


@dataclass(frozen=True)
class ProductionConfigGate:
    path: str
    checked_fields: int
    mismatches: list[str]
    pass_gate: bool


@dataclass(frozen=True)
class TrainerNsysHookGate:
    path: str
    checked_snippets: int
    missing_snippets: list[str]
    pass_gate: bool


@dataclass(frozen=True)
class InferenceSamplerRuntimeHookGate:
    paths: list[str]
    checked_snippets: int
    missing_snippets: list[str]
    pass_gate: bool


@dataclass(frozen=True)
class SamplerContractSweepResultGate:
    path: str
    rows: int
    failures: int
    max_expected_patched_logprob_diff: float | None
    max_forced_logprob_diff: float | None
    native_cols: list[int]
    patched_cols: list[int]
    mismatches: list[str]
    pass_gate: bool


@dataclass(frozen=True)
class TailSpecializationGate:
    kernel_parameters: list[str]
    kernel_constexprs: list[int]
    precompile_top_p_values: list[float]
    error: str | None
    pass_gate: bool


@dataclass(frozen=True)
class ProductionPreflight:
    production_config: ProductionConfigGate
    trainer_nsys_hook: TrainerNsysHookGate
    inference_sampler_runtime_hooks: InferenceSamplerRuntimeHookGate
    tail_specialization: TailSpecializationGate
    production_readiness: ProductionReadinessGate
    full_production_ready: bool


@dataclass(frozen=True)
class GoalAudit:
    broad: ServingGate
    strict_decode: ServingGate
    training_canary: TrainingCanaryGate
    jit: JitGate
    sampler_stats: SamplerStatsGate
    sampler_contract_sweep: SamplerContractSweepResultGate
    production_config: ProductionConfigGate
    trainer_nsys_hook: TrainerNsysHookGate
    inference_sampler_runtime_hooks: InferenceSamplerRuntimeHookGate
    tail_specialization: TailSpecializationGate
    production_readiness: ProductionReadinessGate
    full_production_e2e_gate: str
    local_two_node_gates_pass: bool
    goal_complete: bool


def parse_points(path: Path) -> list[ThroughputPoint]:
    points = []
    for line in path.read_text().splitlines():
        match = THROUGHPUT_RE.search(line)
        if match is None:
            continue
        points.append(
            ThroughputPoint(
                generation_tokens_s=float(match.group("generation")),
                prompt_tokens_s=float(match.group("prompt")),
                running=int(match.group("running")),
                waiting=int(match.group("waiting")),
                kv_cache_pct=float(match.group("kv")),
                prefix_cache_hit_pct=float(match.group("prefix")),
            )
        )
    return points


def prompt_matches(
    point: ThroughputPoint,
    exact_prompt_tokens_s: float | None,
    min_prompt_tokens_s: float | None,
    max_prompt_tokens_s: float | None,
) -> bool:
    if exact_prompt_tokens_s is not None:
        return point.prompt_tokens_s == exact_prompt_tokens_s
    if min_prompt_tokens_s is not None and point.prompt_tokens_s < min_prompt_tokens_s:
        return False
    return max_prompt_tokens_s is None or point.prompt_tokens_s <= max_prompt_tokens_s


def summarize_throughput(
    *,
    label: str,
    path: Path,
    first_n: int,
    running: int,
    min_waiting: int,
    min_kv_cache_pct: float,
    max_kv_cache_pct: float,
    exact_prompt_tokens_s: float | None,
    min_prompt_tokens_s: float | None,
    max_prompt_tokens_s: float | None,
) -> FilteredThroughput:
    points = parse_points(path)
    matching = [
        point
        for point in points
        if point.running == running
        and point.waiting >= min_waiting
        and min_kv_cache_pct <= point.kv_cache_pct <= max_kv_cache_pct
        and prompt_matches(
            point,
            exact_prompt_tokens_s=exact_prompt_tokens_s,
            min_prompt_tokens_s=min_prompt_tokens_s,
            max_prompt_tokens_s=max_prompt_tokens_s,
        )
    ]
    first = matching[:first_n]
    return FilteredThroughput(
        label=label,
        path=str(path),
        total_points=len(points),
        matching_points=len(matching),
        first_n=first_n,
        first_n_mean_generation_tokens_s=(mean(point.generation_tokens_s for point in first) if first else None),
        all_matching_mean_generation_tokens_s=(
            mean(point.generation_tokens_s for point in matching) if matching else None
        ),
    )


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def serving_gate(
    *,
    native_log: Path,
    patched_log: Path,
    first_n: int,
    min_matching_points: int,
    min_ratio: float,
    exact_prompt_tokens_s: float | None,
    min_prompt_tokens_s: float | None,
    max_prompt_tokens_s: float | None,
) -> ServingGate:
    native = summarize_throughput(
        label="native",
        path=native_log,
        first_n=first_n,
        running=256,
        min_waiting=1,
        min_kv_cache_pct=15.0,
        max_kv_cache_pct=45.0,
        exact_prompt_tokens_s=exact_prompt_tokens_s,
        min_prompt_tokens_s=min_prompt_tokens_s,
        max_prompt_tokens_s=max_prompt_tokens_s,
    )
    patched = summarize_throughput(
        label="patched",
        path=patched_log,
        first_n=first_n,
        running=256,
        min_waiting=1,
        min_kv_cache_pct=15.0,
        max_kv_cache_pct=45.0,
        exact_prompt_tokens_s=exact_prompt_tokens_s,
        min_prompt_tokens_s=min_prompt_tokens_s,
        max_prompt_tokens_s=max_prompt_tokens_s,
    )
    first_n_ratio = ratio(
        patched.first_n_mean_generation_tokens_s,
        native.first_n_mean_generation_tokens_s,
    )
    all_matching_ratio = ratio(
        patched.all_matching_mean_generation_tokens_s,
        native.all_matching_mean_generation_tokens_s,
    )
    return ServingGate(
        native=native,
        patched=patched,
        first_n_ratio=first_n_ratio,
        all_matching_ratio=all_matching_ratio,
        pass_gate=(
            native.matching_points >= min_matching_points
            and patched.matching_points >= min_matching_points
            and first_n_ratio is not None
            and first_n_ratio >= min_ratio
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_jit(inference_dir: Path) -> JitGate:
    files = sorted(inference_dir.glob("jit_monitor_*.jsonl"))
    kernel_hist: Counter[str] = Counter()
    events = 0
    for path in files:
        for record in read_jsonl(path):
            events += 1
            kernel_hist[str(record["kernel"])] += 1
    sampler_tail_events = kernel_hist.get("_k_tail_uniform_kernel", 0)
    return JitGate(
        directory=str(inference_dir),
        files=len(files),
        events=events,
        kernel_hist=dict(sorted(kernel_hist.items())),
        sampler_tail_events=sampler_tail_events,
        pass_gate=len(files) == 4 and events > 0 and sampler_tail_events == 0,
    )


def audit_sampler_stats(inference_dir: Path) -> SamplerStatsGate:
    files = sorted(inference_dir.glob("finite_topk_sampler_stats_*.jsonl"))
    latest_records = [read_jsonl(path)[-1] for path in files]
    hit_rates = [float(record["learner_row_hit_rate"]) for record in latest_records]
    fallback_reasons: Counter[str] = Counter()
    for record in latest_records:
        fallback_reasons.update(record.get("fallback_reason_by_traffic", {}))
    learner_fallback_rows = sum(int(record.get("learner_fallback_rows", 0)) for record in latest_records)
    learner_fallback_calls = sum(int(record.get("learner_fallback_calls", 0)) for record in latest_records)
    min_hit_rate = min(hit_rates) if hit_rates else None
    return SamplerStatsGate(
        directory=str(inference_dir),
        files=len(files),
        min_learner_row_hit_rate=min_hit_rate,
        learner_fallback_rows=learner_fallback_rows,
        learner_fallback_calls=learner_fallback_calls,
        fallback_reason_by_traffic=dict(sorted(fallback_reasons.items())),
        pass_gate=(
            len(files) == 4
            and min_hit_rate is not None
            and min_hit_rate >= 0.999
            and learner_fallback_rows == 0
            and learner_fallback_calls == 0
        ),
    )


def list_length(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    return len(value) if isinstance(value, list) else None


def count_bad_numeric(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    return sum(1 for value in values if isinstance(value, int | float) and not math.isfinite(float(value)))


def audit_token_export_run(label: str, directory: Path) -> TokenExportRun:
    token_export_dir = directory / "run_default" / "token_exports"
    jsonl_files = sorted(token_export_dir.glob("step_*/rank_*.jsonl"))
    stable_files = sorted(token_export_dir.glob("step_*/STABLE"))
    trainer_log = directory / "logs" / "trainer" / "node_0.log"
    trainer_finished = trainer_log.exists() and "RL trainer finished" in trainer_log.read_text()

    rows = 0
    tokens = 0
    loss_tokens = 0
    shape_mismatches = 0
    bad_numeric_values = 0
    same_length_fields = (
        "loss_mask",
        "trainer_logprobs",
        "inference_logprobs",
        "log_importance_ratio",
        "importance_ratio",
        "mismatch_kl",
        "entropy",
    )
    numeric_fields = (
        "trainer_logprobs",
        "inference_logprobs",
        "log_importance_ratio",
        "importance_ratio",
        "mismatch_kl",
        "entropy",
    )
    for path in jsonl_files:
        for record in read_jsonl(path):
            rows += 1
            token_count = list_length(record, "token_ids")
            if token_count is None:
                shape_mismatches += 1
                continue
            tokens += token_count
            lengths = [list_length(record, field) for field in same_length_fields]
            shape_mismatches += sum(1 for length in lengths if length is not None and length != token_count)
            loss_mask = record.get("loss_mask")
            if isinstance(loss_mask, list):
                loss_tokens += sum(1 for value in loss_mask if value is True)
            for field in numeric_fields:
                bad_numeric_values += count_bad_numeric(record.get(field))

    pass_gate = (
        trainer_finished
        and len(jsonl_files) >= 4
        and len(stable_files) >= 2
        and rows > 0
        and tokens > 0
        and shape_mismatches == 0
        and bad_numeric_values == 0
    )
    return TokenExportRun(
        label=label,
        directory=str(directory),
        trainer_finished=trainer_finished,
        jsonl_files=len(jsonl_files),
        stable_files=len(stable_files),
        rows=rows,
        tokens=tokens,
        loss_tokens=loss_tokens,
        shape_mismatches=shape_mismatches,
        bad_numeric_values=bad_numeric_values,
        pass_gate=pass_gate,
    )


def audit_training_canary(native_dir: Path, patched_dir: Path) -> TrainingCanaryGate:
    native = audit_token_export_run("native", native_dir)
    patched = audit_token_export_run("patched", patched_dir)
    return TrainingCanaryGate(
        native=native,
        patched=patched,
        pass_gate=native.pass_gate and patched.pass_gate,
    )


def split_top_level_commas(raw: str) -> list[str]:
    parts = []
    start = 0
    bracket_depth = 0
    for index, char in enumerate(raw):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and bracket_depth == 0:
            parts.append(raw[start:index])
            start = index + 1
    parts.append(raw[start:])
    return [part.strip() for part in parts if part.strip()]


def expand_host_token(token: str) -> list[str]:
    if "[" not in token:
        return [token]
    prefix, rest = token.split("[", 1)
    inner, suffix = rest.split("]", 1)
    hosts = []
    for part in inner.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            width = len(start)
            hosts.extend(f"{prefix}{value:0{width}d}{suffix}" for value in range(int(start), int(end) + 1))
        else:
            hosts.append(f"{prefix}{part}{suffix}")
    return hosts


def expand_host_list(raw: str) -> list[str]:
    hosts = []
    for token in split_top_level_commas(raw):
        hosts.extend(expand_host_token(token))
    return hosts


def audit_production_readiness(args: argparse.Namespace) -> ProductionReadinessGate:
    allocation_hosts = expand_host_list(args.allocation_hosts) if args.allocation_hosts else []
    allowed_hosts = expand_host_list(args.allowed_hosts) if args.allowed_hosts else []
    required_total_nodes = args.required_train_nodes + args.required_inference_replicas
    allocation_has_required_nodes = len(allocation_hosts) >= required_total_nodes
    allowed_lane_has_required_nodes = len(allowed_hosts) >= required_total_nodes
    return ProductionReadinessGate(
        required_train_nodes=args.required_train_nodes,
        required_inference_replicas=args.required_inference_replicas,
        required_total_nodes=required_total_nodes,
        allocation_hosts=allocation_hosts,
        allowed_hosts=allowed_hosts,
        allocation_has_required_nodes=allocation_has_required_nodes,
        allowed_lane_has_required_nodes=allowed_lane_has_required_nodes,
        pass_gate=allocation_has_required_nodes and allowed_lane_has_required_nodes,
    )


_MISSING = object()


def get_nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def audit_production_config(path: Path) -> ProductionConfigGate:
    if not path.exists():
        return ProductionConfigGate(
            path=str(path),
            checked_fields=len(EXPECTED_PRODUCTION_FIELDS),
            mismatches=["file missing"],
            pass_gate=False,
        )
    config = tomllib.loads(path.read_text())
    mismatches = []
    for field_path, expected in EXPECTED_PRODUCTION_FIELDS:
        actual = get_nested(config, field_path)
        if actual != expected:
            label = ".".join(field_path)
            actual_repr = "<missing>" if actual is _MISSING else repr(actual)
            mismatches.append(f"{label}: expected {expected!r}, got {actual_repr}")
    return ProductionConfigGate(
        path=str(path),
        checked_fields=len(EXPECTED_PRODUCTION_FIELDS),
        mismatches=mismatches,
        pass_gate=not mismatches,
    )


def audit_trainer_nsys_hook(path: Path) -> TrainerNsysHookGate:
    if not path.exists():
        return TrainerNsysHookGate(
            path=str(path),
            checked_snippets=len(EXPECTED_TRAINER_NSYS_SNIPPETS),
            missing_snippets=["file missing"],
            pass_gate=False,
        )
    text = path.read_text()
    missing = [snippet for snippet in EXPECTED_TRAINER_NSYS_SNIPPETS if snippet not in text]
    return TrainerNsysHookGate(
        path=str(path),
        checked_snippets=len(EXPECTED_TRAINER_NSYS_SNIPPETS),
        missing_snippets=missing,
        pass_gate=not missing,
    )


def _read_text_or_missing(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text()


def audit_inference_sampler_runtime_hooks(
    *,
    pyproject_path: Path = DEFAULT_PYPROJECT,
    patches_path: Path = DEFAULT_PATCHES_PATH,
    template_paths: tuple[Path, ...] = DEFAULT_INFERENCE_TEMPLATES,
    sampler_source_path: Path = DEFAULT_SAMPLER_SOURCE,
    contract_sweep_path: Path = DEFAULT_SAMPLER_CONTRACT_SWEEP,
    renderer_client_path: Path = DEFAULT_RENDERER_CLIENT,
    serving_tokens_test_path: Path = DEFAULT_SERVING_TOKENS_TEST,
) -> InferenceSamplerRuntimeHookGate:
    missing = []
    checked = 0

    text_by_label = {
        DEFAULT_PYPROJECT.name: _read_text_or_missing(pyproject_path),
        DEFAULT_PATCHES_PATH.name: _read_text_or_missing(patches_path),
    }
    for label, snippet in EXPECTED_PLUGIN_SNIPPETS:
        checked += 1
        text = text_by_label[label]
        if text is None:
            missing.append(f"{label}: file missing")
        elif snippet not in text:
            missing.append(f"{label}: {snippet}")

    for template_path in template_paths:
        checked += 1
        text = _read_text_or_missing(template_path)
        label = template_path.name
        if text is None:
            missing.append(f"{label}: file missing")
        elif EXPECTED_INFERENCE_TEMPLATE_SNIPPET not in text:
            missing.append(f"{label}: {EXPECTED_INFERENCE_TEMPLATE_SNIPPET}")

    runtime_contract_snippets = (
        (sampler_source_path, "_has_top_p_boundary_tie"),
        (sampler_source_path, '"top_p_boundary_tie"'),
        (sampler_source_path, "_TOP_P_BOUNDARY_TIE_ATOL = 5e-7"),
        (sampler_source_path, "torch.isclose("),
        (sampler_source_path, "atol=_TOP_P_BOUNDARY_TIE_ATOL"),
        (contract_sweep_path, "production_vocab_specs"),
        (contract_sweep_path, "248320"),
        (contract_sweep_path, "--include-width1"),
        (contract_sweep_path, "probe_failed"),
        (contract_sweep_path, "_BOUNDARY_TIE_GUARD_ENV"),
        (renderer_client_path, 'sp["logprobs"] = 0'),
        (renderer_client_path, "isinstance(raw_completion_ids, list)"),
        (renderer_client_path, "token_id < 0"),
        (renderer_client_path, "len(completion_logprobs) != len(completion_ids)"),
        (renderer_client_path, "math.isfinite(logprob)"),
        (serving_tokens_test_path, "test_generate_logprobs_zero_serializes_sampled_completion_logprob"),
    )
    for path, snippet in runtime_contract_snippets:
        checked += 1
        text = _read_text_or_missing(path)
        label = path.name
        if text is None:
            missing.append(f"{label}: file missing")
        elif snippet not in text:
            missing.append(f"{label}: {snippet}")

    return InferenceSamplerRuntimeHookGate(
        paths=[
            str(pyproject_path),
            str(patches_path),
            *(str(path) for path in template_paths),
            str(sampler_source_path),
            str(contract_sweep_path),
            str(renderer_client_path),
            str(serving_tokens_test_path),
        ],
        checked_snippets=checked,
        missing_snippets=missing,
        pass_gate=not missing,
    )


def _summary_number(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _row_bool_mismatch(row: dict[str, Any], index: int, key: str, expected: bool) -> str | None:
    actual = row.get(key)
    if actual == expected:
        return None
    return f"results[{index}].{key}: expected {expected}, got {actual}"


def _row_int_mismatch(row: dict[str, Any], index: int, key: str, expected: int) -> str | None:
    actual = row.get(key)
    if actual == expected:
        return None
    return f"results[{index}].{key}: expected {expected}, got {actual}"


def _validate_sampler_contract_rows(
    results: list[dict[str, Any]],
    *,
    min_rows: int,
    max_diff: float,
) -> list[str]:
    if len(results) < min_rows:
        return [f"results: expected at least {min_rows} rows, got {len(results)}"]

    mismatches = []
    top_ks = set()
    max_num_logprobs_values = set()
    dense_presence_values = set()
    for index, row in enumerate(results):
        expected_cols = int(row.get("max_num_logprobs", -1)) + 1
        top_ks.add(row.get("top_k"))
        max_num_logprobs_values.add(row.get("max_num_logprobs"))
        dense_presence_values.add(row.get("dense_presence_enabled"))

        for key in (
            "env_enabled",
            "patch_marker",
            "fastpath_allowed",
        ):
            mismatch = _row_bool_mismatch(row, index, key, True)
            if mismatch is not None:
                mismatches.append(mismatch)
        for key in (
            "expected_patched_sampled_token_mismatches",
            "expected_patched_logprob_token_id_mismatches",
            "expected_patched_selected_rank_mismatches",
        ):
            mismatch = _row_int_mismatch(row, index, key, 0)
            if mismatch is not None:
                mismatches.append(mismatch)

        if row.get("device") != "cuda":
            mismatches.append(f"results[{index}].device: expected cuda, got {row.get('device')}")
        if row.get("vocab_size") != 248320:
            mismatches.append(f"results[{index}].vocab_size: expected 248320, got {row.get('vocab_size')}")
        if row.get("top_p") != 0.95:
            mismatches.append(f"results[{index}].top_p: expected 0.95, got {row.get('top_p')}")
        if row.get("top_k") not in {20, 64}:
            mismatches.append(f"results[{index}].top_k: expected one of [20, 64], got {row.get('top_k')}")
        if row.get("max_num_logprobs") not in {0, 1}:
            mismatches.append(
                f"results[{index}].max_num_logprobs: expected one of [0, 1], got {row.get('max_num_logprobs')}"
            )
        if row.get("native_cols") != expected_cols:
            mismatches.append(f"results[{index}].native_cols: expected {expected_cols}, got {row.get('native_cols')}")
        if row.get("patched_cols") != expected_cols:
            mismatches.append(f"results[{index}].patched_cols: expected {expected_cols}, got {row.get('patched_cols')}")

        max_expected = row.get("max_expected_patched_logprob_diff")
        if max_expected is None or float(max_expected) > max_diff:
            mismatches.append(
                f"results[{index}].max_expected_patched_logprob_diff: expected <= {max_diff}, got {max_expected}"
            )
        max_forced = row.get("max_forced_logprob_diff")
        if max_forced is None or float(max_forced) > max_diff:
            mismatches.append(f"results[{index}].max_forced_logprob_diff: expected <= {max_diff}, got {max_forced}")

    if top_ks != {20, 64}:
        mismatches.append(f"results.top_k coverage: expected [20, 64], got {sorted(top_ks)}")
    if max_num_logprobs_values != {0, 1}:
        mismatches.append(f"results.max_num_logprobs coverage: expected [0, 1], got {sorted(max_num_logprobs_values)}")
    if dense_presence_values != {False, True}:
        mismatches.append(
            f"results.dense_presence_enabled coverage: expected [False, True], got {sorted(dense_presence_values)}"
        )
    return mismatches


def audit_sampler_contract_sweep_result(
    path: Path = DEFAULT_SAMPLER_CONTRACT_SWEEP_RESULT,
    *,
    min_rows: int = 12,
    max_diff: float = 1e-6,
) -> SamplerContractSweepResultGate:
    if not path.exists():
        return SamplerContractSweepResultGate(
            path=str(path),
            rows=0,
            failures=-1,
            max_expected_patched_logprob_diff=None,
            max_forced_logprob_diff=None,
            native_cols=[],
            patched_cols=[],
            mismatches=["file missing"],
            pass_gate=False,
        )

    try:
        data = json.loads(path.read_text())
        summary = data["summary"]
        results = list(data.get("results", []))
        rows = int(summary.get("rows", 0))
        failures = int(summary.get("failures", -1))
        max_expected = _summary_number(summary, "max_expected_patched_logprob_diff")
        max_forced = _summary_number(summary, "max_forced_logprob_diff")
        native_cols = list(summary.get("native_cols", []))
        patched_cols = list(summary.get("patched_cols", []))
    except Exception as exc:
        return SamplerContractSweepResultGate(
            path=str(path),
            rows=0,
            failures=-1,
            max_expected_patched_logprob_diff=None,
            max_forced_logprob_diff=None,
            native_cols=[],
            patched_cols=[],
            mismatches=[f"{type(exc).__name__}: {exc}"],
            pass_gate=False,
        )

    mismatches = []
    if rows < min_rows:
        mismatches.append(f"rows: expected at least {min_rows}, got {rows}")
    if failures != 0:
        mismatches.append(f"failures: expected 0, got {failures}")
    if max_expected is None or max_expected > max_diff:
        mismatches.append(f"max_expected_patched_logprob_diff: expected <= {max_diff}, got {max_expected}")
    if max_forced is None or max_forced > max_diff:
        mismatches.append(f"max_forced_logprob_diff: expected <= {max_diff}, got {max_forced}")
    if native_cols != [1, 2]:
        mismatches.append(f"native_cols: expected [1, 2], got {native_cols}")
    if patched_cols != [1, 2]:
        mismatches.append(f"patched_cols: expected [1, 2], got {patched_cols}")
    if rows != len(results):
        mismatches.append(f"summary rows: expected {len(results)}, got {rows}")
    mismatches.extend(_validate_sampler_contract_rows(results, min_rows=min_rows, max_diff=max_diff))

    return SamplerContractSweepResultGate(
        path=str(path),
        rows=rows,
        failures=failures,
        max_expected_patched_logprob_diff=max_expected,
        max_forced_logprob_diff=max_forced,
        native_cols=native_cols,
        patched_cols=patched_cols,
        mismatches=mismatches,
        pass_gate=not mismatches,
    )


def audit_tail_specialization() -> TailSpecializationGate:
    try:
        from prime_rl.inference.vllm import flashinfer_sampler

        signature = inspect.signature(flashinfer_sampler._k_tail_uniform_kernel.fn)
        parameters = list(signature.parameters)
        constexprs = list(flashinfer_sampler._k_tail_uniform_kernel.constexprs)
        k_block_index = parameters.index("K_BLOCK") if "K_BLOCK" in parameters else None
        top_p_values = flashinfer_sampler._precompile_tail_top_p_values()
        pass_gate = (
            "TOP_P" not in signature.parameters
            and "top_p" in signature.parameters
            and signature.parameters["K"].annotation is inspect.Signature.empty
            and signature.parameters["top_p"].annotation is inspect.Signature.empty
            and signature.parameters["K_BLOCK"].annotation == "tl.constexpr"
            and constexprs == [k_block_index]
            and top_p_values == [0.95]
        )
        return TailSpecializationGate(
            kernel_parameters=parameters,
            kernel_constexprs=constexprs,
            precompile_top_p_values=top_p_values,
            error=None,
            pass_gate=pass_gate,
        )
    except Exception as exc:
        return TailSpecializationGate(
            kernel_parameters=[],
            kernel_constexprs=[],
            precompile_top_p_values=[],
            error=f"{type(exc).__name__}: {exc}",
            pass_gate=False,
        )


def production_preflight(args: argparse.Namespace) -> ProductionPreflight:
    production_config = audit_production_config(args.production_config)
    trainer_nsys_hook = audit_trainer_nsys_hook(args.trainer_template)
    inference_sampler_runtime_hooks = audit_inference_sampler_runtime_hooks(
        pyproject_path=getattr(args, "pyproject", DEFAULT_PYPROJECT),
        patches_path=getattr(args, "patches_path", DEFAULT_PATCHES_PATH),
        template_paths=tuple(getattr(args, "inference_templates", DEFAULT_INFERENCE_TEMPLATES)),
    )
    tail_specialization = audit_tail_specialization()
    production_readiness = audit_production_readiness(args)
    return ProductionPreflight(
        production_config=production_config,
        trainer_nsys_hook=trainer_nsys_hook,
        inference_sampler_runtime_hooks=inference_sampler_runtime_hooks,
        tail_specialization=tail_specialization,
        production_readiness=production_readiness,
        full_production_ready=(
            production_config.pass_gate
            and trainer_nsys_hook.pass_gate
            and inference_sampler_runtime_hooks.pass_gate
            and tail_specialization.pass_gate
            and production_readiness.pass_gate
        ),
    )


def audit(args: argparse.Namespace) -> GoalAudit:
    broad = serving_gate(
        native_log=args.native_log,
        patched_log=args.patched_log,
        first_n=12,
        min_matching_points=10,
        min_ratio=1.10,
        exact_prompt_tokens_s=None,
        min_prompt_tokens_s=0.0,
        max_prompt_tokens_s=10000.0,
    )
    strict_decode = serving_gate(
        native_log=args.native_log,
        patched_log=args.patched_log,
        first_n=12,
        min_matching_points=1,
        min_ratio=1.0,
        exact_prompt_tokens_s=0.0,
        min_prompt_tokens_s=None,
        max_prompt_tokens_s=None,
    )
    training_canary = audit_training_canary(args.r56_dir, args.r57_dir)
    jit = audit_jit(args.r77_inference_dir)
    sampler_stats = audit_sampler_stats(args.r77_inference_dir)
    sampler_contract_sweep = audit_sampler_contract_sweep_result(args.sampler_contract_sweep_result)
    preflight = production_preflight(args)
    local_two_node_gates_pass = (
        broad.pass_gate
        and strict_decode.pass_gate
        and training_canary.pass_gate
        and jit.pass_gate
        and sampler_stats.pass_gate
        and sampler_contract_sweep.pass_gate
        and preflight.inference_sampler_runtime_hooks.pass_gate
        and preflight.tail_specialization.pass_gate
    )
    return GoalAudit(
        broad=broad,
        strict_decode=strict_decode,
        training_canary=training_canary,
        jit=jit,
        sampler_stats=sampler_stats,
        sampler_contract_sweep=sampler_contract_sweep,
        production_config=preflight.production_config,
        trainer_nsys_hook=preflight.trainer_nsys_hook,
        inference_sampler_runtime_hooks=preflight.inference_sampler_runtime_hooks,
        tail_specialization=preflight.tail_specialization,
        production_readiness=preflight.production_readiness,
        full_production_e2e_gate="missing",
        local_two_node_gates_pass=local_two_node_gates_pass,
        goal_complete=False,
    )


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def print_markdown(report: GoalAudit) -> None:
    print("| gate | status | key result |")
    print("|---|---|---|")
    print(
        "| broad two-node serving | "
        f"{'pass' if report.broad.pass_gate else 'fail'} | "
        f"{report.broad.native.matching_points} native rows, "
        f"{report.broad.patched.matching_points} patched rows, "
        f"ratio {fmt(report.broad.first_n_ratio)}x |"
    )
    print(
        "| strict decode two-node serving | "
        f"{'pass' if report.strict_decode.pass_gate else 'fail'} | "
        f"{report.strict_decode.native.matching_points} native rows, "
        f"{report.strict_decode.patched.matching_points} patched rows, "
        f"ratio {fmt(report.strict_decode.first_n_ratio)}x |"
    )
    print(
        "| sampler-tail JIT | "
        f"{'pass' if report.jit.pass_gate else 'fail'} | "
        f"{report.jit.files} files, {report.jit.events} events, "
        f"_k_tail_uniform_kernel={report.jit.sampler_tail_events} |"
    )
    print(
        "| sampler stats | "
        f"{'pass' if report.sampler_stats.pass_gate else 'fail'} | "
        f"{report.sampler_stats.files} files, "
        f"min learner row hit rate {fmt(report.sampler_stats.min_learner_row_hit_rate)}, "
        f"learner fallback rows {report.sampler_stats.learner_fallback_rows} |"
    )
    print(
        "| sampler contract sweep | "
        f"{'pass' if report.sampler_contract_sweep.pass_gate else 'fail'} | "
        f"{report.sampler_contract_sweep.path}, rows {report.sampler_contract_sweep.rows}, "
        f"failures {report.sampler_contract_sweep.failures}, "
        f"max expected diff {report.sampler_contract_sweep.max_expected_patched_logprob_diff}, "
        f"mismatches {len(report.sampler_contract_sweep.mismatches)} |"
    )
    print(
        "| training canary token export | "
        f"{'pass' if report.training_canary.pass_gate else 'fail'} | "
        f"native {report.training_canary.native.jsonl_files}/{report.training_canary.native.stable_files} "
        f"files/STABLE, rows {report.training_canary.native.rows}, "
        f"bad numeric {report.training_canary.native.bad_numeric_values}; "
        f"patched {report.training_canary.patched.jsonl_files}/{report.training_canary.patched.stable_files} "
        f"files/STABLE, rows {report.training_canary.patched.rows}, "
        f"bad numeric {report.training_canary.patched.bad_numeric_values} |"
    )
    print(
        "| production config shape | "
        f"{'pass' if report.production_config.pass_gate else 'fail'} | "
        f"{report.production_config.path}, checked {report.production_config.checked_fields} fields, "
        f"mismatches {len(report.production_config.mismatches)} |"
    )
    print(
        "| trainer Nsight hook | "
        f"{'pass' if report.trainer_nsys_hook.pass_gate else 'fail'} | "
        f"{report.trainer_nsys_hook.path}, checked {report.trainer_nsys_hook.checked_snippets} snippets, "
        f"missing {len(report.trainer_nsys_hook.missing_snippets)} |"
    )
    print(
        "| inference sampler runtime hooks | "
        f"{'pass' if report.inference_sampler_runtime_hooks.pass_gate else 'fail'} | "
        f"{len(report.inference_sampler_runtime_hooks.paths)} files, "
        f"checked {report.inference_sampler_runtime_hooks.checked_snippets} snippets, "
        f"missing {len(report.inference_sampler_runtime_hooks.missing_snippets)} |"
    )
    print(
        "| sampler-tail specialization | "
        f"{'pass' if report.tail_specialization.pass_gate else 'fail'} | "
        f"constexprs={report.tail_specialization.kernel_constexprs}, "
        f"top_p_values={report.tail_specialization.precompile_top_p_values}, "
        f"error={report.tail_specialization.error or 'none'} |"
    )
    print("| full production E2E | missing | requires real production topology; not provable from the two-node lane |")
    print(
        "| full production topology preflight | "
        f"{'pass' if report.production_readiness.pass_gate else 'missing'} | "
        f"requires {report.production_readiness.required_total_nodes} nodes "
        f"({report.production_readiness.required_train_nodes} train + "
        f"{report.production_readiness.required_inference_replicas} inference); "
        f"allocation has {len(report.production_readiness.allocation_hosts)}, "
        f"allowed lane has {len(report.production_readiness.allowed_hosts)} |"
    )
    print()
    print(f"local_two_node_gates_pass: {str(report.local_two_node_gates_pass).lower()}")
    full_production_ready = (
        report.production_config.pass_gate
        and report.trainer_nsys_hook.pass_gate
        and report.inference_sampler_runtime_hooks.pass_gate
        and report.tail_specialization.pass_gate
        and report.production_readiness.pass_gate
    )
    print(f"full_production_ready: {str(full_production_ready).lower()}")
    print("goal_complete: false")


def print_preflight_markdown(report: ProductionPreflight) -> None:
    print("| gate | status | key result |")
    print("|---|---|---|")
    print(
        "| production config shape | "
        f"{'pass' if report.production_config.pass_gate else 'fail'} | "
        f"{report.production_config.path}, checked {report.production_config.checked_fields} fields, "
        f"mismatches {len(report.production_config.mismatches)} |"
    )
    print(
        "| trainer Nsight hook | "
        f"{'pass' if report.trainer_nsys_hook.pass_gate else 'fail'} | "
        f"{report.trainer_nsys_hook.path}, checked {report.trainer_nsys_hook.checked_snippets} snippets, "
        f"missing {len(report.trainer_nsys_hook.missing_snippets)} |"
    )
    print(
        "| inference sampler runtime hooks | "
        f"{'pass' if report.inference_sampler_runtime_hooks.pass_gate else 'fail'} | "
        f"{len(report.inference_sampler_runtime_hooks.paths)} files, "
        f"checked {report.inference_sampler_runtime_hooks.checked_snippets} snippets, "
        f"missing {len(report.inference_sampler_runtime_hooks.missing_snippets)} |"
    )
    print(
        "| sampler-tail specialization | "
        f"{'pass' if report.tail_specialization.pass_gate else 'fail'} | "
        f"constexprs={report.tail_specialization.kernel_constexprs}, "
        f"top_p_values={report.tail_specialization.precompile_top_p_values}, "
        f"error={report.tail_specialization.error or 'none'} |"
    )
    print(
        "| full production topology preflight | "
        f"{'pass' if report.production_readiness.pass_gate else 'missing'} | "
        f"requires {report.production_readiness.required_total_nodes} nodes "
        f"({report.production_readiness.required_train_nodes} train + "
        f"{report.production_readiness.required_inference_replicas} inference); "
        f"allocation has {len(report.production_readiness.allocation_hosts)}, "
        f"allowed lane has {len(report.production_readiness.allowed_hosts)} |"
    )
    print()
    print(f"full_production_ready: {str(report.full_production_ready).lower()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-log", type=Path, default=DEFAULT_NATIVE_LOG)
    parser.add_argument("--patched-log", type=Path, default=DEFAULT_PATCHED_LOG)
    parser.add_argument("--r77-inference-dir", type=Path, default=DEFAULT_R77_INFERENCE_DIR)
    parser.add_argument("--r56-dir", type=Path, default=DEFAULT_R56_DIR)
    parser.add_argument("--r57-dir", type=Path, default=DEFAULT_R57_DIR)
    parser.add_argument(
        "--allocation-hosts",
        default=os.environ.get("SLURM_JOB_NODELIST", ""),
        help="Allocated Slurm hosts or nodelist expression. Defaults to SLURM_JOB_NODELIST.",
    )
    parser.add_argument("--allowed-hosts", default=DEFAULT_ALLOWED_HOSTS)
    parser.add_argument("--required-train-nodes", type=int, default=DEFAULT_REQUIRED_TRAIN_NODES)
    parser.add_argument("--required-inference-replicas", type=int, default=DEFAULT_REQUIRED_INFER_REPLICAS)
    parser.add_argument("--production-config", type=Path, default=DEFAULT_PRODUCTION_CONFIG)
    parser.add_argument("--trainer-template", type=Path, default=DEFAULT_TRAINER_TEMPLATE)
    parser.add_argument("--sampler-contract-sweep-result", type=Path, default=DEFAULT_SAMPLER_CONTRACT_SWEEP_RESULT)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--patches-path", type=Path, default=DEFAULT_PATCHES_PATH)
    parser.add_argument(
        "--inference-template",
        dest="inference_templates",
        action="append",
        type=Path,
        default=None,
        help="Inference launch template to check. May be repeated; defaults to all production templates.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check only production config/template/topology readiness, without reading prior run logs.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.inference_templates is None:
        args.inference_templates = DEFAULT_INFERENCE_TEMPLATES
    return args


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        preflight = production_preflight(args)
        if args.json:
            print(json.dumps(asdict(preflight), indent=2, sort_keys=True))
        else:
            print_preflight_markdown(preflight)
        if not preflight.full_production_ready:
            raise SystemExit(2)
        return
    report = audit(args)
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print_markdown(report)
    if not report.local_two_node_gates_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
