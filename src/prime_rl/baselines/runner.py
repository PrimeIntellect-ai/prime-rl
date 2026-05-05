from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prime_rl.baselines.config import BaselineConfig
from prime_rl.baselines.metrics import summarize_records
from prime_rl.baselines.provision import Endpoint, InferenceProvisioner
from prime_rl.baselines.records import flatten_output, token_summary_row
from prime_rl.utils.logger import ProgressTracker

STATE_COLUMNS = ["trajectory", "sampling_args"]
RUBRIC_CHILD_ATTRS = ("rubric", "rubrics", "grader", "matcher", "math_rubric", "judge_rubric")
VLLM_EXTRA_BODY_KEYS = frozenset(
    {
        "cache_salt",
        "top_k",
        "min_p",
        "return_token_ids",
        "repetition_penalty",
        "min_tokens",
        "best_of",
        "chat_template_kwargs",
        "include_reasoning",
        "thinking_token_budget",
        "bad_words",
    }
)


def _module_name(env_id: str) -> str:
    return env_id.replace("-", "_").split("/")[-1]


def _default_paths(config: BaselineConfig) -> tuple[Path | None, list[Path]]:
    module = _module_name(config.env_id)
    env_paths = list(config.env_paths)
    sibling_verifiers = Path.cwd().parent / "verifiers"
    tmp_verifiers = Path("/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs")

    for root in (sibling_verifiers, tmp_verifiers):
        candidate = root / "environments" / module
        if candidate.exists() and candidate not in env_paths:
            env_paths.append(candidate)

    verifiers_path = config.verifiers_path
    if verifiers_path is None and module.startswith("hf_") and tmp_verifiers.exists():
        verifiers_path = tmp_verifiers

    return verifiers_path, env_paths


def prepare_import_paths(config: BaselineConfig) -> None:
    verifiers_path, env_paths = _default_paths(config)
    paths: list[Path] = []
    if verifiers_path is not None:
        paths.append(verifiers_path)
    paths.extend(env_paths)
    for path in reversed(paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _make_client_config(config: BaselineConfig, endpoint: Endpoint, max_connections: int):
    import verifiers as vf

    profile = config.api_profile
    if profile is None and config.launch.mode in {"local", "srun"}:
        profile = "vllm_permissive"
    return vf.ClientConfig(
        client_type=config.client_type,
        api_base_url=endpoint.base_url,
        api_key_var=endpoint.api_key_var,
        profile=profile,
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )


def _requires_group_scoring(env: Any) -> bool:
    rubric = getattr(env, "rubric", None)
    if rubric is None:
        return False
    get_funcs = getattr(rubric, "_get_reward_funcs", None)
    is_group = getattr(rubric, "_is_group_func", None)
    if get_funcs is None or is_group is None:
        return False
    return any(is_group(func) for func in get_funcs())


def _collect_judge_cache_stats(obj: Any) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    seen: set[int] = set()

    def visit(value: Any, path: str) -> None:
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        cache_stats = getattr(value, "judge_cache_stats", None)
        if isinstance(cache_stats, dict):
            stats[path] = dict(cache_stats)

        for attr in RUBRIC_CHILD_ATTRS:
            if not hasattr(value, attr):
                continue
            child = getattr(value, attr)
            if isinstance(child, dict):
                for key, item in child.items():
                    visit(item, f"{path}.{attr}.{key}")
            elif isinstance(child, (list, tuple)):
                for index, item in enumerate(child):
                    visit(item, f"{path}.{attr}[{index}]")
            elif child is not None:
                visit(child, f"{path}.{attr}")

    visit(obj, "env")
    return stats


def _sampling_args(config: BaselineConfig) -> dict[str, Any]:
    sampling = {}
    flattened_extra_body = {}
    explicit_extra_body = dict(config.sampling_args.get("extra_body", {}))
    for key, value in config.sampling_args.items():
        if key in {"extra_body", "n"}:
            continue
        elif key in VLLM_EXTRA_BODY_KEYS:
            flattened_extra_body[key] = value
        else:
            sampling[key] = value
    if "max_tokens" in sampling:
        max_tokens = sampling.pop("max_tokens")
        sampling.setdefault("max_completion_tokens", max_tokens)
    extra_body = {**flattened_extra_body, **explicit_extra_body}
    if extra_body:
        sampling["extra_body"] = extra_body
    sampling["n"] = 1
    return sampling


def _example_id(example: dict[str, Any]) -> str:
    return str(example.get("example_id", example.get("id")))


def _eval_examples(config: BaselineConfig, env: Any) -> list[dict[str, Any]]:
    if not config.record_ids:
        return env.get_eval_dataset(n=config.num_examples, seed=config.seed).to_list()

    examples = env.get_eval_dataset(n=-1, seed=config.seed).to_list()
    by_id = {_example_id(example): example for example in examples}
    missing = [record_id for record_id in config.record_ids if record_id not in by_id]
    if missing:
        available_preview = ", ".join(sorted(by_id)[:10])
        raise ValueError(
            f"Requested record_ids not found: {missing}. "
            f"Loaded {len(examples)} eval examples; first available IDs: {available_preview}"
        )
    return [by_id[record_id] for record_id in config.record_ids]


async def _run_rollouts(
    config: BaselineConfig,
    env: Any,
    endpoint: Endpoint,
    sampling: dict[str, Any],
) -> list[tuple[int, Any]]:
    import verifiers as vf
    from verifiers.clients import resolve_client
    from verifiers.utils.async_utils import maybe_retry
    from verifiers.utils.save_utils import state_to_output

    examples = _eval_examples(config, env)
    client = _make_client_config(config, endpoint, config.max_concurrency)
    generation_semaphore = asyncio.Semaphore(config.max_concurrency)
    scoring_semaphore = asyncio.Semaphore(config.score_max_concurrency or config.max_concurrency)
    requires_group = _requires_group_scoring(env)
    resolved_client = resolve_client(client)
    total_rollouts = len(examples) * config.rollouts_per_example
    decoupled = not requires_group and getattr(env, "env_client", None) is None
    generation_pbar = (
        ProgressTracker(total=total_rollouts, desc="Baseline generations", position=0) if decoupled else None
    )
    scoring_pbar = ProgressTracker(total=total_rollouts, desc="Baseline scoring", position=1) if decoupled else None
    rollout_pbar = (
        ProgressTracker(total=total_rollouts, desc="Baseline rollouts", position=0) if not decoupled else None
    )

    async def run_one_decoupled(example: dict[str, Any], trial_index: int) -> tuple[int, Any]:
        rollout_input = vf.RolloutInput(**example)

        async def run_attempt():
            async with generation_semaphore:
                state = await env.rollout(
                    rollout_input,
                    resolved_client,
                    config.model,
                    sampling,
                )
            if generation_pbar is not None:
                generation_pbar.update(1)

            async with scoring_semaphore:
                if env.score_rollouts:
                    await env.rubric.score_rollout(state)
                else:
                    await env.rubric.dummy_score_rollout(state)
                await env.rubric.cleanup(state)
            if scoring_pbar is not None:
                scoring_pbar.update(1)

            return state

        state = await maybe_retry(run_attempt, max_retries=config.max_retries)()
        return trial_index, state_to_output(state, STATE_COLUMNS)

    async def run_one_coupled(example: dict[str, Any], trial_index: int) -> tuple[int, Any]:
        async with generation_semaphore:
            output = await env.run_rollout(
                vf.RolloutInput(**example),
                client=client,
                model=config.model,
                sampling_args=sampling,
                max_retries=config.max_retries,
                state_columns=STATE_COLUMNS,
            )
        if rollout_pbar is not None:
            rollout_pbar.update(1)
        return trial_index, output

    async def run_group(example: dict[str, Any]) -> list[tuple[int, Any]]:
        async with generation_semaphore:
            outputs = await env.run_group(
                [vf.RolloutInput(**example) for _ in range(config.rollouts_per_example)],
                client=client,
                model=config.model,
                sampling_args=sampling,
                max_retries=config.max_retries,
                state_columns=STATE_COLUMNS,
            )
        if rollout_pbar is not None:
            rollout_pbar.update(len(outputs))
        return list(enumerate(outputs))

    try:
        if requires_group:
            grouped = await asyncio.gather(*(run_group(example) for example in examples))
            return [item for group in grouped for item in group]

        run_one = run_one_coupled if getattr(env, "env_client", None) is not None else run_one_decoupled
        tasks = [
            run_one(example, trial_index) for example in examples for trial_index in range(config.rollouts_per_example)
        ]
        return list(await asyncio.gather(*tasks))
    finally:
        for pbar in (generation_pbar, scoring_pbar, rollout_pbar):
            if pbar is not None:
                pbar.close()


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    from verifiers.utils.save_utils import make_serializable

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            json.dump(row, f, default=make_serializable)
            f.write("\n")


def _question_rows(records: list[dict[str, Any]], config: BaselineConfig) -> list[dict[str, Any]]:
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_example[str(row["example_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for example_id, group in sorted(by_example.items()):
        group.sort(key=lambda r: int(r.get("trial_index", 0)))
        correct_votes = sum(1 for r in group if r.get("correct") is True)
        first = group[0]
        info = first.get("info") if isinstance(first.get("info"), dict) else {}
        is_grouped = len(group) > 1
        first_hit = (
            next((i for i, r in enumerate(group, start=1) if r.get("correct") is True), None) if is_grouped else None
        )
        rows.append(
            {
                "record_id": example_id,
                "domain": info.get("domain") or info.get("Domain") or info.get("High-level domain"),
                "subdomain": info.get("subdomain") or info.get("Subdomain"),
                "target": first.get("target"),
                "raw_answer": first.get("response"),
                "parse_success": 1.0 if first.get("parsed_answer") else 0.0,
                "accuracy": 1.0 if first.get("correct") else 0.0,
                "correct_votes": correct_votes if is_grouped else None,
                "total_votes": len(group) if is_grouped else None,
                "first_hit_index": first_hit,
                "per_draw_success_rate": correct_votes / len(group) if is_grouped else None,
                "majority_correct": correct_votes > len(group) / 2 if is_grouped else None,
                "best_of_n_correct": correct_votes > 0 if is_grouped else None,
                "dataset": config.dataset_label,
                "model": config.model.split("/")[-1],
                "effort": None,
                "protocol": config.protocol,
                "seed": config.seed,
                "eval_id": config.run_id,
                "output_tokens": sum(float(r.get("output_tokens") or 0.0) for r in group),
                "input_tokens": sum(float(r.get("input_tokens") or 0.0) for r in group),
                "reasoning_tokens": None,
            }
        )
    return rows


def run_baseline(config: BaselineConfig) -> dict[str, Any]:
    prepare_import_paths(config)
    import verifiers as vf

    run_id = config.run_id or uuid.uuid4().hex[:12]
    config.run_id = run_id
    config.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    env_args = dict(config.env_args)
    if env_args.get("dataset_streaming") and int(env_args.get("num_eval_examples", -1)) < 0:
        env_args["num_eval_examples"] = config.num_examples
    env = vf.load_environment(config.env_id, **env_args)
    sampling = _sampling_args(config)
    with InferenceProvisioner(config) as endpoint:
        outputs = asyncio.run(_run_rollouts(config, env, endpoint, sampling))
    elapsed_s = time.perf_counter() - t0

    records = [
        flatten_output(
            output=output,
            env=env,
            run_id=run_id,
            env_id=config.env_id,
            protocol=config.protocol,
            dataset=config.dataset_label,
            model=config.model,
            seed=config.seed,
            trial_index=trial_index,
            success_threshold=config.success_threshold,
        )
        for trial_index, output in outputs
    ]
    records.sort(key=lambda r: (str(r["example_id"]), int(r["trial_index"])))
    summary = summarize_records(records, config.ks)
    summary.update(
        {
            "run_id": run_id,
            "env_id": config.env_id,
            "protocol": config.protocol,
            "dataset": config.dataset_label,
            "model": config.model,
            "elapsed_s": elapsed_s,
            "config": {
                "num_examples": config.num_examples,
                "record_ids": config.record_ids,
                "rollouts_per_example": config.rollouts_per_example,
                "max_concurrency": config.max_concurrency,
                "score_max_concurrency": config.score_max_concurrency,
                "sampling": sampling,
                "env_args": env_args,
            },
            "judge_cache": _collect_judge_cache_stats(env),
        }
    )

    raw_outputs = [output for _, output in outputs]
    question_rows = _question_rows(records, config)
    data_dir = config.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.output_dir / "raw_rollouts.jsonl", raw_outputs)
    _write_jsonl(config.output_dir / "records.jsonl", records)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (data_dir / f"{config.protocol}.json").write_text(json.dumps(question_rows, indent=2))
    (data_dir / "tokens.json").write_text(
        json.dumps(
            [
                token_summary_row(
                    records,
                    model=config.model,
                    protocol=config.protocol,
                    dataset=config.dataset_label,
                    seed=config.seed,
                )
            ],
            indent=2,
        )
    )
    if config.fail_on_error and any(record.get("error") for record in records):
        error_count = sum(1 for record in records if record.get("error"))
        raise RuntimeError(f"{error_count}/{len(records)} rollouts errored; artifacts written to {config.output_dir}")
    return summary
