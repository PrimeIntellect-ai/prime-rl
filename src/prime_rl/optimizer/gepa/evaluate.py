from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from prime_rl.eval.registry import get_benchmark_dataset
from prime_rl.orchestrator.client import generate_completion, setup_client
from prime_rl.orchestrator.config import ClientConfig, ModelConfig as OrchestratorModelConfig, SamplingConfig
from prime_rl.orchestrator.utils import compute_rewards, parse_completion_tokens, parse_completions
from prime_rl.utils.logger import get_logger


@dataclass
class PromptScore:
    prompt: str
    avg_reward: float
    pass_at_k: float | None
    avg_completion_len: float
    meta: dict[str, Any]


async def _score_on_benchmark(
    client: AsyncOpenAI,
    benchmark: str,
    system_prompt: str,
    model_config: OrchestratorModelConfig,
    sampling: SamplingConfig,
    subset_size: int,
    rollouts_per_prompt: int,
) -> tuple[float, float | None, float]:
    logger = get_logger()
    dataset = get_benchmark_dataset(benchmark)
    dataset = dataset.select(range(min(len(dataset), subset_size)))

    prompts = [item["prompt"] for item in dataset]
    prompts = [p for p in prompts for _ in range(rollouts_per_prompt)]
    problem_ids = list(range(len(dataset)))
    problem_ids = [pid for pid in problem_ids for _ in range(rollouts_per_prompt)]

    batch_messages = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
                      for p in prompts]

    # Generate
    chat_completions = await asyncio.gather(
        *(generate_completion(client, model_config, sampling, messages) for messages in batch_messages)
    )

    # Stats
    completion_lengths = [len(parse_completion_tokens(c)) for c in chat_completions]
    avg_completion_len = sum(completion_lengths) / max(1, len(completion_lengths))

    completions = parse_completions(chat_completions)
    task_types = [item.get("task_type", "") for item in dataset]
    verification_infos = [item.get("verification_info", "{}") for item in dataset]
    # Duplicate for k samples
    task_types = [t for t in task_types for _ in range(rollouts_per_prompt)]
    verification_infos = [v for v in verification_infos for _ in range(rollouts_per_prompt)]

    try:
        import json
        verification_infos = [json.loads(v) if isinstance(v, str) else v for v in verification_infos]
    except Exception:
        logger.warning("Failed to parse some verification_info entries; using raw values.")

    rewards = compute_rewards(completions, task_types, verification_infos)

    # pass@k for binary rewards
    unique = set(rewards)
    pass_at_k = None
    if unique.issubset({0, 1, 0.0, 1.0}):
        k = rollouts_per_prompt
        rows: dict[int, list[float]] = {}
        for pid, r in zip(problem_ids, rewards):
            rows.setdefault(pid, []).append(float(r))
        solved = [any(x == 1.0 for x in rs) for rs in rows.values()]
        pass_at_k = sum(solved) / max(1, len(solved))

    avg_reward = float(sum(map(float, rewards)) / max(1, len(rewards)))
    return avg_reward, pass_at_k, float(avg_completion_len)


async def score_prompt(
    system_prompt: str,
    client_cfg: ClientConfig,
    model_cfg: OrchestratorModelConfig,
    benchmark: str,
    subset_size: int,
    rollouts_per_prompt: int,
    max_tokens: int | None,
    min_tokens: int,
) -> PromptScore:
    logger = get_logger()
    client = setup_client(client_cfg)

    sampling = SamplingConfig(
        temperature=1.0,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        seed=None,
    )

    # Single benchmark specified by caller
    avg_reward, pass_at_k, avg_len = await _score_on_benchmark(
        client,
        benchmark=benchmark,
        system_prompt=system_prompt,
        model_config=model_cfg,
        sampling=sampling,
        subset_size=subset_size,
        rollouts_per_prompt=rollouts_per_prompt,
    )

    return PromptScore(
        prompt=system_prompt,
        avg_reward=avg_reward,
        pass_at_k=pass_at_k,
        avg_completion_len=avg_len,
        meta={},
    )


async def score_prompt_dry_run(system_prompt: str, subset_size: int) -> PromptScore:
    # Deterministic-ish fake scoring for local CPU testing without dependencies
    base = sum(ord(c) for c in system_prompt) % 1000 / 1000.0
    avg_reward = 0.3 + 0.5 * base
    pass_at_k = None
    avg_completion_len = 20.0 + (len(system_prompt) % 17)
    return PromptScore(system_prompt, avg_reward, pass_at_k, avg_completion_len, meta={"dry_run": True})


async def score_prompt_instances(
    system_prompt: str,
    client_cfg: ClientConfig,
    model_cfg: OrchestratorModelConfig,
    benchmark: str,
    num_instances: int,
    rollouts_per_prompt: int,
    max_tokens: int | None,
    min_tokens: int,
    offset: int = 0,
) -> list[float]:
    """Return per-instance average reward for a contiguous slice of the dataset."""
    logger = get_logger()
    client = setup_client(client_cfg)
    sampling = SamplingConfig(
        temperature=1.0,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        seed=None,
    )

    dataset = get_benchmark_dataset(benchmark)
    start = max(0, offset)
    end = min(len(dataset), start + num_instances)
    dataset = dataset.select(range(start, end))

    prompts = [item["prompt"] for item in dataset]
    prompts = [p for p in prompts for _ in range(rollouts_per_prompt)]
    problem_ids = list(range(len(dataset)))
    problem_ids = [pid for pid in problem_ids for _ in range(rollouts_per_prompt)]
    batch_messages = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in prompts]

    chat_completions = await asyncio.gather(
        *(generate_completion(client, model_cfg, sampling, messages) for messages in batch_messages)
    )
    completions = parse_completions(chat_completions)
    task_types = [item.get("task_type", "") for item in dataset]
    verification_infos = [item.get("verification_info", "{}") for item in dataset]
    task_types = [t for t in task_types for _ in range(rollouts_per_prompt)]
    try:
        import json
        verification_infos = [json.loads(v) if isinstance(v, str) else v for v in verification_infos]
    except Exception:
        pass
    verification_infos = [v for v in verification_infos for _ in range(rollouts_per_prompt)]
    rewards = compute_rewards(completions, task_types, verification_infos)

    per_problem: dict[int, list[float]] = {}
    for pid, r in zip(problem_ids, rewards):
        per_problem.setdefault(pid, []).append(float(r))
    return [sum(rs) / len(rs) for _, rs in sorted(per_problem.items())]


async def score_prompt_instances_dry_run(system_prompt: str, num_instances: int) -> list[float]:
    base = (sum(ord(c) for c in system_prompt) % 1000) / 1000.0
    return [0.3 + 0.5 * ((base + i * 0.013) % 1.0) for i in range(num_instances)]


