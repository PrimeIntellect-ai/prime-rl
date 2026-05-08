import asyncio
import random
from dataclasses import dataclass
from pathlib import Path

import httpx
import verifiers as vf

from prime_rl.configs.es import ESConfig
from prime_rl.orchestrator.envs import TrainEnv
from prime_rl.orchestrator.vf_utils import get_model_completion_len, get_num_turns
from prime_rl.trainer.es.candidates import Candidate
from prime_rl.utils.client import load_lora_adapter, setup_admin_clients, setup_clients, unload_lora_adapter
from prime_rl.utils.logger import ProgressTracker, get_logger


@dataclass
class CandidateRolloutResult:
    candidate_idx: int
    reward: float
    num_rollouts: int
    failed_rollouts: int
    generated_tokens: int
    turns: int


def sample_examples(env: TrainEnv, count: int, seed: int) -> list[dict]:
    dataset = env.get_dataset(seed=seed)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)
    if hasattr(dataset, "to_list"):
        rows = dataset.to_list()
    else:
        rows = list(dataset)
    if not rows:
        return []
    if count <= len(rows):
        return rows[:count]
    rng = random.Random(seed)
    return [rng.choice(rows) for _ in range(count)]


async def start_train_envs(config: ESConfig) -> list[TrainEnv]:
    envs = [TrainEnv(env_config) for env_config in config.train.env]
    log_dir = config.output_dir / "logs" / "envs"
    for env in envs:
        await env.start(log_dir=log_dir, log_level=config.log.level, json_logging=config.log.json_logging)
    return envs


def shutdown_train_envs(envs: list[TrainEnv]) -> None:
    for env in envs:
        env.shutdown()


def build_clients(config: ESConfig) -> list[vf.ClientConfig]:
    return setup_clients(config.client)


def build_admin_clients(config: ESConfig) -> list[httpx.AsyncClient]:
    return setup_admin_clients(config.client)


async def close_admin_clients(admin_clients: list[httpx.AsyncClient]) -> None:
    for client in admin_clients:
        await client.aclose()


async def load_candidate_adapters(
    admin_clients: list[httpx.AsyncClient],
    candidate_paths: dict[int, Path],
    candidate_names: dict[int, str],
) -> None:
    await asyncio.gather(
        *[
            load_lora_adapter(admin_clients, candidate_names[candidate_idx], candidate_path.resolve())
            for candidate_idx, candidate_path in candidate_paths.items()
        ]
    )


async def unload_candidate_adapters(
    admin_clients: list[httpx.AsyncClient],
    candidate_names: dict[int, str],
) -> None:
    await asyncio.gather(*[unload_lora_adapter(admin_clients, name) for name in candidate_names.values()])


async def init_lora_slots(
    admin_clients: list[httpx.AsyncClient],
    theta_path: Path,
    specs: list[dict],
    adapter_config: dict,
    slots: list[dict],
) -> None:
    payload = {
        "theta_path": theta_path.resolve().as_posix(),
        "specs": specs,
        "adapter_config": adapter_config,
        "slots": slots,
    }
    responses = await asyncio.gather(*[client.post("/es/init_lora_slots", json=payload) for client in admin_clients])
    for response in responses:
        response.raise_for_status()


async def materialize_lora_slots(
    admin_clients: list[httpx.AsyncClient],
    slots: list[dict],
    sigma: float,
) -> None:
    payload = {"slots": slots, "sigma": sigma}
    responses = await asyncio.gather(
        *[client.post("/es/materialize_lora_slots", json=payload, timeout=None) for client in admin_clients]
    )
    for response in responses:
        response.raise_for_status()


async def update_lora_slot_theta(
    admin_clients: list[httpx.AsyncClient],
    candidates: list[dict],
    rewards: list[float],
    lr: float,
    normalization: str,
    mirrored: bool,
    sigma: float,
) -> None:
    payload = {
        "candidates": candidates,
        "rewards": rewards,
        "lr": lr,
        "normalization": normalization,
        "mirrored": mirrored,
        "sigma": sigma,
    }
    responses = await asyncio.gather(
        *[client.post("/es/update_lora_theta", json=payload, timeout=None) for client in admin_clients]
    )
    for response in responses:
        response.raise_for_status()


async def evaluate_candidate(
    candidate: Candidate,
    candidate_name: str,
    envs: list[TrainEnv],
    examples_by_env: dict[str, list[dict]],
    clients: list[vf.ClientConfig],
    config: ESConfig,
    *,
    step: int,
) -> CandidateRolloutResult:
    semaphore = asyncio.Semaphore(config.train.max_concurrent_rollouts_per_rank)
    logger = get_logger()
    outputs = []
    failed = 0
    client_cursor = 0

    async def run_one(env: TrainEnv, example: dict, rollout_idx: int):
        nonlocal failed, client_cursor
        async with semaphore:
            client = clients[client_cursor % len(clients)]
            client_cursor += 1
            cache_salt = f"es-{step}-{candidate.idx}-{rollout_idx}"
            try:
                if env.requires_group_scoring:
                    group = await env.run_group(
                        client=client,
                        example=example,
                        model_name=candidate_name,
                        rollouts_per_example=config.train.rollouts_per_example,
                        cache_salt=cache_salt,
                    )
                    return group
                rollout = await env.run_rollout(
                    client=client,
                    example=example,
                    model_name=candidate_name,
                    cache_salt=cache_salt,
                )
                return [rollout]
            except Exception as exc:
                failed += 1
                logger.warning(f"Candidate {candidate.idx} rollout failed in {env.name}: {exc}")
                return []

    coros = []
    for env in envs:
        examples = examples_by_env[env.name]
        repeats = 1 if env.requires_group_scoring else config.train.rollouts_per_example
        for example in examples:
            for rollout_idx in range(repeats):
                coros.append(run_one(env, example, rollout_idx))

    for group in await asyncio.gather(*coros):
        outputs.extend(group)

    valid = [o for o in outputs if o.get("error") is None and o.get("trajectory")]
    failed += len(outputs) - len(valid)
    rewards = [float(o.get("reward") or 0.0) for o in valid]
    reward = sum(rewards) / len(rewards) if rewards else 0.0
    return CandidateRolloutResult(
        candidate_idx=candidate.idx,
        reward=reward,
        num_rollouts=len(valid),
        failed_rollouts=failed,
        generated_tokens=sum(get_model_completion_len(o) for o in valid),
        turns=sum(get_num_turns(o) for o in valid),
    )


async def evaluate_candidate_chunk(
    candidates: list[Candidate],
    candidate_names: dict[int, str],
    envs: list[TrainEnv],
    examples_by_env: dict[str, list[dict]],
    clients: list[vf.ClientConfig],
    config: ESConfig,
    *,
    step: int,
) -> list[CandidateRolloutResult]:
    pbar = ProgressTracker(
        total=len(candidates),
        desc=f"Evaluating ES candidates (step {step})",
        json_logging=config.log.json_logging,
        step=step,
    )

    async def run(candidate: Candidate):
        try:
            return await evaluate_candidate(
                candidate,
                candidate_names[candidate.idx],
                envs,
                examples_by_env,
                clients,
                config,
                step=step,
            )
        finally:
            pbar.update(1)

    try:
        return await asyncio.gather(*(run(candidate) for candidate in candidates))
    finally:
        pbar.close()
