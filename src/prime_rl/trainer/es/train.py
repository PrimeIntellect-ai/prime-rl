import asyncio
import gc
import json
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import tomli_w
import torch
import torch.distributed as dist

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive import
from prime_rl.configs.es import ESConfig
from prime_rl.trainer.es.candidates import estimate_gradient, make_candidates, noise_like
from prime_rl.trainer.es.ckpt import latest_es_step, load_es_state, maybe_clean_es_checkpoints, save_es_state
from prime_rl.trainer.es.lora_materialize import build_adapter_template, write_adapter_from_theta
from prime_rl.trainer.es.rollout import (
    build_admin_clients,
    build_clients,
    close_admin_clients,
    evaluate_candidate_chunk,
    init_lora_slots,
    load_candidate_adapters,
    materialize_lora_slots,
    sample_examples,
    shutdown_train_envs,
    start_train_envs,
    unload_candidate_adapters,
    update_lora_slot_theta,
)
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.utils import GarbageCollection, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.process import set_proc_title


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def rank_assigned_candidates(candidates, rank: int, world_size: int):
    return [candidate for i, candidate in enumerate(candidates) if i % world_size == rank]


def all_reduce_float(value: float, op: dist.ReduceOp) -> float:
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def es_compute_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def write_candidate_adapters(
    chunk,
    step_chunk_root: Path,
    template,
    theta: torch.Tensor,
    sigma: float,
    max_workers: int,
) -> dict[int, Path]:
    candidate_paths: dict[int, Path] = {}
    pending: list[Future[None]] = []
    worker_count = max(1, min(max_workers, len(chunk)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for candidate in chunk:
            candidate_theta = theta + candidate.sign * sigma * noise_like(theta, candidate.seed)
            adapter_dir = step_chunk_root / candidate.name_suffix
            candidate_paths[candidate.idx] = adapter_dir
            pending.append(executor.submit(write_adapter_from_theta, adapter_dir, template, candidate_theta))
            if len(pending) >= worker_count:
                pending.pop(0).result()
        for future in pending:
            future.result()

    return candidate_paths


def serialize_specs(template) -> list[dict]:
    return [
        {
            "name": spec.name,
            "shape": list(spec.shape),
            "dtype": str(spec.dtype),
            "numel": spec.numel,
        }
        for spec in template.specs
    ]


def slot_definitions(chunk_size: int) -> list[dict]:
    return [{"lora_name": f"es_slot_{idx}", "lora_int_id": 10_000 + idx, "slot": idx} for idx in range(chunk_size)]


def slot_payload_for_chunk(chunk, slots: list[dict]) -> list[dict]:
    payload = []
    for slot, candidate in zip(slots, chunk, strict=True):
        payload.append(
            {
                "lora_name": slot["lora_name"],
                "lora_int_id": slot["lora_int_id"],
                "candidate_idx": candidate.idx,
                "seed": candidate.seed,
                "sign": candidate.sign,
            }
        )
    return payload


def candidate_payload(candidates) -> list[dict]:
    return [{"idx": candidate.idx, "seed": candidate.seed, "sign": candidate.sign} for candidate in candidates]


async def train_async(config: ESConfig) -> None:
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Starting synchronous ES trainer in {world} in {config.output_dir}")

    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload,
    )
    torch.set_float32_matmul_precision(config.matmul_precision)

    if world.is_master:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config_dir = get_config_dir(config.output_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "es.toml", "wb") as f:
            tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)
    dist.barrier()

    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)
    heart = Heartbeat(config.heartbeat.url) if config.heartbeat is not None and world.is_master else None
    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    template = None
    theta = None
    if world.is_master:
        device = es_compute_device()
        template = build_adapter_template(config.output_dir, config.model, device=device)
        theta = template.theta

        if config.ckpt and config.ckpt.resume_step is not None:
            resume_step = (
                latest_es_step(config.output_dir) if config.ckpt.resume_step == -1 else config.ckpt.resume_step
            )
            if resume_step is not None:
                state = load_es_state(config.output_dir, resume_step)
                theta = state["theta"].to(device=device, dtype=torch.float32)
                progress = Progress(step=int(state["step"]))
                logger.info(f"Resumed ES state from step {progress.step}")
            else:
                progress = Progress()
        else:
            progress = Progress()
    else:
        progress = Progress()

    progress_box = [progress]
    dist.broadcast_object_list(progress_box, src=0)
    progress = progress_box[0]

    envs = await start_train_envs(config)
    clients = build_clients(config)
    admin_clients = build_admin_clients(config) if world.is_master else []

    metrics_path = config.output_dir / "metrics.jsonl"
    adapter_root = config.output_dir / "adapters"
    candidate_root = adapter_root / "candidates_tmp"
    final_adapter = config.output_dir / "adapter"
    use_lora_slots = config.algorithm.adapter_transport == "slots"
    slots: list[dict] = []
    if world.is_master and use_lora_slots:
        assert template is not None and theta is not None
        slots = slot_definitions(config.algorithm.candidate_chunk_size)
        theta_init_path = config.output_dir / "es_lora_theta_init.pt"
        torch.save({"theta": theta.detach().cpu()}, theta_init_path)
        t0 = time.perf_counter()
        await init_lora_slots(admin_clients, theta_init_path, serialize_specs(template), template.adapter_config, slots)
        logger.info(f"Initialized {len(slots)} persistent ES LoRA slots in {time.perf_counter() - t0:.2f}s")
        theta_init_path.unlink(missing_ok=True)

    try:
        while config.max_steps is None or progress.step < config.max_steps:
            step = progress.step + 1
            if gc_handler is not None:
                gc_handler.run(step)

            step_start = time.perf_counter()
            examples_by_env = {
                env.name: sample_examples(env, config.train.examples_per_env, config.algorithm.seed + 100_000 + step)
                for env in envs
            }

            candidates = make_candidates(
                config.algorithm.seed,
                step,
                config.algorithm.population_size,
                config.algorithm.mirrored,
            )
            chunk_size = min(config.algorithm.candidate_chunk_size, len(candidates))
            all_results = []
            adapter_write_s = 0.0
            adapter_load_s = 0.0
            adapter_unload_s = 0.0
            adapter_slot_update_s = 0.0
            generation_s = 0.0

            for chunk_start in range(0, len(candidates), chunk_size):
                chunk = candidates[chunk_start : chunk_start + chunk_size]
                step_chunk_root = candidate_root / f"step_{step:06d}" / f"chunk_{chunk_start:04d}"
                candidate_paths: dict[int, Path] = {}
                if use_lora_slots:
                    chunk_slots = slots[: len(chunk)] if world.is_master else []
                    candidate_names = {
                        candidate.idx: chunk_slots[i]["lora_name"] if world.is_master else ""
                        for i, candidate in enumerate(chunk)
                    }
                else:
                    candidate_names = {candidate.idx: f"es_step_{step}_cand_{candidate.idx}" for candidate in chunk}

                if world.is_master:
                    assert template is not None and theta is not None and config.model.lora is not None
                    if use_lora_slots:
                        t0 = time.perf_counter()
                        await materialize_lora_slots(
                            admin_clients,
                            slot_payload_for_chunk(chunk, slots[: len(chunk)]),
                            config.algorithm.sigma,
                        )
                        adapter_slot_update_s += time.perf_counter() - t0
                    else:
                        if step_chunk_root.exists():
                            shutil.rmtree(step_chunk_root)
                        t0 = time.perf_counter()
                        candidate_paths = write_candidate_adapters(
                            chunk,
                            step_chunk_root,
                            template,
                            theta,
                            config.algorithm.sigma,
                            config.algorithm.adapter_write_workers,
                        )
                        adapter_write_s += time.perf_counter() - t0

                        t0 = time.perf_counter()
                        await load_candidate_adapters(admin_clients, candidate_paths, candidate_names)
                        adapter_load_s += time.perf_counter() - t0

                names_box = [candidate_names]
                dist.broadcast_object_list(names_box, src=0)
                candidate_names = names_box[0]
                dist.barrier()

                local_chunk = rank_assigned_candidates(chunk, world.rank, world.world_size)
                t0 = time.perf_counter()
                local_results = await evaluate_candidate_chunk(
                    local_chunk,
                    candidate_names,
                    envs,
                    examples_by_env,
                    clients,
                    config,
                    step=step,
                )
                local_generation_s = time.perf_counter() - t0
                generation_s += all_reduce_float(local_generation_s, dist.ReduceOp.MAX)

                gathered: list[list] = [None for _ in range(world.world_size)]  # type: ignore[list-item]
                dist.all_gather_object(gathered, local_results)
                if world.is_master:
                    for rank_results in gathered:
                        all_results.extend(rank_results)
                    if not use_lora_slots:
                        t0 = time.perf_counter()
                        await unload_candidate_adapters(admin_clients, candidate_names)
                        adapter_unload_s += time.perf_counter() - t0
                        if not config.algorithm.keep_candidate_adapters:
                            shutil.rmtree(step_chunk_root, ignore_errors=True)
                dist.barrier()

            update_s = 0.0
            payload = None
            if world.is_master:
                assert theta is not None and template is not None and config.model.lora is not None
                rewards = {result.candidate_idx: result.reward for result in all_results}
                t0 = time.perf_counter()
                grad = estimate_gradient(
                    theta,
                    candidates,
                    rewards,
                    config.algorithm.sigma,
                    config.algorithm.reward_normalization,
                    config.algorithm.mirrored,
                )
                theta = theta + config.algorithm.lr * grad
                if theta.device.type == "cuda":
                    torch.cuda.synchronize(theta.device)
                update_s = time.perf_counter() - t0
                if use_lora_slots:
                    ordered_rewards = [rewards[candidate.idx] for candidate in candidates]
                    t0 = time.perf_counter()
                    await update_lora_slot_theta(
                        admin_clients,
                        candidate_payload(candidates),
                        ordered_rewards,
                        config.algorithm.lr,
                        config.algorithm.reward_normalization,
                        config.algorithm.mirrored,
                        config.algorithm.sigma,
                    )
                    adapter_slot_update_s += time.perf_counter() - t0

                progress.step = step
                progress.total_samples += sum(result.num_rollouts for result in all_results)
                progress.total_tokens += sum(result.generated_tokens for result in all_results)

                reward_values = [result.reward for result in all_results]
                generated_tokens = sum(result.generated_tokens for result in all_results)
                failed_rollouts = sum(result.failed_rollouts for result in all_results)
                valid_rollouts = sum(result.num_rollouts for result in all_results)
                step_s = time.perf_counter() - step_start
                grad_norm = float(torch.linalg.vector_norm(grad).item())
                step_norm = float(torch.linalg.vector_norm(config.algorithm.lr * grad).item())
                payload = {
                    "phase": "train",
                    "step": step,
                    "population_size": len(candidates),
                    "candidate_chunk_size": chunk_size,
                    "candidate_reward_mean": float(sum(reward_values) / len(reward_values)) if reward_values else 0.0,
                    "candidate_reward_min": float(min(reward_values)) if reward_values else 0.0,
                    "candidate_reward_max": float(max(reward_values)) if reward_values else 0.0,
                    "generated_tokens": int(generated_tokens),
                    "tokens_per_second": float(generated_tokens / generation_s) if generation_s > 0 else 0.0,
                    "valid_rollouts": int(valid_rollouts),
                    "failed_rollouts": int(failed_rollouts),
                    "generation_s": generation_s,
                    "adapter_write_s": adapter_write_s,
                    "adapter_write_workers": config.algorithm.adapter_write_workers,
                    "adapter_load_s": adapter_load_s,
                    "adapter_unload_s": adapter_unload_s,
                    "adapter_slot_update_s": adapter_slot_update_s,
                    "update_s": update_s,
                    "adapter_transport": config.algorithm.adapter_transport,
                    "iteration_wall_clock_s": step_s,
                    "grad_norm": grad_norm,
                    "step_norm": step_norm,
                }
                append_jsonl(metrics_path, payload)
                monitor.log(payload, step=step)
                logger.success(
                    f"Step {step} | reward={payload['candidate_reward_mean']:.4f} "
                    f"| tok/s={payload['tokens_per_second']:.0f} | gen={generation_s:.2f}s "
                    f"| write={adapter_write_s:.2f}s | load={adapter_load_s:.2f}s "
                    f"| unload={adapter_unload_s:.2f}s | slot={adapter_slot_update_s:.2f}s "
                    f"| update={update_s:.2f}s"
                )

                should_ckpt = config.ckpt is not None and (
                    config.ckpt.interval is None or step % config.ckpt.interval == 0 or step == config.max_steps
                )
                if should_ckpt:
                    save_es_state(config.output_dir, step, theta, template.specs, payload)
                    write_adapter_from_theta(final_adapter, template, theta)
                    maybe_clean_es_checkpoints(config.output_dir, step, config.ckpt.keep_last)

                if heart is not None:
                    heart.beat()

            progress_box = [progress]
            dist.broadcast_object_list(progress_box, src=0)
            progress = progress_box[0]
            dist.barrier()

    finally:
        if admin_clients:
            await close_admin_clients(admin_clients)
        shutdown_train_envs(envs)
        if world.is_master and candidate_root.exists() and not config.algorithm.keep_candidate_adapters:
            shutil.rmtree(candidate_root, ignore_errors=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()


def train(config: ESConfig) -> None:
    asyncio.run(train_async(config))


def main():
    set_proc_title("ES Trainer")
    train(cli(ESConfig))


if __name__ == "__main__":
    main()
