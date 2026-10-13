import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

import asyncio
import math
import time
from datetime import timedelta

# ruff: noqa: I001

from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig

from prime_rl import monitors
from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.model import (
    forward,
    get_expert_load_stats,
    get_full_offload_dtype_policy,
    get_global_moe_stats,
    get_load_balance_stats,
    is_tt_moe_model,
    reshard_module,
    setup_model,
    setup_tokenizer,
)
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.pretrain.data import load_text_dataset, setup_text_dataloader, split_holdout
from prime_rl.trainer.pretrain.evals.common import token_byte_lengths
from prime_rl.trainer.pretrain.evals.perplexity import build_perplexity_set, evaluate_perplexity, load_perplexity_set
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.sft.data import get_dataset_progress, get_dataset_state, setup_dataloader, setup_dataset
from prime_rl.trainer.utils import (
    GarbageCollection,
    begin_backward,
    clip_grad_norm_,
    finish_backward,
    get_ckpt_disk_metrics,
    prepare_gradient_offload,
    scale_gradients_,
    setup_full_cpu_optimizer_offload,
    setup_torch_distributed,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


def check_arch(config: PretrainConfig) -> None:
    """``[model.arch]`` is applied by the launcher, which writes the model dir that ``model.name``
    points at. Fail if the trainer was started on a config that doesn't match."""
    hf_config = AutoConfig.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)
    mismatched = {key: value for key, value in config.model.arch.items() if getattr(hf_config, key, None) != value}
    if mismatched:
        raise ValueError(
            f"{config.model.name} does not match [model.arch] {mismatched}. Launch through `uv run pretrain`, "
            "which writes the model dir with the overrides applied."
        )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total, embedding (input embedding and LM head) and non-embedding parameter counts."""
    total = embedding = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "embed_tokens" in name or "lm_head" in name:
            embedding += param.numel()
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}


@clean_exit
def train(config: PretrainConfig):
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Starting pretrain trainer in {world} (output_dir={config.run_dir})")

    asyncio.run(
        monitors.setup(
            producer="trainer",
            wandb=config.monitors.wandb,
            prime=None,
            file=config.monitors.file,
            output_dir=config.run_dir,
            run_config=config,
            overview_flavor="sft",
        )
    )

    heart = Heartbeat(config.heartbeat) if config.heartbeat is not None and world.rank == 0 else None

    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds), enable_gloo=config.model.offload == "full"
    )
    if config.model.offload == "full":
        setup_full_cpu_optimizer_offload()
    torch.set_float32_matmul_precision(config.matmul_precision)

    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model)

    total_micro_batches = config.data.batch_size * config.model.cp
    micro_batches_per_step = world.world_size * config.data.micro_batch_size
    assert total_micro_batches % micro_batches_per_step == 0, (
        f"batch_size * cp ({total_micro_batches}) must be divisible by "
        f"world_size * micro_batch_size ({micro_batches_per_step})"
    )
    grad_accum_steps = total_micro_batches // micro_batches_per_step

    if config.model.arch:
        check_arch(config)

    ckpt_manager = setup_ckpt_manager(config.run_dir, config.ckpt, resume=config.resume)
    checkpoint_step = None
    if config.resume is not None:
        if config.resume.dir is not None:
            checkpoint_step = config.resume.dir_step
        else:
            checkpoint_step = config.resume.step or resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)

    logger.info(f"Initializing model ({config.model})")
    model = setup_model(config.model, parallel_dims, loading_from_checkpoint_later=checkpoint_step is not None)
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config.model, parallel_dims)
    num_params = count_parameters(model)
    logger.info(
        f"Parameters: {num_params['non_embedding'] / 1e6:,.1f}M non-embedding, "
        f"{num_params['embedding'] / 1e6:,.1f}M embedding, {num_params['total'] / 1e6:,.1f}M total"
    )

    tokenizer = setup_tokenizer(config.tokenizer)

    optimizer, gradient_manager = setup_optimizer(
        config.optim,
        list(model.named_parameters()),
        parallel_dims,
        offload=config.model.offload,
        model=model,
        full_offload_dtype_policy=(
            get_full_offload_dtype_policy(model, config.model) if config.model.offload == "full" else None
        ),
    )

    scheduler_steps = (
        config.max_steps - checkpoint_step
        if config.max_steps is not None and (config.ckpt and config.ckpt.skip_scheduler and checkpoint_step is not None)
        else config.max_steps
    )
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    holdout = None
    if config.data.type == "text":
        documents, holdout = split_holdout(load_text_dataset(config.data), config.data.holdout_docs)
        dataloader = setup_text_dataloader(documents, tokenizer, config.data, config.model.cp)
    else:
        dataloader = setup_dataloader(
            setup_dataset(tokenizer, config.data, config.model.cp), config.data, config.model.cp
        )

    eval_seq_len = config.data.seq_len
    perplexity_sets = []
    if config.val is not None:
        if holdout is not None:
            perplexity_sets.append(
                build_perplexity_set(
                    "holdout", holdout, tokenizer, eval_seq_len, max_tokens=10**12, text_column=config.data.text_column
                )
            )
        perplexity_sets += [load_perplexity_set(source, tokenizer, eval_seq_len) for source in config.val.source]
        for eval_set in perplexity_sets:
            logger.info(f"Perplexity eval {eval_set.name!r}: {len(eval_set.rows) * eval_seq_len:,} tokens")
    byte_lengths = token_byte_lengths(tokenizer) if perplexity_sets else None

    progress = Progress()
    if checkpoint_step is not None:
        skip = config.ckpt or CheckpointConfig()
        ckpt_manager.load(
            checkpoint_step,
            model,
            [optimizer],
            scheduler if not skip.skip_scheduler else None,
            progress,
            dataloader=dataloader if not skip.skip_dataloader else None,
            path=config.resume.dir / "trainer" if config.resume.dir is not None else None,
        )
        progress.step += 1
        if skip.skip_scheduler:
            scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)
        logger.info(f"Resuming from step {checkpoint_step} (dataset_state={get_dataset_state(dataloader)})")

    # Create the iterator only after a potential resume: iter() forks the workers with the
    # dataset's current state.
    dataiter = iter(dataloader)

    cp_enabled = parallel_dims.cp_enabled
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
    cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
    cp_size = parallel_dims.cp
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    ep_group = parallel_dims.get_mesh("ep").get_group() if parallel_dims.ep_enabled else None
    is_moe_model = is_tt_moe_model(model)

    def compute_loss(micro_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (loss_sum, token_count) over unmasked tokens."""
        input_ids = micro_batch["input_ids"].to("cuda", non_blocking=True)
        position_ids = micro_batch["position_ids"].to("cuda", non_blocking=True)
        target_ids = micro_batch["target_ids"].to("cuda", non_blocking=True)
        loss_mask = micro_batch["loss_mask"].to("cuda", non_blocking=True)
        seq_lens = micro_batch["seq_lens"].to("cuda", non_blocking=True)
        if cp_enabled:
            input_ids, position_ids = setup_cp_params(
                input_ids, position_ids, cp_rank, cp_size, cp_group, seq_lens=seq_lens, cp_style=config.model.cp_style
            )
            target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=cp_size)
            loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)
        labels = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
        with maybe_activation_offloading(config.model.ac_offloading):
            out = forward(
                model, input_ids, position_ids, seq_lens=seq_lens, labels=labels, seq_lens_are_pre_shard=cp_enabled
            )
        return out["loss"], loss_mask.sum(dtype=torch.int64)

    def run_evals(step: int) -> None:
        """Perplexity evals on the weights after ``step`` optimizer updates."""
        metrics = evaluate_perplexity(model, perplexity_sets, byte_lengths)
        # The no-grad eval forwards leave the [lm_head, norm] FSDP group unsharded, and their
        # routing must not count toward the next step's MoE statistics.
        reshard_module(model)
        if is_moe_model:
            get_load_balance_stats(model)
        logger.success(f"Evals | Step {step} | " + " | ".join(f"{k} {v:.4f}" for k, v in sorted(metrics.items())))
        asyncio.run(monitors.log({**metrics, "step": step}, step=step))

    if checkpoint_step is None and config.val is not None and config.val.eval_on_start:
        run_evals(step=0)

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3
    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    while True:
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step >= config.max_steps
        step_start_time = time.perf_counter()

        step_loss_sum = torch.tensor(0.0, device="cuda")
        nan_loss_count = torch.tensor(0, device="cuda")
        moe_stats = (
            {
                "max_vio/mean": torch.tensor(0.0),
                "max_vio/max": torch.tensor(0.0),
                "routing_confidence/mean": torch.tensor(0.0),
            }
            if is_moe_model
            else {}
        )
        if gradient_manager is None:
            micro_batches = (next(dataiter) for _ in range(grad_accum_steps))
            step_local_token_count = torch.tensor(0, dtype=torch.int64, device="cuda")
        else:
            micro_batches = [next(dataiter) for _ in range(grad_accum_steps)]
            local_token_count = sum(int(micro_batch["loss_mask"].sum()) for micro_batch in micro_batches)
            global_step_token_count = torch.tensor(local_token_count, dtype=torch.int64, device="cuda")
            dist.all_reduce(global_step_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
            global_token_count_val = global_step_token_count.item() // cp_size
            grad_scale = parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count_val
            prepare_gradient_offload(gradient_manager, grad_scale, overlap_optimizer=True)

        step_tokens_per_expert = 0
        step_local_num_tokens = 0
        for micro_step, micro_batch in enumerate(micro_batches):
            step_local_num_tokens += micro_batch["input_ids"].shape[1]
            local_loss_sum, batch_token_count = compute_loss(micro_batch)
            if gradient_manager is None:
                step_local_token_count += batch_token_count
            if torch.isnan(local_loss_sum.detach()):
                nan_loss_count += 1
                logger.warning("Local loss is nan, excluding this micro step from backward")
                scaled_loss = torch.nan_to_num(local_loss_sum, nan=0.0) / grad_accum_steps
            else:
                step_loss_sum += local_loss_sum.detach()
                scaled_loss = local_loss_sum / grad_accum_steps
            begin_backward(gradient_manager, final_backward=micro_step == grad_accum_steps - 1)
            scaled_loss.backward()
            finish_backward(gradient_manager)

            if is_moe_model:
                micro_moe_stats, tokens_per_expert = get_global_moe_stats(model, ep_group, dp_cp_group)
                for name, value in micro_moe_stats.items():
                    moe_stats[f"{name}/mean"] += value / grad_accum_steps
                    if name == "max_vio":
                        moe_stats["max_vio/max"] = torch.maximum(moe_stats["max_vio/max"], value)
                step_tokens_per_expert += tokens_per_expert

        forward_backward_time = time.perf_counter() - step_start_time
        expert_load_stats = get_expert_load_stats(step_tokens_per_expert, dp_cp_group) if is_moe_model else {}

        if gradient_manager is None:
            global_step_token_count = step_local_token_count.clone()
            dist.all_reduce(global_step_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
            global_token_count_val = global_step_token_count.item()
            grad_scale = parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count_val
            scale_gradients_(None, model, grad_scale)

        dist.all_reduce(step_loss_sum, op=dist.ReduceOp.SUM, group=dp_cp_group)
        dist.all_reduce(nan_loss_count, op=dist.ReduceOp.SUM)
        batch_loss = (step_loss_sum / global_token_count_val).item()
        nan_loss_count = nan_loss_count.item()

        grad_norm = clip_grad_norm_(gradient_manager, model, config.optim.max_norm) if config.optim.max_norm else None
        optimizer.step()
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # The last step's checkpoint is written once after the loop.
        save_ckpt_time = 0.0
        if config.ckpt and config.ckpt.interval and progress.step % config.ckpt.interval == 0 and not is_last_step:
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
            ckpt_manager.maybe_clean()
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

        eval_start_time = time.perf_counter()
        if config.val is not None and (is_last_step or progress.step % config.val.interval == 0):
            run_evals(progress.step)
        eval_time = time.perf_counter() - eval_start_time

        global_num_tokens = torch.tensor(step_local_num_tokens, dtype=torch.int64, device="cuda")
        dist.all_reduce(global_num_tokens, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp").get_group())
        num_tokens = global_num_tokens.item()
        progress.total_tokens += num_tokens
        dataset_progress = get_dataset_progress(dataloader)
        progress.total_samples = dataset_progress["step"]
        step_time = time.perf_counter() - step_start_time - eval_time
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_step_tokens_per_second(num_tokens, step_time)
        mfu = perf_counter.get_step_mfu(num_tokens, step_time)
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3

        step_message = f"Step {progress.step} | {step_time:.2f}s | Loss {batch_loss:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm {grad_norm:.4f}"
        step_message += (
            f" | LR {current_lr:.2e} | Throughput {throughput:.0f} tokens/s | MFU {mfu:.1f}%"
            f" | Peak Mem. {peak_memory:.1f}/{max_memory:.1f} GiB"
        )
        logger.success(step_message)

        metrics = {
            "loss/mean": batch_loss,
            "loss/perplexity": math.exp(min(batch_loss, 20)),
            "loss/nan_count": nan_loss_count,
            "optim/lr": current_lr,
            "progress/epoch": dataset_progress["epoch"],
            "progress/num_samples": progress.total_samples,
            "progress/num_tokens": progress.total_tokens,
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/peak_memory": peak_memory,
            "perf/mfu": mfu,
            "time/step": step_time,
            "time/forward_backward": forward_backward_time,
            "time/save_ckpt": save_ckpt_time,
            "time/evals": eval_time,
            **get_ckpt_disk_metrics(config.run_dir),
            **{name: value.item() for name, value in moe_stats.items()},
            **expert_load_stats,
            "step": progress.step,
        }
        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm.item()
        asyncio.run(monitors.log(metrics, step=progress.step))

        if heart is not None:
            heart.beat()
        if is_last_step:
            break
        progress.step += 1

    if config.ckpt is not None:
        logger.info(f"Saving final checkpoint at step {progress.step}")
        ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
        ckpt_manager.maybe_clean()
    if gradient_manager is not None:
        gradient_manager.close()
    logger.success("Pretrain trainer finished")
    asyncio.run(monitors.finalize())


def main():
    set_proc_title("PretrainTrainer")
    train(cli(PretrainConfig))


if __name__ == "__main__":
    main()
