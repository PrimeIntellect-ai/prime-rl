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
    get_full_offload_dtype_policy,
    is_tt_moe_model,
    iter_moe_blocks,
    reshard_module,
    setup_model,
    setup_tokenizer,
    update_selection_biases,
)
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.models.layers.moe import sequence_balance
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.pretrain.completep import completep_param_groups
from prime_rl.trainer.pretrain.diagnostics import UpdateTracker, probe_batch, run_probe
from prime_rl.trainer.pretrain.metrics import (
    DeadExperts,
    SpikeDetector,
    expert_load_metrics,
    grad_norms_by_group,
    reduce_stats,
    take_routing_counts,
)
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
        param_groups=(
            completep_param_groups(model, config.model.init, config.optim)
            if config.model.init.scheme == "completep"
            else None
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
    is_moe_model = is_tt_moe_model(model)
    moes_with_seq_balance = [moe for moe in iter_moe_blocks(model) if moe.seq_aux_loss_coeff is not None]
    bias_freeze_step = config.max_steps - round(config.moe_bias_freeze * config.max_steps) if config.max_steps else None

    def compute_loss(micro_batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning (loss_sum, token_count) over unmasked tokens, the MoE sequence-wise balance
        loss summed over tokens (each token carries its row's loss, so after the division by the token count it
        is the mean over rows), and the per-row balance of each MoE layer that computes it."""
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
        # A micro-batch is `micro_batch_size` packed rows concatenated into one sample; the sequences are the rows.
        num_rows = config.data.micro_batch_size
        seq_balances = [sequence_balance(*moe.routing, num_rows) for moe in moes_with_seq_balance]
        aux_loss_sum = torch.zeros((), device="cuda")
        for moe, balance in zip(moes_with_seq_balance, seq_balances):
            aux_loss_sum += moe.seq_aux_loss_coeff * balance.sum() * (input_ids.shape[1] // num_rows)
        return out["loss"], loss_mask.sum(dtype=torch.int64), aux_loss_sum, seq_balances

    def run_evals(step: int) -> None:
        """Perplexity evals on the weights after ``step`` optimizer updates."""
        metrics = evaluate_perplexity(model, perplexity_sets, byte_lengths, config.data.micro_batch_size)
        # The no-grad eval forwards leave the [lm_head, norm] FSDP group unsharded, and their
        # routing must not count toward the next step's MoE statistics.
        reshard_module(model)
        if is_moe_model:
            take_routing_counts(model)
        logger.success(f"Evals | Step {step} | " + " | ".join(f"{k} {v:.4f}" for k, v in sorted(metrics.items())))
        asyncio.run(monitors.log({**metrics, "step": step}, step=step))

    if checkpoint_step is None and config.val is not None and config.val.eval_on_start:
        run_evals(step=0)

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3
    num_dp_cp_ranks = dist.get_world_size(dp_cp_group)
    dead_experts = DeadExperts()
    loss_spikes, grad_norm_spikes = SpikeDetector(), SpikeDetector()
    last_bias_moves = None
    probe = None
    update_tracker = UpdateTracker()
    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    while True:
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step >= config.max_steps
        step_start_time = time.perf_counter()

        if gradient_manager is not None:
            micro_batches = [next(dataiter) for _ in range(grad_accum_steps)]
            local_token_count = sum(int(micro_batch["loss_mask"].sum()) for micro_batch in micro_batches)
            global_step_token_count = torch.tensor(local_token_count, dtype=torch.int64, device="cuda")
            dist.all_reduce(global_step_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
            global_token_count_val = global_step_token_count.item() // cp_size
            grad_scale = parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count_val
            prepare_gradient_offload(gradient_manager, grad_scale, overlap_optimizer=True)

        # Per-rank statistics, kept on the GPU and reduced once after the micro-batches (`reduce_stats`).
        zero = torch.zeros((), device="cuda", dtype=torch.float64)
        sums = {"loss": zero.clone(), "loss_tokens": zero.clone(), "nan_count": zero.clone(), "tokens": zero.clone()}
        maxes = {}
        data_time = 0.0
        for micro_step in range(grad_accum_steps):
            data_start_time = time.perf_counter()
            micro_batch = next(dataiter) if gradient_manager is None else micro_batches[micro_step]
            data_time += time.perf_counter() - data_start_time
            if config.diagnostics is not None and probe is None:
                probe = probe_batch(micro_batch, config.diagnostics.probe_tokens)
            sums["tokens"] += micro_batch["input_ids"].shape[1]
            local_loss_sum, batch_token_count, aux_loss_sum, seq_balances = compute_loss(micro_batch)
            sums["loss_tokens"] += batch_token_count
            # A NaN micro-batch is left out of the loss and the backward, without a host sync.
            is_nan = torch.isnan(local_loss_sum.detach())
            sums["nan_count"] += is_nan
            sums["loss"] += torch.where(is_nan, 0.0, local_loss_sum.detach())
            scaled_loss = torch.where(is_nan, torch.nan_to_num(local_loss_sum, nan=0.0), local_loss_sum + aux_loss_sum)
            scaled_loss = scaled_loss / grad_accum_steps
            begin_backward(gradient_manager, final_backward=micro_step == grad_accum_steps - 1)
            scaled_loss.backward()
            finish_backward(gradient_manager)

            if is_moe_model:
                counts, confidence = take_routing_counts(model)
                sums["expert_counts"] = sums.get("expert_counts", zero) + counts
                sums["routing_confidence"] = sums.get("routing_confidence", zero) + confidence
                # One rank's micro-batch: the imbalance that expert parallelism and the dropless dispatch see.
                micro_max_mean = counts.amax(dim=1) / counts.mean(dim=1)
                sums["micro_max_mean"] = sums.get("micro_max_mean", zero) + micro_max_mean.mean()
                maxes["micro_max_mean_max"] = torch.maximum(
                    maxes.get("micro_max_mean_max", zero), micro_max_mean.amax()
                )
                if seq_balances:
                    rows = torch.cat(seq_balances).detach()
                    sums["seq_balance"] = sums.get("seq_balance", zero) + rows.sum()
                    sums["seq_balance_rows"] = sums.get("seq_balance_rows", zero) + rows.numel()

        sums["data_time"] = zero + data_time
        maxes["data_time_max"] = zero + data_time
        # Peak memory since the previous step's reduction: this step's forward/backward and the previous optimizer step.
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3
        torch.cuda.reset_peak_memory_stats()
        sums["peak_memory"] = zero + peak_memory
        maxes["peak_memory_max"] = zero + peak_memory
        stats = reduce_stats(sums, maxes, dp_cp_group)
        forward_backward_time = time.perf_counter() - step_start_time

        if gradient_manager is None:
            global_token_count_val = int(stats["loss_tokens"])
            grad_scale = parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count_val
            scale_gradients_(None, model, grad_scale)
        batch_loss = stats["loss"].item() / global_token_count_val
        if stats["nan_count"] > 0:
            logger.warning(f"{int(stats['nan_count'])} micro-batches had a NaN loss and were left out of this step")

        moe_metrics = {}
        if is_moe_model:
            counts = stats["expert_counts"]
            routed_tokens = counts.sum(dim=1) / next(iter_moe_blocks(model)).router.top_k
            moe_metrics = {
                **expert_load_metrics(counts),
                "expert_load/dead": dead_experts.update(counts),
                "expert_load/micro_max_mean/mean": stats["micro_max_mean"].item()
                / (grad_accum_steps * num_dp_cp_ranks),
                "expert_load/micro_max_mean/max": stats["micro_max_mean_max"].item(),
                "router/confidence": (stats["routing_confidence"] / routed_tokens).mean().item(),
            }
            if "seq_balance" in stats:
                moe_metrics["loss/seq_balance"] = (stats["seq_balance"] / stats["seq_balance_rows"]).item()
            biases = [moe.router.selection_bias for moe in iter_moe_blocks(model) if moe.load_balance_coeff is not None]
            if bias_freeze_step is None or progress.step <= bias_freeze_step:
                previous_biases = [bias.clone() for bias in biases]
                update_selection_biases(model, counts.to("cuda", torch.float32))
                if biases:
                    # Share of experts whose bias moved in the opposite direction to the previous step's move.
                    moves = torch.stack([bias - previous for bias, previous in zip(biases, previous_biases)]).sign()
                    if last_bias_moves is not None:
                        moe_metrics["router/bias/flip_frac"] = (moves * last_bias_moves < 0).float().mean().item()
                    last_bias_moves = moves
            if biases:
                stacked = torch.stack(biases)
                moe_metrics["router/bias/absmax"] = stacked.abs().max().item()
                moe_metrics["router/bias/std"] = stacked.std(dim=1).mean().item()

        group_grad_norms = grad_norms_by_group(model)
        grad_norm = clip_grad_norm_(gradient_manager, model, config.optim.max_norm) if config.optim.max_norm else None
        is_diagnostic_step = config.diagnostics is not None and progress.step % config.diagnostics.interval == 0
        if is_diagnostic_step:
            update_tracker.before_step(model)
        optimizer.step()
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # The last step's checkpoint is written once after the loop.
        save_ckpt_time = 0.0
        disk_metrics = {}
        if config.ckpt and config.ckpt.interval and progress.step % config.ckpt.interval == 0 and not is_last_step:
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
            ckpt_manager.maybe_clean()
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time
            disk_metrics = get_ckpt_disk_metrics(config.run_dir)

        diagnostic_metrics = {}
        if is_diagnostic_step:
            diagnostic_metrics = update_tracker.after_step(model) | run_probe(model, probe, dp_cp_group)
            reshard_module(model)
            if is_moe_model:
                take_routing_counts(model)

        eval_start_time = time.perf_counter()
        if config.val is not None and (is_last_step or progress.step % config.val.interval == 0):
            run_evals(progress.step)
        eval_time = time.perf_counter() - eval_start_time

        # Every CP rank counts its rows in full.
        num_tokens = int(stats["tokens"]) // cp_size
        progress.total_tokens += num_tokens
        dataset_progress = get_dataset_progress(dataloader)
        progress.total_samples = dataset_progress["step"]
        step_time = time.perf_counter() - step_start_time - eval_time
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_step_tokens_per_second(num_tokens, step_time)
        mfu = perf_counter.get_step_mfu(num_tokens, step_time)
        grad_norm = grad_norm.item() if grad_norm is not None else None

        step_message = f"Step {progress.step} | {step_time:.2f}s | Loss {batch_loss:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm {grad_norm:.4f}"
        step_message += (
            f" | LR {current_lr:.2e} | Throughput {throughput:.0f} tokens/s | MFU {mfu:.1f}%"
            f" | Peak Mem. {stats['peak_memory_max'].item():.1f}/{max_memory:.1f} GiB"
        )
        logger.success(step_message)

        metrics = {
            "loss/mean": batch_loss,
            "loss/perplexity": math.exp(min(batch_loss, 20)),
            "loss/nan_count": int(stats["nan_count"]),
            "loss/spike": loss_spikes.update(batch_loss),
            "optim/lr": current_lr,
            **{f"optim/grad_norm/{group}": norm.item() for group, norm in group_grad_norms.items()},
            "progress/epoch": dataset_progress["epoch"],
            "progress/num_samples": progress.total_samples,
            "progress/num_tokens": progress.total_tokens,
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/peak_memory/max": stats["peak_memory_max"].item(),
            "perf/peak_memory/mean": stats["peak_memory"].item() / num_dp_cp_ranks,
            "perf/mfu": mfu,
            "time/step": step_time,
            "time/forward_backward": forward_backward_time,
            "time/data/max": stats["data_time_max"].item(),
            "time/data/mean": stats["data_time"].item() / num_dp_cp_ranks,
            "time/save_ckpt": save_ckpt_time,
            "time/evals": eval_time,
            **disk_metrics,
            **moe_metrics,
            **diagnostic_metrics,
            "step": progress.step,
        }
        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm
            metrics["optim/grad_norm_spike"] = grad_norm_spikes.update(grad_norm)
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
