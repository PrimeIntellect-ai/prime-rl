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

from prime_rl import monitors
from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.model import (
    forward,
    get_expert_load_stats,
    get_full_offload_dtype_policy,
    get_load_balance_stats,
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
from prime_rl.trainer.init import init_weights
from prime_rl.trainer.pretrain.completep import completep_param_groups
from prime_rl.trainer.pretrain.bytes import token_byte_lengths
from prime_rl.trainer.pretrain.data import (
    TokenShards,
    check_eos,
    setup_dataloader,
    setup_train_dataset,
    setup_val_dataset,
)
from prime_rl.trainer.pretrain.schedules import LRSchedule
from prime_rl.trainer.sft.data import get_dataset_progress, get_dataset_state
from prime_rl.trainer.step import run_micro_batches
from prime_rl.trainer.utils import (
    GarbageCollection,
    clip_grad_norm_,
    get_ckpt_disk_metrics,
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

    ckpt_manager = setup_ckpt_manager(config.run_dir, config.ckpt, resume=config.resume)
    checkpoint_step = None
    if config.resume is not None:
        if config.resume.dir is not None:
            checkpoint_step = config.resume.dir_step
        else:
            checkpoint_step = config.resume.step or resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)

    logger.info(f"Initializing model ({config.model})")
    # Weights come from the checkpoint when resuming and from the random init otherwise.
    model = setup_model(config.model, parallel_dims, loading_from_checkpoint_later=True)
    if checkpoint_step is None:
        init_weights(model, config.model.init)
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

    # With ckpt.skip_scheduler, the schedule restarts at the resumed step and spans the remaining steps.
    restart_schedule = config.ckpt is not None and config.ckpt.skip_scheduler and checkpoint_step is not None
    schedule_offset = checkpoint_step if restart_schedule else 0
    lr_schedule = LRSchedule(config.schedule, optimizer, config.max_steps - schedule_offset, schedule_offset)

    dataloader = setup_dataloader(
        setup_train_dataset(
            config.data, config.max_steps, len(tokenizer), tokenizer.eos_token_id, non_dp_size=config.model.cp
        ),
        config.data.micro_batch_size,
        config.data.num_workers,
    )

    val_sources = {}
    if config.val is not None:
        val_sources = {name: TokenShards(source) for name, source in config.val.sources.items()}
        check_eos(val_sources, tokenizer.eos_token_id)
        byte_lengths = token_byte_lengths(tokenizer).to("cuda")
        # Whole micro-batches on every data-parallel rank, as every forward is a collective.
        windows_per_round = parallel_dims.get_mesh("dp").size() * config.data.micro_batch_size
        val_windows = {}
        for name, source in val_sources.items():
            val_windows[name] = (source.num_tokens - 1) // config.data.seq_len // windows_per_round * windows_per_round
            if val_windows[name] == 0:
                raise ValueError(f"Val source {name!r} has fewer than {windows_per_round} windows of data.seq_len")
            logger.info(f"Val source {name!r}: {val_windows[name] * config.data.seq_len:,} tokens")

    progress = Progress()
    if checkpoint_step is not None:
        skip = config.ckpt or CheckpointConfig()
        ckpt_manager.load(
            checkpoint_step,
            model,
            [optimizer],
            None,
            progress,
            dataloader=dataloader if not skip.skip_dataloader else None,
            path=config.resume.dir / "trainer" if config.resume.dir is not None else None,
        )
        progress.step += 1
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
    bias_freeze_step = config.max_steps - round(config.moe_bias_freeze * config.max_steps)
    # Per-row balance of every MoE layer and micro-batch of the current step, for the metrics.
    step_seq_balances: list[torch.Tensor] = []

    def compute_loss(micro_batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (loss_sum, token_count) over unmasked tokens and, in training, the MoE
        sequence-wise balance loss summed over tokens (each token carries its row's loss, so after the division
        by the token count it is the mean over rows)."""
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
        aux_loss_sum = torch.zeros((), device="cuda")
        if torch.is_grad_enabled():
            # A micro-batch is `micro_batch_size` packed rows concatenated into one sample; the sequences are the rows.
            num_rows = config.data.micro_batch_size
            for moe in moes_with_seq_balance:
                balance = sequence_balance(*moe.routing, num_rows)
                aux_loss_sum += moe.seq_aux_loss_coeff * balance.sum() * (input_ids.shape[1] // num_rows)
                step_seq_balances.append(balance.detach().mean())
        return out["loss"], loss_mask.sum(dtype=torch.int64), aux_loss_sum

    @torch.no_grad()
    def run_evals(step: int) -> None:
        """Held-out loss, perplexity and bits per byte on the weights after ``step`` optimizer updates."""
        metrics = {}
        for name, source in val_sources.items():
            val_dataset = setup_val_dataset(source, config.data.seq_len, val_windows[name], config.model.cp)
            totals = torch.zeros(3, dtype=torch.float64, device="cuda")  # loss sum, tokens, bytes
            for micro_batch in setup_dataloader(val_dataset, config.data.micro_batch_size, num_workers=1):
                loss_sum, token_count, _ = compute_loss(micro_batch)
                # The bytes are counted on the unsharded targets, so every CP rank counts them once.
                num_bytes = byte_lengths[micro_batch["target_ids"].to("cuda")].sum() / cp_size
                totals += torch.stack([loss_sum.double(), token_count.double(), num_bytes.double()])
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=dp_cp_group)
            loss_sum, num_tokens, num_bytes = totals.tolist()
            loss = loss_sum / num_tokens
            metrics[f"val/{name}/loss"] = loss
            metrics[f"val/{name}/ppl"] = math.exp(min(loss, 20))
            metrics[f"val/{name}/bpb"] = loss_sum / (math.log(2) * num_bytes)
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
    logger.info(f"Starting training loop (max_steps={config.max_steps})")
    while True:
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = progress.step >= config.max_steps
        step_start_time = time.perf_counter()
        lr_multiplier = lr_schedule.apply(progress.step - 1)

        result = run_micro_batches(model, dataiter, compute_loss, grad_accum_steps, parallel_dims, gradient_manager)
        forward_backward_time = time.perf_counter() - step_start_time
        expert_load_stats = {}
        if is_moe_model:
            expert_load_stats = get_expert_load_stats(result.tokens_per_expert, dp_cp_group)
            if step_seq_balances:
                expert_load_stats["expert_load/seq_balance"] = torch.stack(step_seq_balances).mean().item()
                step_seq_balances.clear()
            if progress.step <= bias_freeze_step:
                update_selection_biases(model, result.tokens_per_expert)
            biases = [moe.router.selection_bias for moe in iter_moe_blocks(model) if moe.load_balance_coeff is not None]
            if biases:
                expert_load_stats["expert_load/selection_bias_absmax"] = torch.stack(biases).abs().max().item()

        grad_norm = clip_grad_norm_(gradient_manager, model, config.optim.max_norm) if config.optim.max_norm else None
        optimizer.step()
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]

        # The last step's checkpoint is written once after the loop.
        save_ckpt_time = 0.0
        if config.ckpt and config.ckpt.interval and progress.step % config.ckpt.interval == 0 and not is_last_step:
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], None, progress, dataloader=dataloader)
            ckpt_manager.maybe_clean()
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

        eval_start_time = time.perf_counter()
        if config.val is not None and (is_last_step or progress.step % config.val.interval == 0):
            run_evals(progress.step)
        eval_time = time.perf_counter() - eval_start_time

        num_tokens = result.num_tokens
        progress.total_tokens += num_tokens
        dataset_progress = get_dataset_progress(dataloader)
        progress.total_samples = dataset_progress["step"]
        step_time = time.perf_counter() - step_start_time - eval_time
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_step_tokens_per_second(num_tokens, step_time)
        mfu = perf_counter.get_step_mfu(num_tokens, step_time)
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3

        step_message = f"Step {progress.step} | {step_time:.2f}s | Loss {result.loss:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm {grad_norm:.4f}"
        step_message += (
            f" | LR {current_lr:.2e} | Throughput {throughput:.0f} tokens/s | MFU {mfu:.1f}%"
            f" | Peak Mem. {peak_memory:.1f}/{max_memory:.1f} GiB"
        )
        logger.success(step_message)

        metrics = {
            "loss/mean": result.loss,
            "loss/perplexity": math.exp(min(result.loss, 20)),
            "loss/nan_count": result.nan_count,
            "optim/lr": current_lr,
            **{f"progress/{name}/epochs": epochs for name, epochs in dataset_progress["epochs"].items()},
            **{f"progress/{name}/num_tokens": tokens for name, tokens in dataset_progress["tokens"].items()},
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
            **result.moe_stats,
            **expert_load_stats,
            "schedule/lr": lr_multiplier,
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
        ckpt_manager.save(progress.step, model, [optimizer], None, progress, dataloader=dataloader)
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
