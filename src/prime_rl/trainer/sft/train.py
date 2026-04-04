import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

import time
from contextlib import nullcontext
from datetime import timedelta

from ring_flash_attn import substitute_hf_flash_attn
from torch.nn import CrossEntropyLoss

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.trainer.models.layers.attn import substitute_ring_attn
from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.configs.sft import SFTConfig
from prime_rl.utils.cp import setup_cp_params, shard_for_cp
from prime_rl.trainer.runs import Progress, get_multi_run_manager, setup_multi_run_manager
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.utils.logger import setup_logger
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.model import (
    forward,
    get_load_balance_stats,
    is_tt_moe_model,
    setup_tokenizer,
    setup_model,
)
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.sft.data import load_sft_dataset, setup_dataloader, setup_dataset
from prime_rl.trainer.utils import (
    GarbageCollection,
    MemoryProfiler,
    export_benchmark_json,
    get_zero_gradient_ratio,
    get_ckpt_disk_metrics,
    print_sample,
    setup_torch_distributed,
    print_benchmark,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit, to_col_format
import torch.distributed as dist
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from prime_rl.trainer.models.layers.lm_head import FUSED_CE_IGNORE_INDEX

from torchtitan.distributed.utils import clip_grad_norm_


@clean_exit
def train(config: SFTConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting SFT trainer in {world}")

    # Print warning if running in benchmark mode
    if config.bench is not None:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Setup heartbeat (only on rank 0)
    heart = None
    if config.heartbeat is not None and world.rank == 0:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat.url)

    # Set precision
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds), enable_gloo=config.model.fsdp_cpu_offload
    )
    torch.set_float32_matmul_precision("high")

    if config.model.lora is not None:
        setup_multi_run_manager(config.output_dir, 1, torch.device("cuda", world.local_rank), config.model.lora)

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)

    total_micro_batches = config.data.batch_size * config.model.cp
    micro_batches_per_step = world.world_size * config.data.micro_batch_size
    assert total_micro_batches % micro_batches_per_step == 0, (
        f"batch_size * cp ({total_micro_batches}) must be divisible by "
        f"world_size * micro_batch_size ({micro_batches_per_step})"
    )
    grad_accum_steps = total_micro_batches // micro_batches_per_step

    if parallel_dims.cp_enabled:
        assert config.data.seq_len % parallel_dims.cp == 0, "Sequence length must be divisible by CP degree"
        cp_group = parallel_dims.world_mesh["cp"].get_group()
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank()
        substitute_hf_flash_attn(cp_group, heads_k_stride=1)
        substitute_ring_attn(cp_group, heads_k_stride=1, attn_impl=config.model.attn)
        from prime_rl.utils.cp import setup_hybrid_cp, setup_sparse_mla_cp

    # Set up checkpoint manager
    logger.info(f"Initializing checkpoint managers ({config.ckpt})")
    ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(config.output_dir, config.ckpt, config.model.lora)

    checkpoint_step = None
    if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
        if config.ckpt.resume_step == -1:
            checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            checkpoint_step = config.ckpt.resume_step

    # Initialize the model and tokenizer
    logger.info(f"Initializing model ({config.model})")
    loading_from_ckpt_later = config.ckpt and checkpoint_step is not None
    fused_cross_entropy: bool | str = {"liger_fused": "liger", "quack_fused": "quack"}.get(config.loss_impl, False)
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later, fused_cross_entropy=fused_cross_entropy)

    if parallel_dims.cp_enabled:
        setup_hybrid_cp(model, cp_group, cp_rank, parallel_dims.cp)
        setup_sparse_mla_cp(model, cp_group, cp_rank, parallel_dims.cp)

    if config.model.lora is not None:
        multi_run_manager = get_multi_run_manager()
        multi_run_manager.reset_run_parameters(0)
        multi_run_manager.scaling_factors[0] = config.model.lora.alpha / config.model.lora.rank

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(
        config.optim, list(model.named_parameters()), parallel_dims, cpu_offload=config.model.optim_cpu_offload
    )

    # Set up the learning rate scheduler
    scheduler_steps = (
        config.max_steps - config.ckpt.resume_step
        if config.max_steps is not None
        and (config.ckpt and config.ckpt.skip_scheduler and config.ckpt.resume_step is not None)
        else config.max_steps
    )
    logger.info(f"Setting up {config.scheduler.type} scheduler with {scheduler_steps} steps ({config.scheduler})")
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    # Set up the dataset and dataloader
    logger.info(f"Initializing data ({config.data})")
    dataset = setup_dataset(tokenizer, config.data, config.model.cp)
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    val_raw_dataset = None
    if config.val is not None:
        logger.info(f"Loading validation dataset ({config.val.data.name})")
        val_raw_dataset = load_sft_dataset(config.val.data)

    # Optionally, resume training from a checkpoint
    progress = Progress()

    if checkpoint_step is not None:
        ckpt_manager.load(
            checkpoint_step,
            model,
            [optimizer],
            scheduler if not config.ckpt.skip_scheduler else None,
            progress if not config.ckpt.skip_progress else None,
            dataloader=dataloader if not config.ckpt.skip_dataloader else None,
        )
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
        # This redundant setup is necessary because loading the optimizer's state has side effects on the scheduler state dict
        if config.ckpt.skip_scheduler:
            scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)
    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples}, dataset_state={dataloader.state_dict()['dataset_state']})"
    )

    cp_enabled = parallel_dims.cp_enabled
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
    cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    cp_size = parallel_dims.cp

    ce_loss = None
    match config.loss_impl:
        case "liger":
            ce_loss = LigerCrossEntropyLoss(reduction="none")
        case "torch":
            ce_loss = CrossEntropyLoss(reduction="none")
        case "liger_fused" | "quack_fused":
            pass  # loss is computed inside the fused lm_head
        case _:
            raise ValueError(f"Invalid loss implementation: {config.loss_impl}")

    def compute_loss(micro_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (loss_sum, token_count) over unmasked tokens."""
        input_ids = micro_batch["input_ids"].to("cuda")
        position_ids = micro_batch["position_ids"].to("cuda")
        target_ids = micro_batch["target_ids"].to("cuda")
        loss_mask = micro_batch["loss_mask"].to("cuda")

        if cp_enabled:
            input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size, cp_group)
            target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=cp_size)
            loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)

        if config.model.lora is not None:
            set_lora_num_tokens(torch.full((1,), input_ids.numel(), dtype=torch.int32, device="cuda"))

        token_count = loss_mask.sum(dtype=torch.int64)

        with maybe_activation_offloading(config.model.ac_offloading):
            if config.loss_impl in ("liger_fused", "quack_fused"):
                masked_target_ids = target_ids.clone()
                masked_target_ids[~loss_mask] = FUSED_CE_IGNORE_INDEX
                out = forward(model, input_ids, position_ids, labels=masked_target_ids)
                loss_sum = out["loss"] * token_count
            else:
                out = forward(model, input_ids, position_ids)
                logits = out["logits"]
                B, L, V = logits.shape
                token_loss = ce_loss(logits.view(-1, V), target_ids.view(-1)).view(B, L)
                loss_sum = token_loss[loss_mask].sum()
                del logits

        del out
        return loss_sum, token_count

    maybe_record_function = nullcontext

    def run_eval_loop(data_iter):
        """Validation forward loop. Returns token-weighted global mean loss."""
        total_loss_sum = torch.tensor(0.0, device="cuda")
        total_token_count = torch.tensor(0, dtype=torch.int64, device="cuda")
        nan_count = torch.tensor(0, device="cuda")

        with torch.no_grad():
            for micro_batch in data_iter:
                loss_sum, token_count = compute_loss(micro_batch)
                if not torch.isnan(loss_sum.detach()):
                    total_loss_sum += loss_sum.detach()
                    total_token_count += token_count
                else:
                    nan_count += 1

        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM, group=dp_cp_group)
        dist.all_reduce(total_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
        dist.all_reduce(nan_count, op=dist.ReduceOp.SUM)

        mean_loss = (total_loss_sum / total_token_count).item() if total_token_count.item() > 0 else float("nan")
        return mean_loss, nan_count.item()

    def run_validation(step: int) -> None:
        val_dataset = setup_dataset(
            tokenizer, config.val.data, config.model.cp, max_epochs=1, raw_dataset=val_raw_dataset
        )
        val_dataloader = setup_dataloader(val_dataset, config.val.data)

        # No train/eval switch: no dropout in these models, and toggling would trigger torch.compile recompilation
        mean_loss, nan_count = run_eval_loop(val_dataloader)
        if nan_count > 0:
            logger.warning(f"Validation at step {step}: {nan_count} batches had NaN loss")
        if mean_loss != mean_loss:
            logger.warning(f"Validation at step {step} had no valid tokens")
        else:
            logger.success(f"Validation | Step {step} | Loss: {mean_loss:.4f}")
        monitor.log({"val/loss": mean_loss, "step": step}, step=step)

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3  # GiB
    is_first_step = True
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function  # noqa: F841 – captured by run_forward_loop closure
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step == config.max_steps

        if (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            save_ckpt_time = 0

            if not config.ckpt.weights_only:
                # Save full checkpoint
                logger.info(f"Saving checkpoint at step {progress.step}")
                save_ckpt_start_time = time.perf_counter()
                ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
                save_ckpt_time += time.perf_counter() - save_ckpt_start_time

            ckpt_manager.maybe_clean()

            # Save weight checkpoint
            if weight_ckpt_manager is not None:
                logger.info(f"Saving weight checkpoint at step {progress.step}")
                save_ckpt_start_time = time.perf_counter()
                weight_ckpt_manager.save(progress.step, model, tokenizer)
                save_ckpt_time += time.perf_counter() - save_ckpt_start_time
                weight_ckpt_manager.maybe_clean()
        else:
            save_ckpt_time = 0

        # Break if we have reached the maximum number of steps
        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        memory_profiler = (
            MemoryProfiler(progress.step, config.memory_profiler_path) if config.memory_profiler_path else None
        )

        step_start_time = time.perf_counter()
        forward_backward_start_time = time.perf_counter()

        step_loss_sum = torch.tensor(0.0, device="cuda")
        step_local_token_count = torch.tensor(0, dtype=torch.int64, device="cuda")
        nan_loss_count = torch.tensor(0, device="cuda")
        batch_max_vio = torch.tensor(0.0, device="cuda")
        for micro_step in range(grad_accum_steps):
            micro_batch = next(dataiter)

            if config.log.log_data:
                print_sample(
                    micro_batch["input_ids"].flatten().tolist(), micro_batch["loss_mask"].flatten().tolist(), tokenizer
                )

            with maybe_record_function("forward"):
                local_loss_sum, local_token_count = compute_loss(micro_batch)

            step_local_token_count += local_token_count

            if torch.isnan(local_loss_sum.detach()):
                nan_loss_count += 1
                logger.warning("Local loss is nan, excluding this micro step from backward")
                scaled_loss = torch.nan_to_num(local_loss_sum, nan=0.0) / grad_accum_steps
            else:
                step_loss_sum += local_loss_sum.detach()
                scaled_loss = local_loss_sum / grad_accum_steps

            with maybe_record_function("backward"):
                scaled_loss.backward()

            if is_tt_moe_model(model):
                max_vio = get_load_balance_stats(model)["max_vio"]
                if max_vio is not None:
                    max_vio = max_vio.mean()
                    dist.all_reduce(max_vio, op=dist.ReduceOp.MAX)
                    batch_max_vio += max_vio / grad_accum_steps

        forward_backward_time = time.perf_counter() - forward_backward_start_time

        # All-reduce token counts and rescale gradients to get a global token-weighted mean.
        # FSDP already divided grads by fsdp_gradient_divide_factor, so we undo that and
        # divide by the true global token count instead.
        global_step_token_count = step_local_token_count.clone()
        dist.all_reduce(global_step_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
        global_token_count_val = global_step_token_count.item()

        if global_token_count_val > 0:
            grad_scale = parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count_val
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(grad_scale)

        # Run validation after forward-backward (so torch.compile sees training graph first) but before
        # optimizer step (so eval_on_start evaluates untrained weights)
        if config.val is not None and (
            (is_first_step and config.val.eval_on_start)
            or (not is_first_step and progress.step % config.val.interval == 0)
        ):
            run_validation(progress.step)

        # Compute the global mean loss for logging.
        dist.all_reduce(step_loss_sum, op=dist.ReduceOp.SUM, group=dp_cp_group)
        dist.all_reduce(nan_loss_count, op=dist.ReduceOp.SUM)
        if global_token_count_val > 0:
            batch_loss = (step_loss_sum / global_token_count_val).item()
        else:
            batch_loss = 0.0
        nan_loss_count = nan_loss_count.item()

        logger.debug(f"Clipping gradients with max norm {config.optim.max_norm}")
        grad_norm = clip_grad_norm_(
            model.parameters(), max_norm=config.optim.max_norm, ep_enabled=parallel_dims.ep_enabled
        )
        if grad_norm.device.type == "cpu":
            grad_norm = grad_norm.to(torch.device("cuda"))
        zero_grad_ratio = get_zero_gradient_ratio(model.parameters(), parallel_dims.dp_replicate)

        logger.debug("Optimizer step")
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Compute step metrics
        # Divide by CP since those ranks process the same data
        num_tokens = config.data.batch_size * config.data.seq_len // config.model.cp
        progress.total_tokens += num_tokens
        progress.total_samples = dataset.step
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3  # GiB

        # Log step metrics
        step_time = time.perf_counter() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {batch_loss:.4f} | Grad. Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}% | Peak Mem.: {peak_memory:.1f}/{max_memory:.1f} GiB ({peak_memory / max_memory * 100:.1f}%)"
        if is_tt_moe_model(model) and batch_max_vio.item() > 0:
            step_message += f" | Max Vio: {batch_max_vio.item():.4f}"
        logger.success(step_message)

        # Log progress metrics
        total_samples = sum(dataset.num_samples.values())
        total_tokens = sum(dataset.num_tokens.values())
        progress_metrics = {
            "progress/epoch": dataset.epoch,
            "progress/num_samples": progress.total_samples,
            "progress/num_tokens": progress.total_tokens,
            "step": progress.step,
        }
        # At least two subsets/splits
        if len(dataset.num_samples) > 1:
            progress_metrics.update(
                **{
                    f"progress/{subset_or_split}/ratio_samples": num_samples / total_samples
                    for subset_or_split, num_samples in dataset.num_samples.items()
                },
                **{
                    f"progress/{subset_or_split}/ratio_tokens": num_tokens / total_tokens
                    for subset_or_split, num_tokens in dataset.num_tokens.items()
                },
            )
        monitor.log(progress_metrics, step=progress.step)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/peak_memory": peak_memory,
            "perf/mfu": mfu,
            "step": progress.step,
        }
        monitor.log(perf_metrics, step=progress.step)

        # Log optimizer metrics
        optim_metrics = {
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm.item(),
            "optim/zero_grad_ratio": zero_grad_ratio,
            "step": progress.step,
        }
        monitor.log(optim_metrics, step=progress.step)

        loss_log_metrics = {
            "loss/mean": batch_loss,
            "loss/nan_count": nan_loss_count,
            "step": progress.step,
        }
        # Log tensor stats
        monitor.log(loss_log_metrics, step=progress.step)

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/save_ckpt": save_ckpt_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        monitor.log(time_metrics, step=progress.step)

        # Log disk metrics
        disk_metrics = get_ckpt_disk_metrics(config.output_dir)
        disk_metrics["step"] = progress.step
        monitor.log(disk_metrics, step=progress.step)

        if is_tt_moe_model(model) and batch_max_vio.item() > 0:
            monitor.log({"max_vio/mean": batch_max_vio.item(), "step": progress.step}, step=progress.step)

        is_first_step = False
        progress.step += 1

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)
        logger.info(f"Saved trace to {trace_file}")

    # Write final checkpoint
    if ckpt_manager is not None:
        if not (config.ckpt and config.ckpt.weights_only):
            logger.info("Writing final checkpoint")
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
        ckpt_manager.maybe_clean()

    # Write final weight checkpoint
    if weight_ckpt_manager is not None:
        logger.info("Writing final weight checkpoint")
        weight_ckpt_manager.save(progress.step, model, tokenizer)
        weight_ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("SFT trainer finished!")

    # Optionally, print benchmark table and export JSON
    if config.bench is not None and world.is_master:
        history = to_col_format(monitor.history)
        print_benchmark(history)
        if config.bench.output_json:
            export_benchmark_json(history, config.bench.output_json)
            logger.info(f"Benchmark results written to {config.bench.output_json}")


def main():
    set_proc_title("SFTTrainer")
    train(cli(SFTConfig))


if __name__ == "__main__":
    main()
