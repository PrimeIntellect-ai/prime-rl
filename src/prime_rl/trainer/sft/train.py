import time
from contextlib import nullcontext
from datetime import timedelta

from ring_flash_attn import substitute_hf_flash_attn
from torch.nn import CrossEntropyLoss

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.trainer.models.layers.attn import substitute_prime_rl_flash_attn
from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from loguru import logger
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.cp import setup_cp_params, shard_for_cp
from prime_rl.trainer.runs import Progress
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
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.utils import (
    MemoryProfiler,
    export_benchmark_json,
    get_ckpt_disk_metrics,
    print_sample,
    setup_torch_distributed,
    print_benchmark,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, to_col_format
import torch.distributed as dist
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

from torchtitan.distributed.utils import clip_grad_norm_


@clean_exit
@logger.catch(reraise=True)
def train(config: SFTTrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log" if config.log.file else None,
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

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)

    total_micro_batches = config.data.batch_size * config.model.cp * config.model.tp
    micro_batches_per_step = world.world_size * config.data.micro_batch_size
    assert total_micro_batches % micro_batches_per_step == 0, (
        f"batch_size * cp * tp ({total_micro_batches}) must be divisible by "
        f"world_size * micro_batch_size ({micro_batches_per_step})"
    )
    grad_accum_steps = total_micro_batches // micro_batches_per_step

    if parallel_dims.cp_enabled:
        assert config.data.seq_len % parallel_dims.cp == 0, "Sequence length must be divisible by CP degree"
        substitute_hf_flash_attn(parallel_dims.world_mesh["cp"].get_group(), heads_k_stride=1)
        substitute_prime_rl_flash_attn(parallel_dims.world_mesh["cp"].get_group(), heads_k_stride=1)

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
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, list(model.named_parameters()), parallel_dims)

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
    dataset = setup_dataset(tokenizer, config.data, config.model.cp * config.model.tp)
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

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
    cp_size = parallel_dims.cp

    match config.loss_impl:
        case "liger":
            ce_loss = LigerCrossEntropyLoss(reduction="none")
        case "torch":
            ce_loss = CrossEntropyLoss(reduction="none")
        case _:
            raise ValueError(f"Invalid loss implementation: {config.loss_impl}")

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3  # GiB
    is_first_step = True
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        is_last_step = config.max_steps is not None and progress.step == config.max_steps

        if (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            # Save full checkpoint
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

            # Maybe clean up old checkpoints
            ckpt_manager.maybe_clean()

            # Save weight checkpoint
            if weight_ckpt_manager is not None:
                logger.info(f"Saving weight checkpoint at step {progress.step}")
                weight_ckpt_manager.save(progress.step, model, tokenizer)
                # Maybe clean up old weight checkpoint
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

        batch_loss = torch.tensor(0.0).to("cuda")
        nan_loss_count = torch.tensor(0).to("cuda")
        batch_max_vio, max_vio = torch.tensor(0.0).to("cuda"), None
        for micro_step in range(grad_accum_steps):
            micro_batch = next(dataiter)
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            target_ids = micro_batch["target_ids"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")

            if cp_enabled:
                input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size, cp_group)
                target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=cp_size)
                loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)

            assert input_ids.shape == position_ids.shape == target_ids.shape == loss_mask.shape, (
                f"input_ids.shape: {input_ids.shape}, position_ids.shape: {position_ids.shape}, target_ids.shape: {target_ids.shape}, loss_mask.shape: {loss_mask.shape}"
            )

            if config.log.log_data:
                logger.debug("Printing samples of the first micro batch")
                print_sample(input_ids.flatten().tolist(), loss_mask.flatten().tolist(), tokenizer)

                # Forward pass
            logger.debug("Starting forward pass")
            with maybe_record_function("forward"), maybe_activation_offloading(config.model.ac_offloading):
                out = forward(model, input_ids, position_ids)

            logits = out["logits"]
            B, L, V = logits.shape

            # Compute loss
            loss = ce_loss(logits.view(-1, V), target_ids.view(-1)).view(B, L)

            # Compute average loss over unmasked tokens
            loss = loss[loss_mask].mean()

            # Accumulate average loss over gradient accumulation steps

            current_loss = loss.detach() / grad_accum_steps

            # only add if the loss is not nan
            if not torch.isnan(current_loss):
                batch_loss += current_loss
            else:
                nan_loss_count += 1
                logger.warning("Loss is nan, not taking into account in the batch loss calculation")

            # Delete logits before backward pass to avoid memory spike
            del logits

            # Backward pass
            logger.debug("Starting backward pass")
            with maybe_record_function("backward"):
                (loss / grad_accum_steps).backward()

            if is_tt_moe_model(model):
                max_vio = get_load_balance_stats(model)["max_vio"]
                if max_vio is not None:
                    max_vio = max_vio.mean()
                    dist.all_reduce(max_vio, op=dist.ReduceOp.MAX)
                    batch_max_vio += max_vio / grad_accum_steps

            # Debug log with *local, micro step* stats
            micro_step_message = f"Micro Step {micro_step}/{grad_accum_steps} | Loss: {loss.item():.4f} | Dataloader Step: {dataloader.state_dict()['dataset_state']['dataset']['step']}"
            if is_tt_moe_model(model) and max_vio is not None:
                micro_step_message += f" | Max Vio: {max_vio.item():.4f}"
            logger.debug(micro_step_message)

        logger.debug(f"Clipping gradients with max norm {config.optim.max_norm}")
        grad_norm = clip_grad_norm_(
            model.parameters(), max_norm=config.optim.max_norm, ep_enabled=parallel_dims.ep_enabled
        )
        if grad_norm.device.type == "cpu":
            grad_norm = grad_norm.to(torch.device("cuda"))

        logger.debug("Optimizer step")
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        forward_backward_time = time.perf_counter() - forward_backward_start_time

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize the tensor metrics across all steps and ranks
        logger.debug("Synchronizing tensor metrics across all steps and ranks")
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(nan_loss_count, op=dist.ReduceOp.SUM)

        # Compute step metrics
        # Divide by CP and TP since those ranks process the same data
        num_tokens = config.data.batch_size * config.data.seq_len // (config.model.cp * config.model.tp)
        progress.total_tokens += num_tokens
        progress.total_samples = dataset.step
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3  # GiB

        # Log step metrics
        step_time = time.perf_counter() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {batch_loss.item():.4f} | Grad. Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}% | Peak Mem.: {peak_memory:.1f}/{max_memory:.1f} GiB ({peak_memory / max_memory * 100:.1f}%)"
        if is_tt_moe_model(model) and max_vio is not None:
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
            "step": progress.step,
        }
        monitor.log(optim_metrics, step=progress.step)

        loss_log_metrics = {
            "loss/mean": batch_loss.item(),
            "loss/nan_count": nan_loss_count.item(),
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

        if is_tt_moe_model(model):
            max_vio_log_metrics = {
                "max_vio/mean": batch_max_vio.item(),
                "step": progress.step,
            }
            monitor.log(max_vio_log_metrics, step=progress.step)

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
    train(parse_argv(SFTTrainerConfig))


if __name__ == "__main__":
    main()
