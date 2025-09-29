import time
from contextlib import nullcontext
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

import torch
from torch.nn.functional import cross_entropy
from torch.distributed.tensor.experimental import context_parallel
from torch.profiler import profile, ProfilerActivity, record_function
from loguru import logger
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.weights import setup_weight_ckpt_manager
from prime_rl.trainer.sft.config import SFTTrainerConfig
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
    print_sample,
    setup_torch_distributed,
    print_benchmark,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, to_col_format
import torch.distributed as dist


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
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Set precision
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision("high")

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)

    # Initialize the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model})")
    model = setup_model(config.model, parallel_dims)
    tokenizer = setup_tokenizer(config.model)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, model, parallel_dims.world_mesh["dp_shard_cp"])

    # Set up the learning rate scheduler
    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    logger.info(f"Using `{config.scheduler.type}` scheduler ({config.scheduler})")

    # Set up weight checkpoint manager
    logger.info(f"Initializing weight checkpoint manager ({config.weights})")
    weight_ckpt_manager = setup_weight_ckpt_manager(config.output_dir, config.weights, config.ckpt, async_level=0)

    # Set up checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
    assert ckpt_manager is None or (ckpt_manager is not None and weight_ckpt_manager is not None), (
        "If ckpt_manager is set, weight_ckpt_manager must also be set"
    )

    # Set up the dataset and dataloader
    logger.info(f"Initializing data ({config.data})")
    dataset = setup_dataset(tokenizer, config.data, config.model.cp * config.model.tp)
    dataloader = setup_dataloader(dataset, tokenizer, config.data)
    dataiter = iter(dataloader)

    # Check that the world size and batch configuration is compatible
    num_micro_batches = config.data.batch_size // config.data.micro_batch_size
    if world.world_size > num_micro_batches:
        raise ValueError(
            f"There must be at least one micro batch per rank, but only have {num_micro_batches} micro batches for {world.world_size} ranks."
        )
    if num_micro_batches % world.world_size != 0:
        raise ValueError(
            f"The number of micro batches ({num_micro_batches}) must be divisible by the world size ({world.world_size})."
        )

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if ckpt_manager is not None and config.ckpt and config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step {config.ckpt.resume_step}")
        ckpt_manager.load(model, [optimizer], scheduler, progress, step=config.ckpt.resume_step, dataloader=dataloader)
    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples}, dataloader_state={dataloader.state_dict()['dataset_state']})"
    )

    logger.info(f"Starting training loop ({config.max_steps=})")
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

        # Save the weight checkpoint (if we are at an interval step and not at the first or last step)
        save_weights_time = 0
        if (
            weight_ckpt_manager is not None
            and (config.weights and config.weights.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.weights.interval == 0
        ):
            logger.info(f"Saving weight checkpoint at step {progress.step}")
            save_weights_start_time = time.time()
            weight_ckpt_manager.save(model, tokenizer, step=progress.step)
            save_weights_time = time.time() - save_weights_start_time

        # Save the full checkpoint (if we are at an interval step and not at the first or last step)
        save_ckpt_time = 0
        if (
            ckpt_manager is not None
            and weight_ckpt_manager is not None
            and config.ckpt
            and config.ckpt.interval
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step, dataloader=dataloader)
            save_ckpt_time = time.time() - save_ckpt_start_time

            # Maybe clean up old trainer checkpoints
            ckpt_manager.maybe_clean()

        # Break if we have reached the maximum number of steps
        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        memory_profiler = (
            MemoryProfiler(progress.step, config.memory_profiler_path) if config.memory_profiler_path else None
        )

        step_start_time = time.time()
        forward_backward_start_time = time.time()
        epoch = 0
        grad_accum_steps = (
            config.data.batch_size
            * config.model.cp
            * config.model.tp
            // (config.data.micro_batch_size * world.world_size)
        )

        batch_loss = torch.tensor(0.0).to("cuda")
        batch_max_vio, max_vio = torch.tensor(0.0).to("cuda"), None
        for micro_step in range(grad_accum_steps):
            micro_batch = next(dataiter)
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            target_ids = micro_batch["target_ids"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            epoch = micro_batch["epoch"]
            assert input_ids.shape == position_ids.shape == target_ids.shape == loss_mask.shape, (
                f"input_ids.shape: {input_ids.shape}, position_ids.shape: {position_ids.shape}, target_ids.shape: {target_ids.shape}, loss_mask.shape: {loss_mask.shape}"
            )

            if config.log.log_data:
                logger.debug("Printing samples of the first micro batch")
                print_sample(input_ids.flatten().tolist(), loss_mask.flatten().tolist(), tokenizer)

            if config.model.cp > 1:
                maybe_context_parallel = context_parallel(
                    parallel_dims.world_mesh["cp"],
                    buffers=tuple([input_ids, position_ids, target_ids, loss_mask]),
                    buffer_seq_dims=(1, 1, 1, 1),
                )
            else:
                maybe_context_parallel = nullcontext()

            with maybe_context_parallel:
                # Forward pass
                with maybe_record_function("forward"):
                    logits = forward(model, input_ids, position_ids)
                B, L, V = logits.shape

                # Compute loss
                loss = cross_entropy(logits.view(-1, V), target_ids.view(-1), reduction="none").view(B, L)

                # Compute average loss over unmasked tokens
                loss = loss[loss_mask].mean()

                # Accumulate average loss over gradient accumulation steps
                batch_loss += loss.detach() / grad_accum_steps

                # Delete logits before backward pass to avoid memory spike
                del logits

                # Backward pass
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm).full_tensor()

        logger.debug("Optimizer step")
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        forward_backward_time = time.time() - forward_backward_start_time

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize the tensor metrics across all steps and ranks
        logger.debug("Synchronizing tensor metrics across all steps and ranks")
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

        # Compute step metrics
        num_tokens = config.data.batch_size * config.data.seq_len
        progress.total_tokens += num_tokens
        progress.total_samples = dataloader.state_dict()["dataset_state"]["dataset"]["step"]
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GiB

        # Log step metrics
        step_time = time.time() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {batch_loss.item():.4f} | Grad. Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}% | Peak Mem.: {peak_memory:.1f}/{max_memory:.1f} GiB ({peak_memory / max_memory * 100:.1f}%)"
        if is_tt_moe_model(model) and max_vio is not None:
            step_message += f" | Max Vio: {batch_max_vio.item():.4f}"
        logger.success(step_message)

        # Log progress metrics
        progress_metrics = {
            "progress/epoch": epoch,
            "progress/total_samples": progress.total_samples,
            "progress/total_tokens": progress.total_tokens,
            "step": progress.step,
        }
        monitor.log(progress_metrics)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/peak_memory": peak_memory,
            "perf/mfu": mfu,
            "step": progress.step,
        }
        monitor.log(perf_metrics)

        # Log optimizer metrics
        optim_metrics = {
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm.item(),
            "step": progress.step,
        }
        monitor.log(optim_metrics)

        loss_log_metrics = {
            "loss/mean": batch_loss.item(),
            "step": progress.step,
        }
        # Log tensor stats
        monitor.log(loss_log_metrics)

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/save_ckpt": save_ckpt_time,
            "time/save_weights": save_weights_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        if is_tt_moe_model(model):
            max_vio_log_metrics = {
                "max_vio/mean": batch_max_vio.item(),
                "step": progress.step,
            }
            monitor.log(max_vio_log_metrics)

        is_first_step = False
        progress.step += 1

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)
        logger.info(f"Saved trace to {trace_file}")

    # Log final (immutable) distributions to W&B table
    monitor.log_final_distributions()

    # Write final weight checkpoint
    if weight_ckpt_manager is not None:
        logger.info("Writing final weight checkpoint")
        weight_ckpt_manager.save(model, tokenizer, step=progress.step)

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step, dataloader=dataloader)
        ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("SFT trainer finished!")

    # Optionally, print benchmark table
    if config.bench and world.is_master:
        print_benchmark(to_col_format(monitor.history))


def main():
    train(parse_argv(SFTTrainerConfig))


if __name__ == "__main__":
    main()
