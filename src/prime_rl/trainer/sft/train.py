import time

# Import environment before any other imports
# ruff: noqa: I001

import torch
from torch.nn.functional import cross_entropy, softmax
from loguru import logger
from prime_rl.trainer.ckpt import CheckpointManager, Progress
from prime_rl.trainer.weights import WeightCheckpointManager
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.trainer.logger import setup_logger
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.model import (
    forward,
    setup_tokenizer,
    setup_model,
)
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.utils import (
    Tensors,
    print_benchmark,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, to_col_format


@clean_exit
@logger.catch(reraise=True)
def train(config: SFTTrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(config.log, world)
    logger.info(f"Starting SFT trainer in {world}")

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps}, {config.data.fake=})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(config.monitor, outputs_dir=config.outputs_dir, run_config=config)

    # Set precision and cuda device
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(world.rank)

    # Initialize the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model})")
    model = setup_model(config.model)
    tokenizer = setup_tokenizer(config.model)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, model)

    # Set up the learning rate scheduler
    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps)
    logger.info(f"Using `{config.scheduler.type}` scheduler ({config.scheduler})")

    # Get checkpoint manager
    if config.ckpt:
        logger.info(f"Initializing checkpoint manager ({config.ckpt})")
        weight_ckpt_manager = WeightCheckpointManager(config.outputs_dir, config.weights, config.ckpt, async_level=0)
        ckpt_manager = CheckpointManager(config.outputs_dir, config.ckpt)

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if config.ckpt and config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step `{config.ckpt.resume_step}`")
        ckpt_manager.load(model, [optimizer], scheduler, progress, step=config.ckpt.resume_step)
    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples})"
    )

    # Set up the dataset and dataloader (optionaly, use a fake dataset for debugging)
    logger.info(f"Initializing dataset (name={config.data.name}, split={config.data.split})")
    dataset = setup_dataset(tokenizer, config.data)

    # Set up the dataloader over micro batches
    logger.info(
        f"Initializing dataloader (micro_batch_size={config.data.micro_batch_size}, batch_size={config.data.batch_size}, collate_mode={config.data.collate_mode})"
    )
    dataloader = iter(setup_dataloader(dataset, tokenizer, config.data))

    logger.info(f"Starting training loop ({config.max_steps=})")
    is_first_step = True
    while True:
        # Save the full checkpoint (if we are at an interval step and not at the first or last step)
        is_last_step = progress.step == config.max_steps - 1
        save_ckpt_time = 0
        if (
            config.ckpt
            and config.ckpt.interval
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step)
            weight_ckpt_manager.save(model, tokenizer, step=progress.step)
            save_ckpt_time = time.time() - save_ckpt_start_time

            # Maybe clean up old trainer checkpoints
            ckpt_manager.maybe_clean()

        # Break if we have reached the maximum number of steps
        if progress.step >= config.max_steps:
            break

        if config.profile_path and world.rank == 0:
            torch.cuda.memory._record_memory_history()

        step_start_time = time.time()
        forward_backward_start_time = time.time()
        tensors = Tensors()  # Used to accumulate tensor statistics across grad acc and ranks for logging
        grad_accum_steps = config.data.batch_size // (config.data.micro_batch_size * world.world_size)
        for micro_step in range(grad_accum_steps):
            micro_batch = next(dataloader)
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            target_ids = micro_batch["target_ids"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            epoch = micro_batch["epoch"]
            assert input_ids.shape[0] == position_ids.shape[0]

            # Forward pass
            logits = forward(model, input_ids, position_ids).contiguous()
            B, L, V = logits.shape

            # Compute loss
            loss = cross_entropy(logits.view(-1, V), target_ids.view(-1), reduction="none").view(B, L)

            # Compute accuracy
            probs = softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1)
            accuracy = torch.eq(pred_ids, target_ids).float()

            # Add tensors to tensor dict for logging purposes
            tensors["loss"].append(loss[loss_mask].detach().to("cpu"))
            tensors["accuracy"].append(accuracy[loss_mask].detach().to("cpu"))

            # Mean reduction of unmasked tokens
            loss = loss[loss_mask].mean()

            # Scale loss by number of gradient accumulation steps
            loss /= grad_accum_steps

            # Delete logits before backward pass to avoid memory spike
            del logits

            # Backward pass
            loss.backward()

            # Debug log with *local, micro step* stats
            logger.debug(
                f"Micro Step {micro_step} | Loss: {tensors['loss'][-1].mean().item():.4f} | Accuracy: {tensors['accuracy'][-1].mean().item():.4f}"
            )

        # Optionally, clip the gradients
        logger.debug(f"Clipping gradients to {config.optim.max_norm}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm).full_tensor()

        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        scheduler.step()

        forward_backward_time = time.time() - forward_backward_start_time

        # Optionally, dump memory snapshot
        if config.profile_path and progress.step == 2 and world.rank == 0:
            logger.debug("Dumping memory snapshot")
            profile_path = config.profile_path
            if not profile_path.suffix == ".pickle":
                profile_path = profile_path.with_suffix(".pickle")
            torch.cuda.memory._dump_snapshot(profile_path.as_posix())
            torch.cuda.memory._record_memory_history(enabled=False)

        # Synchronize the tensor metrics across all steps and ranks
        tensor_stats = tensors.compute_stats()

        # Compute step metrics
        num_tokens = config.data.batch_size * config.data.seq_len
        progress.total_tokens += num_tokens
        progress.total_samples += config.data.batch_size
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0

        # Log step metrics
        step_time = time.time() - step_start_time
        current_lr = optimizer.param_groups[0]["lr"]
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {tensor_stats['loss/mean']:.4f} | Accuracy: {tensor_stats['accuracy/mean']:.4f} | Grad. Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}%"
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

        # Log tensor stats
        tensor_stats["step"] = progress.step
        monitor.log(tensor_stats)

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/save_ckpt": save_ckpt_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        # Log distributions to W&B table if enabled
        if monitor.wandb:
            assert all(len(tensors) == 1 for tensors in tensors.values()), "Tensors must be lists of length 1"
            distributions = {key: tensors[key][0] for key in tensors.keys()}
            monitor.wandb.log_distributions(
                distributions=distributions,
                step=progress.step,
            )

        is_first_step = False
        progress.step += 1

    # Log final (immutable) distributions to W&B table
    if monitor.wandb:
        logger.info("Logging final distributions as W&B table")
        monitor.wandb.log_final_distributions()

    # Write final checkpoint
    if config.ckpt:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step)
        weight_ckpt_manager.save(model, tokenizer, step=progress.step)

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("SFT trainer finished!")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    train(parse_argv(SFTTrainerConfig))


if __name__ == "__main__":
    main()
