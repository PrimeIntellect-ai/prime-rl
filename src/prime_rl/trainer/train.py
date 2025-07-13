import logging
import os
import time
from collections import defaultdict
from copy import deepcopy

# Import environment before any other imports
# ruff: noqa: I001
from prime_rl.trainer import envs

import shardcast
import torch
import torch.distributed as dist
from torch._guards import log as torch_log
from loguru import logger
from prime_rl.trainer.ckpt import CheckpointManager, Progress
from prime_rl.trainer.weights import WeightCheckpointManager
from prime_rl.trainer.config import TrainerConfig
from prime_rl.trainer.data import DataLoader, FakeDataLoader
from prime_rl.trainer.logger import setup_logger
from prime_rl.trainer.loss import compute_logprobs, entropy_loss, grpo_loss
from prime_rl.trainer.model import forward, get_tokenizer, reshard_module, setup_model
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    OffloadedTensor,
    copy_model_to_cpu,
    offload_model_to_cpu,
    wake_up_model_from_cpu,
    print_benchmark,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, to_col_format


@clean_exit
@logger.catch(reraise=True)
def train(config: TrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(config.log, world)
    logger.info(f"Starting trainer in {world}")

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps}, {config.data.fake=})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(config.monitor, run_config=config)

    # TODO(Mika): Move this to typed env var
    # Allow eager fallback during production so that training runs don't die if compile fails
    if "ZERO_BAND_DEV" not in os.environ:
        torch_log.setLevel(logging.CRITICAL)
        torch._dynamo.config.suppress_errors = True

    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(world.rank)

    if config.weights.path and world.rank == 0:
        if envs.SHARDCAST_OUTPUT_DIR is not None:
            logger.info(f"Initializing shardcast from {envs.SHARDCAST_OUTPUT_DIR}")
            shardcast.initialize(
                envs.SHARDCAST_OUTPUT_DIR,
                # +1 to ensure to not delete current checkpoint when async_level=0
                max_distribution_folders=config.async_level + 1,
            )

    # Initialize the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model})")
    model = setup_model(config.model)
    tokenizer = get_tokenizer(config.model)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=(config.optim.betas1, config.optim.betas2),
    )

    # Get checkpoint managers
    logger.info(f"Initializing weight checkpoint manager ({config.weights})")
    weight_ckpt_manager = WeightCheckpointManager(config.weights, config.ckpt, config.async_level)
    if config.ckpt:
        logger.info(f"Initializing checkpoint manager ({config.ckpt})")  # TODO: Remove this
        ckpt_manager = CheckpointManager(config.ckpt)

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if config.ckpt and config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step `{config.ckpt.resume_step}`")
        ckpt_manager.load(model, [optimizer], progress, step=config.ckpt.resume_step)
    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples})"
    )

    # Optionally, initialize a model to compute logprobs
    if config.recompute_logprobs:
        # Initialize the logprob model
        tensor_offloaded_repository: dict[int, OffloadedTensor] = {}
        logger.info(f"Initializing logprob model ({config.model})")
        logprob_model = setup_model(config.model)

        # Load async models from weights checkpoint if resuming from checkpoint
        if config.ckpt and config.ckpt.resume_step:
            for step in range(max(progress.step - config.async_level, 0), progress.step):
                logger.info(f"Initializing logprob model ({config.model}) for step {step}")
                model_name_or_path = (
                    config.model.name
                    if not (config.ckpt and config.ckpt.resume_step)
                    else config.weights.path / f"step_{step}"
                )
                model_config = deepcopy(config.model)
                model_config.name = model_name_or_path
                logprob_model = setup_model(model_config)
                tensor_offloaded_repository[step] = offload_model_to_cpu(logprob_model)

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    logger.info(f"Initializing data loader ({config.data})")
    dataloader = DataLoader(config.data.path, progress.step)
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake)

    logger.info(f"Starting training loop ({config.max_steps=})")
    while True:
        # Save the weight checkpoint (if we are not at the first step, because no updates to the model have been made yet)
        save_weights_time = 0
        if progress.step > 0:
            save_weights_start_time = time.time()
            model_path = weight_ckpt_manager.save(model, tokenizer, step=progress.step)
            save_weights_time = time.time() - save_weights_start_time

        # Save the full checkpoint (if we are at an interval step and not at the first step)
        save_ckpt_time = 0
        if config.ckpt and config.ckpt.interval and progress.step > 0 and progress.step % config.ckpt.interval == 0:
            logger.debug(f"Saving checkpoint {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(model, [optimizer], progress, step=progress.step)
            save_ckpt_time = time.time() - save_ckpt_start_time

        # Break if we have reached the maximum number of steps
        if config.max_steps and progress.step >= config.max_steps:
            break

        logger.info(f"Starting training step {progress.step}")
        step_start_time = time.time()

        # Offload the current model to CPU for logprob computation
        if config.recompute_logprobs:
            logger.debug(f"Offloading model for step {progress.step} to CPU for future logprob calculation")
            reshard_module(logprob_model)
            tensor_offloaded_repository[progress.step] = copy_model_to_cpu(model)

        # Wait for the batch to be available
        logger.info("Waiting for training batch to arrive")
        wait_for_batch_start_time = time.time()
        dataloader.wait_for_batch()
        wait_for_batch_time = time.time() - wait_for_batch_start_time
        logger.debug(f"Waited for batch to arrive for {wait_for_batch_time:.2f} seconds")

        # Load the training batch
        logger.debug("Loading batch")
        load_data_start_time = time.time()
        micro_batches = dataloader.get_batch()
        load_data_time = time.time() - load_data_start_time
        logger.debug(f"Loaded batch in {load_data_time:.2f} seconds")

        # Optionally, compute the logprobs for the training batch
        compute_logprobs_time = 0
        if config.recompute_logprobs:
            compute_logprobs_start_time = time.time()
            og_infer_step = progress.step - config.async_level
            infer_step = max(og_infer_step, 0)
            logger.info(f"Recomputing logprobs with model weight checkpoint {infer_step}")

            # Wake up the logprob model from CPU
            wake_up_model_from_cpu(logprob_model, tensor_offloaded_repository[infer_step])
            if og_infer_step == infer_step:
                del tensor_offloaded_repository[infer_step]

            with torch.no_grad():
                num_micro_batches = len(micro_batches)
                for micro_step, micro_batch in enumerate(micro_batches, start=1):
                    input_ids = micro_batch["input_ids"].to("cuda")
                    position_ids = micro_batch["position_ids"].to("cuda")
                    temperature = micro_batch["temperature"]

                    logprobs = compute_logprobs(logprob_model, input_ids, position_ids, temperature)
                    micro_batch["logprobs"] = logprobs.to("cpu")

            # here we sepcifically don't save the tensor offloaded, they are alreay consumed and we will never use it again.
            # this avoid having to make sure we don't keep too much tensor offloaded in cpu memory
            reshard_module(logprob_model)
            offload_model_to_cpu(logprob_model)

            compute_logprobs_time = time.time() - compute_logprobs_start_time
            logger.debug(f"Recomputed logprobs in {compute_logprobs_time:.2f} seconds")

        if config.profile_path and world.rank == 0:
            torch.cuda.memory._record_memory_history()

        forward_backward_start_time = time.time()
        loss_metrics = defaultdict(float)
        num_micro_batches = len(micro_batches)
        micro_batch_size, seq_len = micro_batches[0]["input_ids"].shape
        batch_size = micro_batch_size * num_micro_batches

        # Normalize by the number of unmasked tokens in the batch (per-batch length normalization)
        loss_scale = sum(micro_batch["loss_mask"].sum() for micro_batch in micro_batches)

        logger.info(f"Starting forward and backward pass ({num_micro_batches=}, {loss_scale=})")
        for micro_step, micro_batch in enumerate(micro_batches, start=1):
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            logprobs = micro_batch["logprobs"].to("cuda")
            temperature = micro_batch["temperature"]
            micro_batch_size, seq_len = input_ids.shape

            # Forward pass
            logits = forward(model, input_ids, position_ids).contiguous()

            # Compute loss
            loss, importance_ratio = grpo_loss(
                logits=logits,
                input_ids=input_ids,
                advantages=advantages,
                original_logprobs=logprobs,
                loss_mask=loss_mask,
                temperature=temperature,
                grpo_loss_config=config.loss.variant,
            )

            # Compute entropy
            with torch.no_grad():
                entropy = entropy_loss(
                    logits=logits,
                    loss_mask=loss_mask,
                    temperature=temperature,
                )

            # Accumulate unnormalized local metrics
            loss_metrics["loss/loss"] += loss.detach().clone().float()
            loss_metrics["loss/entropy"] += entropy.detach().clone().float()
            loss_metrics["loss/importance_ratio"] += importance_ratio.detach().clone().float()

            # Scale loss by scale factor before backward pass
            loss = loss / loss_scale

            # Backward pass (ensures loss reduction across FSDP ranks)
            loss.backward()

            # We report per-micro batch length normalized metrics here
            logger.debug(
                f"Completed micro batch {micro_step}/{num_micro_batches} (loss={(loss.item() / loss_mask.sum()):.2f}, entropy={(entropy.item() / loss_mask.sum()):.2f}, importance_ratio={(importance_ratio.item() / loss_mask.sum()):.2f})"
            )

            del micro_batch, logits, input_ids, position_ids, advantages, logprobs, loss_mask
            del loss, entropy, importance_ratio

        # Normalize all loss metrics globally before reporting
        for key, value in loss_metrics.items():
            loss_metrics[key] = value / loss_scale

        # Synchronize the batch metrics across all ranks
        logger.debug(f"All-reduce loss metrics keys {list(loss_metrics.keys())}")
        for key, value in loss_metrics.items():
            dist.all_reduce(value.to("cuda"), op=dist.ReduceOp.AVG)
            loss_metrics[key] = value

        # Optionally, clip the gradients
        logger.debug(f"Clipping gradients to {config.loss.max_norm}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.loss.max_norm).full_tensor()
        loss_metrics["loss/grad_norm"] += grad_norm.detach().clone()

        # Update the model parameters
        logger.debug("Updating model")
        optimizer.step()
        optimizer.zero_grad()

        forward_backward_time = time.time() - forward_backward_start_time

        # Optionally, broadcast the weight checkpoint from master rank
        if world.rank == 0 and envs.SHARDCAST_OUTPUT_DIR is not None:
            logger.info(f"Broadcasting weights from {model_path} via shardcast")
            shardcast.broadcast(model_path.as_posix())  # TODO: Is this blocking?

        # Maybe clean up weight checkpoint
        weight_ckpt_manager.maybe_clean(progress.step)

        # Optionally, dump memory snapshot
        if config.profile_path and progress.step == 2 and world.rank == 0:
            logger.debug("Dumping memory snapshot")
            profile_path = config.profile_path
            if not profile_path.suffix == ".pickle":
                profile_path = profile_path.with_suffix(".pickle")
            torch.cuda.memory._dump_snapshot(profile_path.as_posix())
            torch.cuda.memory._record_memory_history(enabled=False)

        # Compute step metrics
        num_local_tokens = micro_batch_size * seq_len * num_micro_batches
        num_tokens = world.world_size * num_local_tokens
        batch_size = micro_batch_size * num_micro_batches
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size
        perf_counter = get_perf_counter(model, seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        loss_metrics = {key: value.item() for key, value in loss_metrics.items()}

        # Log step metrics
        step_time = time.time() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {loss_metrics['loss/loss']:.2f} | Entropy: {loss_metrics['loss/entropy']:.2f} | Importance Ratio: {loss_metrics['loss/importance_ratio']:.2f} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}%"
        logger.success(step_message)

        # Log progress metrics
        progress_metrics = {
            "progress/train/total_tokens": progress.total_tokens,
            "progress/train/total_samples": progress.total_samples,
            "progress/train/step": progress.step,  # Shared W&B axis
            "step": progress.step,
        }
        monitor.log(progress_metrics)

        # Log performance metrics
        perf_metrics = {
            "perf/train/throughput": throughput,
            "perf/train/mfu": mfu,
            "step": progress.step,
        }
        monitor.log(perf_metrics)

        # Log loss metrics
        loss_metrics["step"] = progress.step
        monitor.log(loss_metrics)

        # Log time metrics
        time_metrics = {
            "time/train": step_time,
            "time/train/wait_for_batch": wait_for_batch_time,
            "time/train/load_data": load_data_time,
            "time/train/save_weights": save_weights_time,
            "time/train/compute/logprobs": compute_logprobs_time,
            "time/train/save_ckpt": save_ckpt_time,
            "time/train/compute/forward_backward": forward_backward_time,
            "time/train/compute": forward_backward_time + compute_logprobs_time,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        progress.step += 1

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("Trainer finished!")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    train(parse_argv(TrainerConfig))


if __name__ == "__main__":
    main()
