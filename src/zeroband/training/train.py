import asyncio
import logging
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import shardcast
import torch
from jaxtyping import Float
from torch._guards import log as torch_log

from zeroband.training import envs
from zeroband.training.ckpt import (
    TrainingProgress,
    load_full_checkpoint,
    save_full_checkpoint,
    save_weight_checkpoint,
)
from zeroband.training.config import Config as TrainingConfig
from zeroband.training.data import DataLoader, FakeDataLoader
from zeroband.training.logger import setup_logger
from zeroband.training.loss import compute_logprobs, entropy_loss, grpo_loss
from zeroband.training.metrics import BatchMetrics
from zeroband.training.model import get_tokenizer, reshard_module, setup_model
from zeroband.training.orchestrator.orchestrator import orchestrate
from zeroband.training.perf import get_perf_counter
from zeroband.training.utils import (
    OffloadedTensor,
    copy_model_to_cpu,
    offload_model_to_cpu,
    wake_up_model_from_cpu,
)
from zeroband.training.world import get_world
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


def run_orchestrator_in_subprocess(orchestrator_config):
    """Run the orchestrator in a completely separate process with its own event loop."""
    # This function runs in a separate process, so it has its own memory space
    # and can initialize its own logger/monitor without conflicts
    asyncio.run(orchestrate(orchestrator_config))


@clean_exit
def train(config: TrainingConfig):
    # Get world info as populated by torchrun
    world = get_world()

    # Setup logger
    logger = setup_logger(config.log, world)
    logger.info(f"Starting trainer on rank {world.rank} and local rank {world.local_rank} ({world.world_size} rank(s)")

    # Setup the monitor
    monitor = setup_monitor(config.monitor, run_config=config)

    # Optionally, sidecar the orchestrator
    orchestrator = None
    if config.orchestrator and world.rank == 0:
        logger.info("Starting orchestrator in a separate process")
        orchestrator = mp.get_context("spawn").Process(
            target=run_orchestrator_in_subprocess, args=(config.orchestrator,), daemon=True
        )
        orchestrator.start()

    # Optionally, clean the checkpoints path
    if config.ckpt.clean:
        logger.info(f"Cleaning checkpoint path {config.ckpt.path}")
        shutil.rmtree(config.ckpt.path, ignore_errors=True)

    # TODO(Mika): Move this to typed env var
    # Allow eager fallback during production so that training runs don't die if compile fails
    if "ZERO_BAND_DEV" not in os.environ:
        torch_log.setLevel(logging.CRITICAL)
        torch._dynamo.config.suppress_errors = True

    torch.set_float32_matmul_precision("high")

    if config.weights.path and world.rank == 0:
        if envs.SHARDCAST_OUTPUT_DIR is not None:
            shardcast.initialize(
                envs.SHARDCAST_OUTPUT_DIR,
                max_distribution_folders=config.async_level + 1,
            )

    # Initialize the model and tokenizer
    model = setup_model(config.model)
    tokenizer = get_tokenizer(config.model)

    # Optionally, initialize a model to compute logprobs
    if config.recompute_logprobs:
        logprob_model = setup_model(config.model)

        # Offload the logprob model to CPU
        tensor_offloaded_repository: dict[int, OffloadedTensor] = {}
        tensor_offloaded_repository[0] = offload_model_to_cpu(logprob_model)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=(config.optim.betas1, config.optim.betas2),
    )

    # TODO(Mika): Add this back but without dependency on the seq_len

    # Optionally, resume training from a checkpoint
    progress = TrainingProgress(total_tokens=0, step=0, total_samples=0)
    if config.ckpt.resume_path:
        logger.info(f"Resuming training from checkpoint {config.ckpt.resume_path}")
        load_full_checkpoint(model, [optimizer], progress, config.ckpt.resume_path)

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    dataloader = DataLoader(config.data.path, progress.step)
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake)

    logger.info("Starting training loop")
    active_weight_checkpoint_paths: list[Path] = []
    while True:
        train_step_start_time = time.time()
        logger.info(f"Starting training step {progress.step}")

        # Load the training batch
        load_data_start_time = time.time()
        micro_batches = dataloader.get_batch()
        load_data_time = time.time() - load_data_start_time
        logger.info(f"Loaded batch in {load_data_time:.2f} seconds")

        # Optionally, Compute the logprobs for the training batch
        compute_logprobs_time = 0
        if config.recompute_logprobs:
            logger.info(f"Starting recomputing logprobs for step {progress.step}")
            compute_logprobs_start_time = time.time()
            og_infer_step = progress.step - config.async_level
            infer_step = max(og_infer_step, 0)

            # Wake up the logprob model from CPU
            wake_up_model_from_cpu(logprob_model, tensor_offloaded_repository[infer_step])
            if og_infer_step == infer_step:
                del tensor_offloaded_repository[infer_step]

            with torch.no_grad():
                num_micro_batches = len(micro_batches)
                for micro_step, micro_batch in enumerate(micro_batches, start=1):
                    logger.debug(f"Computing logprobs for micro batch {micro_step} / {num_micro_batches}")
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
            logger.info(f"Computed logprobs in {compute_logprobs_time:.2f} seconds")

        if config.profile_path and world.rank == 0:
            torch.cuda.memory._record_memory_history()

        batch_metrics = BatchMetrics()
        num_micro_batches = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches, start=1):
            logger.debug(f"Training on micro batch {micro_step} / {num_micro_batches}")
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = (
                torch.ones_like(input_ids).int().to("cuda")
            )  # TODO(Mika): Remove this from loss computation, then here
            logprobs = micro_batch["logprobs"].to("cuda")
            temperature = micro_batch["temperature"]
            total_tokens = micro_batch["total_tokens"]
            micro_batch_size, seq_len = input_ids.shape

            # Optionally, normalize the loss to the token count
            max_tokens = micro_batch_size * seq_len
            if config.loss.normalize_to_token_count:
                max_tokens = int(total_tokens)

            # Forward pass
            logger.debug(f"Forwarding on micro batch {micro_step} / {num_micro_batches}")
            logits: Float[torch.Tensor, "batch seq vocab"] = model(
                input_ids=input_ids, position_ids=position_ids
            ).logits.contiguous()

            # Compute loss
            loss, clip_ratio = grpo_loss(
                logits,
                input_ids,
                advantages,
                logprobs,
                loss_mask,
                temperature,
                max_tokens,
                config.loss.variant,
            )

            # Compute the entropy
            with torch.no_grad():
                logger.debug(f"Computing entropy on micro batch {micro_step} / {num_micro_batches}")
                entropy = entropy_loss(logits, loss_mask, temperature, max_tokens)

            # Now we can delete the micro batch CUDA tensors
            del logits, input_ids, position_ids, advantages, loss_mask, logprobs

            # Scale the loss by the number of micro batches (=gradient accumulation steps)
            loss = loss / num_micro_batches

            # Backward pass (ensures loss reduction across FSDP ranks)
            logger.debug(f"Backward pass on micro batch {micro_step} / {num_micro_batches}")
            loss.backward()

            batch_metrics.update("loss/loss", loss.detach().clone())
            batch_metrics.update("loss/entropy", entropy.detach().clone())
            batch_metrics.update("loss/clip_ratio", clip_ratio.detach().clone())

            logger.debug(
                f"Finished training on micro batch {micro_step} / {num_micro_batches} (loss: {loss.item():.2f}, entropy: {entropy.item():.2f}, clip_ratio: {clip_ratio.item():.2f})"
            )

            del loss, entropy, clip_ratio

        # Synchronize the batch metrics across all ranks
        logger.info("Synchronizing batch metrics across all ranks")
        batch_metrics.sync()

        # Optionally, clip the gradients
        logger.info(f"Clipping gradients with max norm {config.loss.max_norm}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.loss.max_norm).full_tensor()  # type: ignore (is a dtensor)
        batch_metrics.update("loss/grad_norm", grad_norm.detach().clone())

        # Update the model parameters
        logger.info("Updating model parameters")
        optimizer.step()
        optimizer.zero_grad()
        progress.step += 1

        # Update progress
        num_local_tokens = micro_batch_size * seq_len * num_micro_batches
        num_tokens = world.world_size * num_local_tokens
        batch_size = micro_batch_size * num_micro_batches
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size
        # Log progress metrics
        progress_metrics = {
            "step": progress.step,
            "progress/num_tokens": progress.total_tokens,
            "progress/num_samples": progress.total_samples,
        }
        if world.rank == 0:
            monitor.log(progress_metrics)

        # Update the performance counter
        perf_counter = get_perf_counter(model, seq_len)
        perf_counter.count_tokens(num_tokens)

        # Loss loss metrics
        loss_metrics = {key: value.item() for key, value in batch_metrics.items()}
        if world.rank == 0:
            monitor.log(loss_metrics)

        # Log Performance metrics
        tokens_per_second = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        tokens_per_second_per_gpu = tokens_per_second / world.world_size
        perf_metrics = {
            "step": progress.step,
            "perf/tokens_per_second": tokens_per_second,
            "perf/tokens_per_second_per_gpu": tokens_per_second_per_gpu,
            "perf/mfu": mfu,
        }
        if world.rank == 0:
            monitor.log(perf_metrics)

        # Save the weight checkpoint
        step_path = Path(config.weights.path) / f"step_{progress.step}"
        save_weights_start_time = time.time()
        model_path = save_weight_checkpoint(model, tokenizer, step_path, async_save=config.weights.save_async)
        active_weight_checkpoint_paths.append(step_path)
        save_weights_time = time.time() - save_weights_start_time

        # Optionally, broadcast the weight checkpoint from master rank
        if world.rank == 0 and envs.SHARDCAST_OUTPUT_DIR is not None:
            logger.info(f"Broadcasting {model_path} via shardcast")
            shardcast.broadcast(model_path)  # TODO: Is this blocking?

        # Optionally, remove old weight checkpoints to save space
        # +1 to ensure to not delete current checkpoint when async_level=0
        if len(active_weight_checkpoint_paths) > config.async_level + 1:
            path_to_delete = active_weight_checkpoint_paths.pop(0)
            ckpt_step = int(path_to_delete.name.split("_")[-1])
            should_keep = config.weights.interval and ckpt_step % config.weights.interval == 0
            if not should_keep:
                logger.info(f"Removing past weight checkpoint at {path_to_delete}")
                shutil.rmtree(path_to_delete, ignore_errors=True)

        # Optionally, dump memory snapshot
        if config.profile_path and progress.step == 2 and world.rank == 0:
            logger.info("Dumping memory snapshot")
            profile_path = config.profile_path
            if not profile_path.endswith(".pickle"):
                profile_path += ".pickle"
            torch.cuda.memory._dump_snapshot(profile_path)
            torch.cuda.memory._record_memory_history(enabled=False)

        # Optionally, save the full checkpoint
        save_ckpt_time = 0
        if config.ckpt and config.ckpt.interval and progress.step % config.ckpt.interval == 0:
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            save_full_checkpoint(model, [optimizer], progress, config.ckpt.path)
            save_ckpt_time = time.time() - save_ckpt_start_time

        # Update the CPU logprob model to updated model
        if config.recompute_logprobs:
            reshard_module(logprob_model)
            tensor_offloaded_repository[progress.step] = copy_model_to_cpu(model)

        train_step_time = time.time() - train_step_start_time
        logger.success(
            f"Finished training step {progress.step} in {train_step_time:.2f}s (Loss: {loss_metrics['loss/loss']:.2f}, Entropy: {loss_metrics['loss/entropy']:.2f}, Clip Ratio: {loss_metrics['loss/clip_ratio']:.2f}, Throughput: {tokens_per_second:.2f} tokens/s, MFU: {mfu:.2f}%)"
        )

        # Log time metrics
        time_metrics = {
            "step": progress.step,
            "time/step": train_step_time,
            "time/load_data": load_data_time,
            "time/save_weights": save_weights_time,
            "time/compute_logprobs": compute_logprobs_time,
            "time/save_ckpt": save_ckpt_time,
        }
        if world.rank == 0:
            monitor.log(time_metrics)

        if config.max_steps and progress.step >= config.max_steps:
            break

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("Training finished!")

    # Clean up orchestrator process if it was started
    if orchestrator and orchestrator.is_alive():
        logger.info("Terminating orchestrator process")
        orchestrator.terminate()
        orchestrator.join(timeout=5)
        if orchestrator.is_alive():
            logger.warning("Orchestrator process did not terminate gracefully, forcing kill")
            orchestrator.kill()
            orchestrator.join()


def main():
    train(parse_argv(TrainingConfig))


if __name__ == "__main__":
    main()
