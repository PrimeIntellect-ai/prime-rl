import logging
import multiprocessing as mp
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import shardcast
import torch
import torch.distributed as dist
from torch._guards import log as torch_log

from zeroband.training import envs
from zeroband.training.ckpt import (
    TrainingProgress,
    load_full_checkpoint,
    save_full_checkpoint,
    save_weight_checkpoint,
)
from zeroband.training.config import TrainingConfig
from zeroband.training.data import DataLoader, FakeDataLoader
from zeroband.training.logger import setup_logger
from zeroband.training.loss import compute_logprobs, entropy_loss, grpo_loss
from zeroband.training.model import forward, get_tokenizer, reshard_module, setup_model
from zeroband.training.orchestrator.orchestrator import run_orchestrator
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


@clean_exit
def train(config: TrainingConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(config.log, world)
    logger.info(f"Starting trainer in {world}")
    logger.debug(f"ModelConfig({config.model})")
    logger.debug(f"DataConfig({config.data})")
    logger.debug(f"OptimizerConfig({config.optim})")
    logger.debug(f"CheckpointConfig({config.ckpt})")
    logger.debug(f"WeightCheckpointConfig({config.weights})")
    logger.debug(f"GRPOLossConfig({config.loss})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(config.monitor, run_config=config)

    # Optionally, sidecar the orchestrator
    orchestrator = None
    if config.orchestrator and world.rank == 0:
        config.orchestrator.num_train_workers = world.world_size

        logger.info("Starting orchestrator in a separate process")

        # Create a queue for orchestrator to signal when setup is complete
        ctx = mp.get_context("spawn")
        setup_queue = ctx.Queue()
        orchestrator = ctx.Process(
            target=run_orchestrator,
            args=(config.orchestrator, setup_queue),
            daemon=True,
        )
        orchestrator.start()

        # Wait for orchestrator to signal that setup is complete
        logger.info("Waiting for orchestrator to complete setup...")
        signal = setup_queue.get()
        if signal == "ready":
            logger.success("Orchestrator setup complete, continuing with training")
        else:
            raise RuntimeError(f"Unexpected signal from orchestrator: {signal}")

    # Optionally, clean the checkpoints path
    if config.ckpt.clean:
        logger.debug(f"Cleaning checkpoint path {config.ckpt.path}")
        shutil.rmtree(config.ckpt.path, ignore_errors=True)

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
                max_distribution_folders=config.async_level + 1,
            )

    # Initialize the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model})")
    model = setup_model(config.model)
    tokenizer = get_tokenizer(config.model)

    # Optionally, initialize a model to compute logprobs
    if config.recompute_logprobs:
        logger.info(f"Initializing logprob model ({config.model})")
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

    # Optionally, resume training from a checkpoint
    progress = TrainingProgress()
    if config.ckpt.resume_path:
        logger.info(f"Resuming training from checkpoint {config.ckpt.resume_path}")
        load_full_checkpoint(model, [optimizer], progress, config.ckpt.resume_path)

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    logger.info(f"Initializing data loader ({config.data})")
    dataloader = DataLoader(config.data.path, progress.step)
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake)

    logger.info(f"Starting training loop ({config.max_steps=})")
    active_weight_checkpoint_paths: list[Path] = []
    while True:
        logger.debug(f"Training step {progress.step}")
        step_start_time = time.time()

        # Check if orchestrator is still alive (only on rank 0)
        if orchestrator and world.rank == 0:
            if not orchestrator.is_alive():
                if orchestrator.exitcode == 0:
                    logger.info("Detected that orchestrator is finished!")
                else:
                    logger.error(f"Orchestrator process died with exit code {orchestrator.exitcode}")
                    raise RuntimeError(f"Orchestrator process died with exit code {orchestrator.exitcode}")

        # Load the training batch
        logger.debug("Loading training batch")
        load_data_start_time = time.time()
        micro_batches = dataloader.get_batch()
        load_data_time = time.time() - load_data_start_time
        logger.debug(f"Loaded batch in {load_data_time:.2f} seconds")

        # Optionally, Compute the logprobs for the training batch
        compute_logprobs_time = 0
        if config.recompute_logprobs:
            logger.debug("Recomputing logprobs")
            compute_logprobs_start_time = time.time()
            og_infer_step = progress.step - 1 - config.async_level  # -1 because we haven't updated the model yet
            infer_step = max(og_infer_step, 0)

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
            logger.debug(f"Computed logprobs in {compute_logprobs_time:.2f} seconds")

        if config.profile_path and world.rank == 0:
            torch.cuda.memory._record_memory_history()

        loss_metrics = defaultdict(float)
        num_micro_batches = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches, start=1):
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            logprobs = micro_batch["logprobs"].to("cuda")
            temperature = micro_batch["temperature"]
            total_tokens = micro_batch["total_tokens"]
            micro_batch_size, seq_len = input_ids.shape

            # Optionally, normalize the loss to the token count
            max_tokens = micro_batch_size * seq_len
            if config.loss.normalize_to_token_count:
                max_tokens = int(total_tokens)

            # Forward pass
            logits = forward(model, input_ids, position_ids).contiguous()

            # Compute loss
            loss, ratio_info = grpo_loss(
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
                entropy = entropy_loss(logits, loss_mask, temperature, max_tokens)

            # Now we can delete the micro batch CUDA tensors
            del micro_batch, logits, input_ids, position_ids, advantages, loss_mask, logprobs

            # Scale loss, entropy, and clip ratio by the number of micro batches (=gradient accumulation steps)
            loss = loss / num_micro_batches
            entropy = entropy / num_micro_batches

            # Backward pass (ensures loss reduction across FSDP ranks)
            logger.debug(f"Backward pass on micro batch {micro_step} / {num_micro_batches}")
            loss.backward()

            loss_metrics["loss/loss"] += loss.detach().clone()
            loss_metrics["loss/entropy"] += entropy.detach().clone()

            for key, value in ratio_info.items():
                loss_metrics[f"loss/{key}"] += value.detach().clone() / num_micro_batches

            del loss, entropy

        # Synchronize the batch metrics across all ranks
        for key, value in loss_metrics.items():
            dist.all_reduce(value.to("cuda"), op=dist.ReduceOp.AVG)
            loss_metrics[key] = value
        # Optionally, clip the gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.loss.max_norm).full_tensor()
        loss_metrics["loss/grad_norm"] += grad_norm.detach().clone()

        # Update the model parameters
        logger.debug("Updating model")
        optimizer.step()
        optimizer.zero_grad()

        # Save the weight checkpoint
        step_path = Path(config.weights.path) / f"step_{progress.step}"
        save_weights_start_time = time.time()
        model_path = save_weight_checkpoint(model, tokenizer, step_path, async_save=config.weights.save_async)
        active_weight_checkpoint_paths.append(step_path)
        save_weights_time = time.time() - save_weights_start_time

        # Optionally, broadcast the weight checkpoint from master rank
        if world.rank == 0 and envs.SHARDCAST_OUTPUT_DIR is not None:
            logger.info(f"Broadcasting weights from {model_path} via shardcast")
            shardcast.broadcast(model_path)  # TODO: Is this blocking?

        # Optionally, remove old weight checkpoints to save space
        # +1 to ensure to not delete current checkpoint when async_level=0
        if len(active_weight_checkpoint_paths) > config.async_level + 1:
            path_to_delete = active_weight_checkpoint_paths.pop(0)
            ckpt_step = int(path_to_delete.name.split("_")[-1])
            should_keep = config.weights.interval and ckpt_step % config.weights.interval == 0
            if not should_keep:
                logger.debug(f"Removing past weight checkpoint ({path_to_delete})")
                shutil.rmtree(path_to_delete, ignore_errors=True)

        # Optionally, dump memory snapshot
        if config.profile_path and progress.step == 2 and world.rank == 0:
            logger.debug("Dumping memory snapshot")
            profile_path = config.profile_path
            if not profile_path.endswith(".pickle"):
                profile_path += ".pickle"
            torch.cuda.memory._dump_snapshot(profile_path)
            torch.cuda.memory._record_memory_history(enabled=False)

        # Optionally, save the full checkpoint
        save_ckpt_time = 0
        if config.ckpt and config.ckpt.interval and progress.step % config.ckpt.interval == 0:
            logger.debug(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            save_full_checkpoint(model, [optimizer], progress, config.ckpt.path)
            save_ckpt_time = time.time() - save_ckpt_start_time

        # Update the CPU logprob model to updated model
        if config.recompute_logprobs:
            logger.debug("Offloading updated model to CPU")
            reshard_module(logprob_model)
            tensor_offloaded_repository[progress.step] = copy_model_to_cpu(model)

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
        step_message = f"Training     | step {progress.step} | Time:{step_time:.2f}s | Loss: {loss_metrics['loss/loss']:.2f} | Entropy: {loss_metrics['loss/entropy']:.2f} | Clip: {loss_metrics['loss/clip_ratio']:.2f} | {throughput:.0f} tokens/s | MFU: {mfu:.1f}%"

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
            "time/train/load_data": load_data_time,
            "time/train/save_weights": save_weights_time,
            "time/train/compute_logprobs": compute_logprobs_time,
            "time/train/save_ckpt": save_ckpt_time,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        if config.max_steps and progress.step >= config.max_steps:
            break

        progress.step += 1

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
