import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import shardcast
import torch
import torch.distributed as dist
import torch.distributed.tensor
from jaxtyping import Float
from torch._guards import log as torch_log
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from zeroband.training import envs
from zeroband.training.ac import setup_ac
from zeroband.training.ckpt import (
    TrainingProgress,
    load_full_checkpoint,
    save_full_checkpoint,
    save_weight_checkpoint,
)
from zeroband.training.config import Config as TrainingConfig
from zeroband.training.data import DataLoader
from zeroband.training.logger import setup_logger
from zeroband.training.loss import entropy_loss, grpo_loss, selective_log_softmax
from zeroband.training.utils import (
    MetricsAverager,
    OffloadedTensor,
    copy_model_to_cpu,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.training.world_info import WorldInfo, get_world_info
from zeroband.utils.models import ModelType, get_model_and_tokenizer
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


def get_local_batch_size(batch_size: int, micro_bs: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    return batch_size


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=layer_reshard_after_forward,
        )
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_logprobs(
    model: ModelType,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(
        input_ids=input_ids, position_ids=position_ids
    ).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@clean_exit
def train(config: TrainingConfig):
    # Get world info as populated by torchrun
    world = get_world_info()

    # Setup logger
    logger = setup_logger(config.log, world)
    logger.info(f"Starting trainer on rank {world.rank} and local rank {world.local_rank} ({world.world_size} rank(s)")

    # Setup the monitor
    monitor = setup_monitor(config.monitor, run_config=config)

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
    if config.seed:
        seed_everything(config.seed)

    # local_batch_size = get_local_batch_size(config.optim.batch_size, config.train.micro_bs, world_info)

    if config.weights.path and world.rank == 0:
        if envs.SHARDCAST_OUTPUT_DIR is not None:
            shardcast.initialize(
                envs.SHARDCAST_OUTPUT_DIR,
                max_distribution_folders=config.max_async_level,
            )

    # Initialize the model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config.model.name, config.model.attn)

    # TODO(Mika): Add this back but without dependency on the seq_len
    # perf_counter = PerfCounter(window_size=10, model=model, seq_len=config.data.seq_length)

    # Optionally, apply activation checkpointing
    if config.ac:
        setup_ac(model, config.ac)

    # Shard the model for training using FSDP
    apply_fsdp(model, config.reshard_after_forward)

    # Optionally, initialize a model to compute logprobs
    if config.recompute_logprobs:
        model_for_logprob_only, _ = get_model_and_tokenizer(config.model.name, config.model.attn)
        apply_fsdp(model_for_logprob_only, config.reshard_after_forward)

    tensor_offloaded_repository: dict[int, OffloadedTensor] = {}
    if config.recompute_logprobs:
        tensor_offloaded_repository[0] = offload_model_to_cpu(model_for_logprob_only)

    # Optionally, compile the model
    if config.model.compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

        if config.recompute_logprobs:
            model_for_logprob_only: ModelType = torch.compile(model_for_logprob_only)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=(config.optim.betas1, config.optim.betas2),
    )

    # Optionally, resume training from a checkpoint
    progress = TrainingProgress(total_tokens=0, step=0, total_samples=0)
    if config.ckpt.resume_path:
        logger.info(f"Resuming training from checkpoint {config.ckpt.resume_path}")
        load_full_checkpoint(model, [optimizer], progress, config.ckpt.resume_path)

    # Set up the data loader
    train_dataloader = DataLoader(config.data.path, progress.step)

    # TODO(Mika): Add this back but without dependency on the seq_len
    # if config.data.fake:
    #     train_dataloader = FakeDataLoader(
    #         config.data.seq_length, tokenizer.pad_token_id, config.train.micro_bs, local_batch_size
    #     )

    weight_checkpoint_paths = []
    logger.info("Starting training loop")
    while True:
        train_step_start_time = time.time()
        logger.info(f"Starting training step {progress.step}")

        # Load the training batch
        load_data_start_time = time.time()
        micro_batches = train_dataloader.get_batch()
        load_data_time = time.time() - load_data_start_time
        logger.info(f"Loaded batch in {load_data_time:.2f} seconds")

        # Optionally, Compute the logprobs for the training batch
        compute_logprobs_start_time = time.time()
        with torch.no_grad():
            if config.recompute_logprobs:
                og_infer_step = progress.step - config.max_async_level
                infer_step = max(og_infer_step, 0)
                wake_up_model_from_cpu(model_for_logprob_only, tensor_offloaded_repository[infer_step])

                if og_infer_step == infer_step:
                    del tensor_offloaded_repository[infer_step]

            logger.info(f"Starting recomputing logprobs for step {progress.step}")

            num_grad_acc_steps = len(micro_batches)

            for grad_acc_step in range(num_grad_acc_steps):
                batch = micro_batches[grad_acc_step]

                # Only compute logprobs if not using vllm logprobs or if the batch doesn't have them
                if config.recompute_logprobs:
                    logger.debug(
                        f"log prob grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['token_ids'].shape}"
                    )

                    input_ids = batch["token_ids"].to("cuda")

                    model_for_logprob = model_for_logprob_only if config.recompute_logprobs else model
                    per_token_logps = get_logprobs(
                        model_for_logprob, input_ids, batch["position_ids"], batch["temperature"]
                    )

                    batch["logprobs"] = per_token_logps.to("cpu")

            if config.recompute_logprobs:
                # here we sepcifically don't save the tensor offloaded, they are alreay consumed and we will never use it again.
                # this avoid having to make sure we don't keep too much tensor offloaded in cpu memory
                reshard_module(model_for_logprob_only)
                offload_model_to_cpu(model_for_logprob_only)

        compute_logprobs_time = time.time() - compute_logprobs_start_time
        logger.info(f"Recomputed logprobs in {compute_logprobs_time:.2f} seconds")

        metric_averager = MetricsAverager()
        loss_batch = torch.tensor(0.0, device="cuda")

        if config.memory_profile and world.rank == 0:
            torch.cuda.memory._record_memory_history()

        num_micro_batches = len(micro_batches)
        for micro_batch_idx, micro_batch in enumerate(micro_batches, start=1):
            logger.debug(f"Starting training on micro batch {micro_batch_idx} / {num_grad_acc_steps}")
            input_ids = micro_batch["token_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = (
                torch.ones_like(input_ids).int().to("cuda")
            )  # TODO(Mika): Remove this from loss computation, then here
            original_logprobs = micro_batch["logprobs"].to("cuda")
            temperature = micro_batch["temperature"]
            total_tokens = micro_batch["total_tokens"]
            micro_batch_size, seq_len = input_ids.shape

            if config.normalize_batch_to_token_count:
                max_tokens = int(total_tokens)
            else:
                max_tokens = input_ids.shape[0] * input_ids.shape[1]

            # Forward pass
            logits: Float[torch.Tensor, "batch seq vocab"] = model(
                input_ids=input_ids, position_ids=position_ids
            ).logits.contiguous()

            # Loss
            loss, clip_ratio = grpo_loss(
                logits,
                input_ids,
                advantages,
                original_logprobs,
                loss_mask,
                batch["temperature"],
                max_tokens,
                config.loss.variant,
            )

            with torch.no_grad():
                entropy = entropy_loss(logits, loss_mask, temperature, max_tokens)

            # Scale the loss by the number of micro batches (=gradient accumulation steps)
            loss = loss / num_micro_batches

            # Now we can delete the batch data
            del batch, logits, input_ids, advantages, loss_mask, original_logprobs

            # Backward
            loss.backward()
            loss_batch += loss.detach().clone()

            metric_averager.update("losses/entropy_loss", entropy.detach().clone())

            if clip_ratio is not None:
                metric_averager.update("losses/clip_ratio", clip_ratio.detach().clone())

            del loss, entropy, clip_ratio

        metric_averager.sync()

        # All reduce the loss after gradient accumulation
        dist.all_reduce(loss_batch, op=dist.ReduceOp.AVG)

        # Optinally, clip the gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip).full_tensor()  # type: ignore (is a dtensor)

        optimizer.step()
        optimizer.zero_grad()

        progress.step += 1
        inner_lr = [group["lr"] for group in optimizer.param_groups][0]

        token_per_gpu = micro_batch_size * seq_len * num_micro_batches
        new_tokens = world.world_size * token_per_gpu
        # perf_counter.count_tokens(new_tokens)
        # progress.total_tokens += new_tokens
        # progress.total_samples += len()

        metrics = {
            "step": progress.step,
            "train/total_tokens": progress.total_tokens,
            "train/total_samples": progress.total_samples,
            "losses/loss": loss_batch.item(),
            "train/inner_lr": inner_lr,
            "losses/grad_norm": grad_norm.item(),
        }

        for key, value in metric_averager.items():
            metrics[key] = value.item()

        log = f"Step: {progress.step}, loss: {loss_batch.item():.4f}, "

        del loss_batch, grad_norm

        # tokens_per_second = perf_counter.get_tokens_per_second()
        # if tokens_per_second is not None:
        #     tokens_per_second_per_gpu = tokens_per_second / world_info.world_size
        #     mfu = perf_counter.get_mfu()
        #     metrics.update(
        #         {
        #             "perf/tokens_per_second": tokens_per_second,
        #             "perf/tokens_per_second_per_gpu": tokens_per_second_per_gpu,
        #             "perf/mfu": mfu,
        #         }
        #     )

        # log += f", tokens_per_second: {tokens_per_second:.2f}, tokens_per_second_per_gpu: {tokens_per_second_per_gpu:.2f}, mfu: {mfu:.2f}"

        if world.rank == 0:
            monitor.log(metrics)

        logger.info(log)

        time_rollout_ckpt = None
        time_shardcast = None
        time_rollout_delete = None

        # Lets do this first so that clients can start downloading as soon as possible
        if config.weights.path:
            step_path = Path(config.weights.path) / f"step_{progress.step}"
            weight_checkpoint_paths.append(step_path)
            t0 = time.time()
            model_path = save_weight_checkpoint(model, tokenizer, step_path, async_save=config.weights.save_async)
            time_rollout_ckpt = time.time() - t0

            time_shardcast = time.time()
            if world.rank == 0:
                if envs.SHARDCAST_OUTPUT_DIR is not None:
                    logger.info(f"Broadcasting {model_path}")
                    shardcast.broadcast(model_path)  # TODO: Is this blocking?
            time_shardcast = time.time() - time_shardcast

            if len(weight_checkpoint_paths) > config.max_async_level:
                path_to_delete = weight_checkpoint_paths.pop(0)
                ckpt_step = int(str(path_to_delete).split("_")[-1])

                should_keep = config.weights.interval and ckpt_step % config.weights.interval == 0
                if path_to_delete.exists() and not should_keep:
                    logger.info(f"Removing past weight checkpoint at {path_to_delete}")
                    shutil.rmtree(path_to_delete, ignore_errors=True)

        if config.memory_profile and (progress.step == 2) and world.rank == 0:
            logger.info("Dumping memory snapshot.")
            pickle_path: str = config.memory_profile
            if not pickle_path.endswith(".pickle"):
                pickle_path += ".pickle"
            torch.cuda.memory._dump_snapshot(pickle_path)
            torch.cuda.memory._record_memory_history(enabled=False)

        if config.ckpt.interval is not None and progress.step % config.ckpt.interval == 0:
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_full_checkpoint(model, [optimizer], progress, config.ckpt.path)

        if config.recompute_logprobs:
            reshard_module(model_for_logprob_only)
            tensor_offloaded_repository[progress.step] = copy_model_to_cpu(model)

        train_step_time = time.time() - train_step_start_time
        logger.success(f"Finished training step {progress.step} in {train_step_time:.2f}s")
        if world.rank == 0:
            time_metrics = {
                "step": progress.step,
                "perf/time_train_step": train_step_time,
                "perf/time_data_loading": load_data_time,
                "perf/time_logprob": compute_logprobs_time,
            }
            if time_rollout_ckpt is not None:
                time_metrics["perf/time_rollout_ckpt"] = time_rollout_ckpt
            if time_shardcast is not None:
                time_metrics["perf/time_shardcast"] = time_shardcast
            if time_rollout_delete is not None:
                time_metrics["perf/time_rollout_delete"] = time_rollout_delete

            monitor.log(time_metrics)

        if config.max_steps and progress.step >= config.max_steps:
            break

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("Training finished!")


def main():
    train(parse_argv(TrainingConfig))


if __name__ == "__main__":
    main()
