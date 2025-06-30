import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import shardcast
import torch
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
from zeroband.training.metrics import BatchMetrics
from zeroband.training.utils import (
    OffloadedTensor,
    copy_model_to_cpu,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.training.world import World, get_world
from zeroband.utils.models import ModelType, get_model_and_tokenizer
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


def get_local_batch_size(batch_size: int, micro_bs: int, world: World) -> int:
    assert batch_size % world.world_size == 0
    batch_size = batch_size // world.world_size

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
    world = get_world()

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

    # Optionally, apply activation checkpointing
    if config.ac:
        setup_ac(model, config.ac)

    # Shard the model for training using FSDP
    apply_fsdp(model, config.reshard_after_forward)

    # Optionally, compile the model
    if config.model.compile:
        model = torch.compile(model)

    # Optionally, initialize a model to compute logprobs
    if config.recompute_logprobs:
        logprob_model, _ = get_model_and_tokenizer(config.model.name, config.model.attn)
        apply_fsdp(logprob_model, config.reshard_after_forward)

        # Offload the logprob model to CPU
        tensor_offloaded_repository: dict[int, OffloadedTensor] = {}
        tensor_offloaded_repository[0] = offload_model_to_cpu(logprob_model)

        if config.model.compile:
            logprob_model: ModelType = torch.compile(logprob_model)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=(config.optim.betas1, config.optim.betas2),
    )

    # TODO(Mika): Add this back but without dependency on the seq_len
    # perf_counter = PerfCounter(window_size=10, model=model, seq_len=config.data.seq_length)

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

    logger.info("Starting training loop")
    weight_checkpoint_paths = []
    while True:
        train_step_start_time = time.time()
        logger.info(f"Starting training step {progress.step}")

        # Load the training batch
        load_data_start_time = time.time()
        micro_batches = train_dataloader.get_batch()
        load_data_time = time.time() - load_data_start_time
        logger.info(f"Loaded batch in {load_data_time:.2f} seconds")

        # Optionally, Compute the logprobs for the training batch
        if config.recompute_logprobs:
            logger.info(f"Starting recomputing logprobs for step {progress.step}")
            compute_logprobs_start_time = time.time()
            og_infer_step = progress.step - config.max_async_level
            infer_step = max(og_infer_step, 0)

            # Wake up the logprob model from CPU
            wake_up_model_from_cpu(logprob_model, tensor_offloaded_repository[infer_step])
            if og_infer_step == infer_step:
                del tensor_offloaded_repository[infer_step]

            with torch.no_grad():
                num_micro_batches = len(micro_batches)
                for micro_step, micro_batch in enumerate(micro_batches, start=1):
                    logger.debug(f"Computing logprobs for micro batch {micro_step} / {num_micro_batches}")
                    input_ids = micro_batch["token_ids"].to("cuda")
                    position_ids = micro_batch["position_ids"].to("cuda")
                    temperature = micro_batch["temperature"]

                    logprobs = get_logprobs(logprob_model, input_ids, position_ids, temperature)
                    micro_batch["logprobs"] = logprobs.to("cpu")

            # here we sepcifically don't save the tensor offloaded, they are alreay consumed and we will never use it again.
            # this avoid having to make sure we don't keep too much tensor offloaded in cpu memory
            reshard_module(logprob_model)
            offload_model_to_cpu(logprob_model)

            compute_logprobs_time = time.time() - compute_logprobs_start_time
            logger.info(f"Computed logprobs in {compute_logprobs_time:.2f} seconds")

        if config.memory_profile and world.rank == 0:
            torch.cuda.memory._record_memory_history()

        batch_metrics = BatchMetrics()
        num_micro_batches = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches, start=1):
            logger.debug(f"Training on micro batch {micro_step} / {num_micro_batches}")
            input_ids = micro_batch["token_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = (
                torch.ones_like(input_ids).int().to("cuda")
            )  # TODO(Mika): Remove this from loss computation, then here
            logprobs = micro_batch["logprobs"].to("cuda")
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
                entropy = entropy_loss(logits, loss_mask, temperature, max_tokens)

            # Now we can delete the micro batch CUDA tensors
            del logits, input_ids, position_ids, advantages, loss_mask, logprobs

            # Scale the loss by the number of micro batches (=gradient accumulation steps)
            loss = loss / num_micro_batches

            # Backward pass (ensures loss reduction across FSDP ranks)
            loss.backward()

            batch_metrics.update("loss/loss", loss.detach().clone())
            batch_metrics.update("loss/entropy", entropy.detach().clone())
            batch_metrics.update("loss/clip_ratio", clip_ratio.detach().clone())

            del loss, entropy, clip_ratio

        # Synchronize the batch metrics across all ranks
        batch_metrics.sync()

        # Optionally, clip the gradients
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
            "train/inner_lr": inner_lr,
            "losses/grad_norm": grad_norm.item(),
        }

        for key, value in batch_metrics.items():
            metrics[key] = value.item()

        log = f"Step: {progress.step}, loss: {batch_metrics['loss/loss'].item():.4f}, "

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
            reshard_module(logprob_model)
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
