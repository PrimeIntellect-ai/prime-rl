import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy  # type: ignore
import wandb

from zeroband.models import AttnImpl, ModelName, ModelType, get_model_and_tokenizer
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.data import DataConfig, get_dataloader
from zeroband.training.loss import grpo_loss
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import PerfCounter, apply_ac_ckpt

from zeroband.logger import get_logger

from pydantic_config import BaseConfig, parse_argv
from jaxtyping import Float

from zeroband.training.world_info import WorldInfo, get_world_info

from pydantic import model_validator

from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
import logging


class AdamConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    lr: float = 4e-4
    weight_decay: float = 0.01
    betas1: float = 0.9
    betas2: float = 0.99


class OptimConfig(BaseConfig):
    optim: AdamConfig = AdamConfig()
    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512

    step_per_rollout: int = 1


class TrainConfig(BaseConfig):
    micro_bs: int = 1
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard
    memory_profile: str | None = None
    torch_compile: bool = True
    liger_qwen: bool = False

    attn_impl: AttnImpl = "flex_attention"


class CkptConfig(BaseConfig):
    path: str | None = None
    interval: int | None = None
    resume: str | None = None

    rollout_path: str | None = None  # if rollout path is set we saved at each step


class Config(BaseConfig):
    name_model: ModelName = "150M"

    ckpt: CkptConfig = CkptConfig()

    project: str = "prime_simple"
    wandb: bool = True

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig

    gpus_ids: list[int] | None = None

    temperature: float = 0.6  # todo remove this and add this to the data

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.name_model, "train.liger_qwen can only be applied to Qwen2 models."
        return self


def get_gradient_accumulation_steps(batch_size: int, micro_bs: int, data_workers: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    print(f"batch_size: {batch_size}, micro_bs: {micro_bs}, data_workers: {data_workers}")
    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size // micro_bs


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=None)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank

    if world_info.local_rank >= len(gpus_ids):
        raise ValueError(f"Local rank {world_info.local_rank} is greater than the number of available GPUs ({len(gpus_ids)})")

    return gpus_ids[world_info.local_rank]


def train(config: Config):
    if "ZERO_BAND_DEV" not in os.environ:
        torch._logging.set_logs(dynamo=logging.CRITICAL)  # silent flex attn error
        torch_log.setLevel(logging.CRITICAL)  #

    logger = get_logger()
    world_info = get_world_info()

    logger.info(f"start training on {world_info.world_size}")

    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    # batch_size is the total batch size for all GPUs

    gradient_accumulation_steps = get_gradient_accumulation_steps(
        config.optim.batch_size, config.train.micro_bs, config.data.num_workers, world_info
    )

    model, tokenizer = get_model_and_tokenizer(config.name_model, config.train.attn_impl)

    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        micro_batch_size=config.train.micro_bs,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
    )

    train_dataloader_iterator = iter(train_dataloader)

    if config.train.liger_qwen:
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            model=model,
        )

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    apply_fsdp(model, config.train.reshard_after_forward)

    optimizer = torch.optim.AdamW(params=model.parameters(),lr=config.optim.optim.lr,weight_decay=config.optim.optim.weight_decay,betas=(config.optim.optim.betas1, config.optim.optim.betas2))  # fmt: skip

    scheduler = get_scheduler(sched_type=config.optim.sched_type,optimizer=optimizer,num_warmup_steps=config.optim.warmup_steps,num_stable_steps=config.optim.stable_steps,num_training_steps=config.optim.total_steps)  # fmt: skip

    training_progress = TrainingProgress(total_tokens=0, step=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump())

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

    if config.ckpt.resume:
        load_checkpoint_fsdp_state(model, [optimizer], training_progress, train_dataloader, scheduler, config.ckpt.resume)

    perf_counter = PerfCounter(window_size=10, model=model, seq_len=config.data.seq_length)

    previous_ckpt_rollout = []

    while True:
        loss_batch = 0
        average_rewards = 0
        clip_ratio_batch = 0
        seq_lens_batch = 0

        if config.train.memory_profile and world_info.rank == 0:
            torch.cuda.memory._record_memory_history()

        for grad_acc_step in range(gradient_accumulation_steps):
            is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
            model.set_requires_gradient_sync(not is_accumulating)  # no sync if we are accumulating gradients

            # Load args
            batch = next(train_dataloader_iterator)
            input_ids = batch["input_ids"].to("cuda")
            loss_mask = batch["loss_mask"].bool()
            average_rewards += batch["rewards"][loss_mask].mean() / gradient_accumulation_steps
            seq_lens_batch += batch["seq_lens"].float().mean() / gradient_accumulation_steps

            # Forward
            logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids).logits.contiguous()

            # Gather args for grpo loss
            advantages = batch["advantages"].to("cuda")
            loss_mask = loss_mask.to("cuda")
            original_logprobs = batch["logprobs"].to("cuda")
            # Loss
            loss, clip_ratio = grpo_loss(logits, input_ids, advantages, original_logprobs, loss_mask, config.temperature)
            loss = loss / gradient_accumulation_steps
            clip_ratio = clip_ratio / gradient_accumulation_steps

            del batch, logits, input_ids, advantages, loss_mask, original_logprobs

            # Backward
            loss.backward()
            loss_batch += loss.detach().clone()
            clip_ratio_batch += clip_ratio.detach().clone()
            del loss, clip_ratio

        dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)
        dist.all_reduce(tensor=clip_ratio_batch, op=dist.ReduceOp.AVG)

        average_rewards = average_rewards / world_info.world_size
        dist.all_reduce(tensor=average_rewards, op=dist.ReduceOp.SUM)  # need to use gloo here so not AVG

        seq_lens_batch = seq_lens_batch / world_info.world_size
        dist.all_reduce(tensor=seq_lens_batch, op=dist.ReduceOp.SUM)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).full_tensor()  # type: ignore (is a dtensor)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        # logging
        training_progress.step += 1
        inner_lr = [group["lr"] for group in optimizer.param_groups][0]

        # syncing loss across all data parallel rank within a nodes
        new_tokens = config.data.seq_length * config.optim.batch_size
        perf_counter.count_tokens(new_tokens)
        training_progress.total_tokens += new_tokens

        padding_proportion = (config.data.seq_length - seq_lens_batch.item() - 1) / config.data.seq_length

        metrics = {
            "Loss": loss_batch.item(),
            "step": training_progress.step,
            "rollout_step": training_progress.step // config.optim.step_per_rollout,
            "seq_lens": seq_lens_batch.item(),
            "inner_lr": inner_lr,
            "Perplexity": torch.exp(loss_batch).item(),
            "total_tokens": training_progress.total_tokens,
            "time": time.time(),
            "grad_norm": grad_norm.item(),
            "average_rewards": average_rewards.item(),
            "clip_ratio": clip_ratio_batch.item(),
            "padding_proportion": padding_proportion,
        }

        log = f"step: {training_progress.step}, rollout_step: {training_progress.step // config.optim.step_per_rollout}, loss: {loss_batch.item():.4f}, average_rewards: {average_rewards.item():.4f}"

        del loss_batch, average_rewards, grad_norm

        tokens_per_second = perf_counter.get_tokens_per_second()
        if tokens_per_second is not None:
            metrics["tokens_per_second"] = tokens_per_second
            metrics["mfu"] = perf_counter.get_mfu()
            log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

        if world_info.rank == 0 and config.wandb:
            wandb.log(metrics)

        logger.info(log)

        if config.train.memory_profile and (training_progress.step == 2) and world_info.rank == 0:
            logger.info("Dumping memory snapshot.")
            pickle_path: str = config.train.memory_profile
            if not pickle_path.endswith(".pickle"):
                pickle_path += ".pickle"
            torch.cuda.memory._dump_snapshot(pickle_path)
            torch.cuda.memory._record_memory_history(enabled=False)

        if config.ckpt.interval is not None and training_progress.step % config.ckpt.interval == 0:
            save_checkpoint_fsdp_state(model, [optimizer], training_progress, train_dataloader, scheduler, config.ckpt.path)

        if config.ckpt.rollout_path is not None and training_progress.step % config.optim.step_per_rollout == 0:
            rollout_step = training_progress.step // config.optim.step_per_rollout
            path = Path(config.ckpt.rollout_path) / f"step_{rollout_step}"
            previous_ckpt_rollout.append(path)
            save_ckpt_for_rollout(model, path)

            if len(previous_ckpt_rollout) > 2:
                path_to_delete = previous_ckpt_rollout.pop(0)
                if path_to_delete.exists():
                    logger.info(f"Removing past rollout ckpt at {path_to_delete}")
                    shutil.rmtree(path_to_delete, ignore_errors=True)

        if training_progress.step >= config.optim.total_steps:
            break

    if prefetcher is not None:
        prefetcher.shutdown()

    logger.info("Training finished, exiting ...")
    logger.info(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    train(Config(**parse_argv()))
