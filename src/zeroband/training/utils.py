import socket
import time
from itertools import chain
from typing import Any, TypeAlias

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.tensor import DTensor
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizer,
)

from zeroband.training.world_info import get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.models import ModelType


def apply_ac_ckpt(model: ModelType, num: int):
    """Apply activation checkpointing to the model.
    Apply to layers multiple of `num`.

    Example if `num=2` only half of the layers are checkpointed.
    """
    logger = get_logger("TRAIN")

    layers_ckpt = 0

    for layer_id, transformer_block in model.model.layers.named_children():
        if layers_ckpt % num == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
            model.model.layers.register_module(layer_id, transformer_block)
            layers_ckpt += 1

    logger.debug(f"Applied activation checkpointing to {layers_ckpt} layers")


### code above inspired and copied from https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119


# hardcoded BF16 type peak flops for NVIDIA A100 and H100 GPU
def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name or "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        return 1979e12  # sxm  https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200
    else:  # for other GPU types, assume A100
        return 312e12


def get_num_flop_per_token(num_params: int, model_config: LlamaConfig, seq_len: int) -> int:
    l, h, q, t = (  # noqa: E741
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.hidden_size // model_config.num_attention_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def get_num_params(model: ModelType, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.lm_head.weight.numel()
    return num_params


class PerfCounter:
    """A class to count tokens per second with a rolling window.
    we use a rollowing window because time perf counter is not precise enough in some case
    """

    def __init__(self, window_size: int, model: LlamaForCausalLM, seq_len: int):
        self.window_size = window_size
        self.tokens = []
        self.times = []
        self.model = model

        self.gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
        self.num_params = get_num_params(model, exclude_embedding=True)
        self.num_flop_per_token = get_num_flop_per_token(self.num_params, model.config, seq_len=seq_len)

        self._world_info = get_world_info()

    def count_tokens(self, tokens: int):
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_second(self) -> float | None:
        if len(self.tokens) < 2:
            return None
        return sum(self.tokens[1:]) / (self.times[-1] - self.times[0])

    def get_mfu(self) -> float | None:
        tokens_per_second = self.get_tokens_per_second()
        if tokens_per_second is None:
            return None
        return 100 * self.num_flop_per_token * tokens_per_second / self.gpu_peak_flops / self._world_info.world_size


def get_random_available_port_list(num_port):
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    ports = []

    while len(ports) < num_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            new_port = s.getsockname()[1]

        if new_port not in ports:
            ports.append(new_port)

    return ports


def get_random_available_port():
    return get_random_available_port_list(1)[0]


class FakeTokenizer(object):
    def __init__(self):
        self.vocab_size = 1000
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def __len__(self):
        return self.vocab_size


class MetricsAverager:
    """
    Simple class that keep track of metrics over multiple gradient accumulation steps and sync across gpu when calling sync()
    """

    def __init__(self):
        self.metrics = {}
        self.count = {}
        self.world_info = get_world_info()

    @torch.no_grad()
    def update(self, key, value: torch.Tensor | list[torch.Tensor]):
        if isinstance(value, torch.Tensor):
            self._update(key, value)
        else:
            for v in value:
                self._update(key, v)

    def _update(self, key, value: torch.Tensor):
        if key not in self.metrics:
            self.metrics[key] = value
            self.count[key] = 1
        else:
            self.metrics[key] += value
            self.count[key] += 1

    @torch.no_grad()
    def sync(self):
        for key in self.metrics:
            value = self.metrics[key].clone()
            count = torch.tensor(self.count[key])

            dist.all_reduce(value.to("cuda"), op=dist.ReduceOp.SUM)
            dist.all_reduce(count.to("cuda"), op=dist.ReduceOp.SUM)

            value = value / count

            self.metrics[key] = value

    def __getitem__(self, key):
        return self.metrics[key]

    def items(self):
        return self.metrics.items()


def get_real_tensor(tensor: torch.Tensor | DTensor):
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


OffloadedTensor: TypeAlias = list[tuple[torch.Tensor, int]]


def offload_model_to_cpu(model: ModelType) -> OffloadedTensor:
    """
    Retun a list of cpu tensor representing the model weight.
    Also reduce to 0 the gpu memory usage.
    """
    tensors_offloaded = []
    for param in chain(model.parameters(), model.buffers()):
        data = get_real_tensor(param.data)
        cpu_data = data.to("cpu", non_blocking=True)
        storage_size = data.untyped_storage().size()
        data.untyped_storage().resize_(1)
        tensors_offloaded.append((cpu_data, storage_size))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return tensors_offloaded


def copy_model_to_cpu(model: ModelType) -> OffloadedTensor:
    """
    Retun a list of cpu tensor representing the model weight.
    Keep gpu memory intact.
    """

    tensors_offloaded = []
    for param in chain(model.parameters(), model.buffers()):
        data = get_real_tensor(param.data)
        cpu_data = data.to("cpu")
        storage_size = data.untyped_storage().size()
        tensors_offloaded.append((cpu_data, storage_size))

    return tensors_offloaded


def wake_up_model_from_cpu(model: ModelType, tensors: OffloadedTensor):
    for param, (cpu_data, storage_size) in zip(chain(model.parameters(), model.buffers()), tensors):
        data = get_real_tensor(param.data)
        data.untyped_storage().resize_(storage_size)
        data.copy_(cpu_data, non_blocking=True)
    torch.cuda.synchronize()


def reshard_module(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def log_to_wandb(
    metrics_dict: dict[str, Any],
    tokenizer: PreTrainedTokenizer | None = None,
    batch: dict[str, torch.Tensor] | None = None,
    step: int | None = None,
) -> None:
    """Log our metrics to wandb, grouped by type."""
    # Initialize dictionary
    wandb_metrics = {}

    categories = {
        "train/": ["step", "rollout_step", "inner_lr", "total_tokens", "total_samples"],
        "losses/": ["Loss", "pg_loss", "entropy_loss", "kl", "grad_norm", "clip_ratio"],
        "rewards/": ["sample_reward", "task_reward", "batch_reward", "batch_task_reward"],
        "lengths/": [
            "seq_lens",
            "batch_seq_lens",
            "target_lengths",
            "batch_target_lengths",
            "padding_proportion",
        ],
        "perf/": [
            "tokens_per_second",
            "tokens_per_second_per_gpu",
            "mfu",
            "time_rollout_step",
            "time_logprob",
            "time_data_loading",
            "time_packing",
        ],
        "task_rewards/": [],
    }

    # Add metrics to respective groups
    for prefix, keys in categories.items():
        for k in keys:
            if k in metrics_dict and metrics_dict[k] is not None:
                wandb_metrics[f"{prefix}{k}"] = metrics_dict[k]

    # Handle task-specific metrics based on the prefix
    for key, value in metrics_dict.items():
        if key.startswith("individual_task_"):
            wandb_metrics[f"task_rewards/{key.replace('individual_task_', '')}"] = value

    # Log everything
    wandb.log(wandb_metrics, step=metrics_dict.get("step", step))


def log_prompt_response_samples(
    tokenizer: PreTrainedTokenizer, batch: dict[str, torch.Tensor], step: int, sample_history: dict | None = None
) -> dict:
    """Log samples using wandb.Table with accumulated history.
    Only logs every 5 steps to minimize overhead.

    Args:
        tokenizer: The tokenizer to decode tokens
        batch: The current batch of data
        step: The current training step
        sample_history: Optional dict to store sample history between calls
    """
    if sample_history is None:
        sample_history = {
            "step": [],
            "prompt": [],
            "completion": [],
            "rewards": [] if "rewards" in batch else None,
            "task_rewards": [] if "task_rewards" in batch else None,
            "last_logged_step": -5,  # Initialize to trigger first logging
        }

    try:
        batch_size = batch["input_ids"].size(0)
        for i in range(batch_size):
            # Find completion start from the loss mask
            tokens = batch["input_ids"][i].cpu().tolist()
            mask = batch["loss_mask"][i].cpu().tolist()

            try:
                response_start = mask.index(1)
            except ValueError:
                response_start = len(tokens) // 3

            prompt = tokenizer.decode(tokens[:response_start], skip_special_tokens=True)
            completion = tokenizer.decode(tokens[response_start:], skip_special_tokens=True)

            sample_history["step"].append(str(step))
            sample_history["prompt"].append(prompt)
            sample_history["completion"].append(completion)

            if "rewards" in batch and sample_history["rewards"] is not None:
                sample_history["rewards"].append(float(batch["rewards"][i].item()))
            if "task_rewards" in batch and sample_history["task_rewards"] is not None:
                sample_history["task_rewards"].append(float(batch["task_rewards"][i].item()))

        if step >= sample_history["last_logged_step"] + 5:
            # Create table data dictionary (we are forced to remake it each time)
            table_data = {"step": sample_history["step"], "prompt": sample_history["prompt"], "completion": sample_history["completion"]}

            if sample_history["rewards"] is not None:
                table_data["reward"] = sample_history["rewards"]
            if sample_history["task_rewards"] is not None:
                table_data["task_reward"] = sample_history["task_rewards"]

            df = pd.DataFrame(table_data)
            table = wandb.Table(dataframe=df)
            wandb.log({"completions": table}, step=step)
            sample_history["last_logged_step"] = step

        return sample_history

    except Exception as e:
        print(f"Error logging table: {e}")
        return sample_history
