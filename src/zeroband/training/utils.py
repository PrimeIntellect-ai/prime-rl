import socket
from itertools import chain
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
import wandb
from torch.distributed.tensor import DTensor
from transformers import (
    PreTrainedTokenizer,
)

from zeroband.training.model import Model


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def get_real_tensor(tensor: torch.Tensor | DTensor):
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


OffloadedTensor: TypeAlias = list[tuple[torch.Tensor, int]]


def offload_model_to_cpu(model: Model) -> OffloadedTensor:
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


def copy_model_to_cpu(model: Model) -> OffloadedTensor:
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


def wake_up_model_from_cpu(model: Model, tensors: OffloadedTensor):
    for param, (cpu_data, storage_size) in zip(chain(model.parameters(), model.buffers()), tensors):
        data = get_real_tensor(param.data)
        data.untyped_storage().resize_(storage_size)
        data.copy_(cpu_data, non_blocking=True)
    torch.cuda.synchronize()


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
            table_data = {
                "step": sample_history["step"],
                "prompt": sample_history["prompt"],
                "completion": sample_history["completion"],
            }

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
