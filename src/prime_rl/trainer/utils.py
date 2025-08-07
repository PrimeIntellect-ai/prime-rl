from collections import defaultdict
from itertools import chain
from typing import Any, TypeAlias

import pandas as pd
import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.trainer.model import Model
from prime_rl.utils.utils import format_num, format_time


def get_real_tensor(tensor: Tensor | DTensor) -> Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


OffloadedTensor: TypeAlias = list[tuple[Tensor, int]]


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


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted values for the
    training throughput and overall step time. First first N rows show the
    per-step values, and the last row shows the mean, std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "perf/train/throughput": "Throughput",
        "time/train": "Step Time",
    }
    df = df[columns.keys()].rename(columns=columns)
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Throughput"] = df["Throughput"].apply(format_num, precision=2)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    table.add_row(*([""] * len(row)))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)


def flexible_all_gather(input_tensor: Tensor) -> Tensor:
    """
    All gather a tensor between all ranks. All ranks does not need to have the same number of elements.
    return type is a concatanation of all the tensors from all ranks without padded elements.
    """

    assert len(input_tensor.shape) == 1, "input_tensor must be a 1D tensor"

    local_numel = torch.tensor(input_tensor.shape[0], device=input_tensor.device)
    all_numels = [torch.tensor(0, device=input_tensor.device)] * dist.get_world_size()
    dist.all_gather(all_numels, local_numel)
    all_numels = [numel.item() for numel in all_numels]
    max_numel = max(all_numels)

    if local_numel < max_numel:
        inputs_tensor_padded = torch.cat(
            [input_tensor, torch.zeros(max_numel - local_numel, dtype=input_tensor.dtype, device=input_tensor.device)]
        )
    else:
        inputs_tensor_padded = input_tensor

    all_input_tensors = [
        torch.zeros(max_numel, dtype=input_tensor.dtype, device=input_tensor.device)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_input_tensors, inputs_tensor_padded)

    all_non_padded_input_tensors = [all_input_tensors[i][: all_numels[i]] for i in range(dist.get_world_size())]
    return torch.cat(all_non_padded_input_tensors, dim=0)


class TensorMetrics:
    """
    This class acunulate tensor across multiple steps (example grad accumulation steps).

    It then gather the all the tensor across the steps and rank and log distribution metrics including:
    - mean
    - std
    - min
    - max
    - median
    - 1%, 5%, 10%, 90%, 95%, 99% quantiles

    TODO: decide on adding wandb histogram logging
    """

    def __init__(self):
        self.all_tensors = defaultdict(list)

    def update(self, key: str, value: Tensor) -> None:
        self.all_tensors[key].append(value)

    def sync(self) -> dict[str, Any]:
        metrics = {}
        for k, v in self.all_tensors.items():
            # concatenate all the tensors across the steps and rank and move to cuda for nccl communication
            tensor = torch.cat(v, dim=0).to("cuda")
            tensor = flexible_all_gather(tensor)

            metrics[f"{k}/mean"] = tensor.mean().item()
            metrics[f"{k}/std"] = tensor.std().item()
            metrics[f"{k}/min"] = tensor.min().item()
            metrics[f"{k}/max"] = tensor.max().item()
            metrics[f"{k}/median"] = torch.median(tensor).item()
            metrics[f"{k}/1%"] = torch.quantile(tensor, 0.01).item()
            metrics[f"{k}/5%"] = torch.quantile(tensor, 0.05).item()
            metrics[f"{k}/10%"] = torch.quantile(tensor, 0.1).item()
            metrics[f"{k}/90%"] = torch.quantile(tensor, 0.9).item()
            metrics[f"{k}/95%"] = torch.quantile(tensor, 0.95).item()
            metrics[f"{k}/99%"] = torch.quantile(tensor, 0.99).item()

        return metrics
