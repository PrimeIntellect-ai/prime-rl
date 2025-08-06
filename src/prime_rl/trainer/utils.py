from itertools import chain
from typing import Any, Literal, TypeAlias

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


ReduceType: TypeAlias = Literal["min", "max", "mean", "sum", "numel"]


class TensorMetrics(dict):
    """A class to aggregate tensor statistics across multiple steps. Only support synchronizing once."""

    REDUCE_TYPES: list[ReduceType] = ["min", "max", "mean", "sum", "numel"]

    def __init__(self, *args, **kwargs):
        assert dist.is_initialized(), "TensorMetrics requires a distributed environment"
        super().__init__(*args, **kwargs)
        self.stored_keys: dict[str, list[ReduceType]] = {}  # key -> list[ReduceType]
        self.synced = False

    def update(self, key: str, value: Tensor, reduce_types: list[ReduceType] = REDUCE_TYPES) -> None:
        """Accumulate relevant statistics for specified reduction types of a tensor"""
        if key not in self.stored_keys:
            self.stored_keys[key] = reduce_types
        else:
            assert self.stored_keys[key] == reduce_types, (
                f"Key {key} has already been updated with different reduce types. "
                f"Previous: {self.stored_keys[key]}, new: {reduce_types}"
            )
        self[f"{key}/min"] = min(self.get(f"{key}/min", float("inf")), value.min().item())
        self[f"{key}/max"] = max(self.get(f"{key}/max", float("-inf")), value.max().item())
        self[f"{key}/sum"] = self.get(f"{key}/sum", 0.0) + value.sum().item()
        self[f"{key}/numel"] = self.get(f"{key}/numel", 0) + value.numel()

    def sync(self) -> None:
        """Synchronize the specified statistics of all stored keys across ranks."""
        assert not self.synced, (
            "TensorMetrics has already been synced. Syncing again is not supported. It is recommended to re-initialize the object across steps."
        )

        def _reduce_value(value: float | int, op: dist.ReduceOp) -> float | int:
            assert isinstance(value, float) or isinstance(value, int), (
                f"Expected float or int, got {type(value)} for key {key}"
            )
            tensor_value = torch.tensor(value).to("cuda")
            dist.all_reduce(tensor_value, op=op)
            return tensor_value.item()

        for key in self.stored_keys.keys():
            # Min/max/sum/numel reduction
            self[f"{key}/min"] = _reduce_value(self[f"{key}/min"], dist.ReduceOp.MIN)
            self[f"{key}/max"] = _reduce_value(self[f"{key}/max"], dist.ReduceOp.MAX)
            self[f"{key}/sum"] = _reduce_value(self[f"{key}/sum"], dist.ReduceOp.SUM)
            self[f"{key}/numel"] = _reduce_value(self[f"{key}/numel"], dist.ReduceOp.SUM)

            # Mean reduction
            reduce_types = self.stored_keys[key]
            if "mean" in reduce_types:
                assert f"{key}/sum" in self and f"{key}/numel" in self
                self[f"{key}/mean"] = self[f"{key}/sum"] / self[f"{key}/numel"]

            # Delete all keys that are not in the reduce types
            for reduce_type in self.REDUCE_TYPES:
                if reduce_type not in reduce_types and f"{key}/{reduce_type}" in self:
                    del self[f"{key}/{reduce_type}"]

        self.synced = True
