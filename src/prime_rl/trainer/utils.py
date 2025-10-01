import pickle
import time
from collections import defaultdict
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import torch
import torch.distributed as dist
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from rich.text import Text
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num, format_time

DEFAULT_TIMEOUT = timedelta(seconds=600)


def setup_torch_distributed(timeout: timedelta = DEFAULT_TIMEOUT):
    torch.cuda.set_device(get_world().local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", torch.cuda.current_device()),
        timeout=timeout,
    )


def get_response_lengths(position_ids: torch.Tensor) -> list[int]:
    """
    Compute lengths of concatenated sequences from position_ids.

    Each sequence starts at 0 and increments. When position_ids resets to 0,
    it indicates the start of a new sequence. Trailing zeros (padding) are
    counted as part of the last sequence.

    Args:
        position_ids: Tensor of shape [total_seqlen]

    Returns:
        List of sequence lengths
    """
    position_ids = position_ids.flatten()

    boundaries = [0]  # Start of first sequence

    for i in range(1, len(position_ids)):
        if position_ids[i] == 0 and position_ids[i - 1] != 0:
            # This is a potential sequence boundary (0 after non-zero)
            # But only if the next element is 1 (indicating a new incrementing sequence)
            # Otherwise, this 0 is padding and belongs to current sequence
            if i + 1 < len(position_ids) and position_ids[i + 1] == 1:
                boundaries.append(i)

    # Calculate lengths based on boundaries
    lengths = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(position_ids)
        lengths.append(end - start)

    return lengths


def get_real_tensor(tensor: Tensor | DTensor) -> Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


OffloadedTensor: TypeAlias = list[tuple[Tensor, int]]


def offload_model_to_cpu(model: nn.Module) -> OffloadedTensor:
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


def copy_model_to_cpu(model: nn.Module) -> OffloadedTensor:
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


def wake_up_model_from_cpu(model: nn.Module, tensors: OffloadedTensor):
    for param, (cpu_data, storage_size) in zip(chain(model.parameters(), model.buffers()), tensors):
        data = get_real_tensor(param.data)
        data.untyped_storage().resize_(storage_size)
        data.copy_(cpu_data, non_blocking=True)
    torch.cuda.synchronize()


def print_sample(input_ids: list[int], loss_mask: list[bool], tokenizer: PreTrainedTokenizer):
    """
    Visualize the loss mask of a tokenized sample using rich.
    Reference: https://huggingface.co/Qwen/Qwen3-8B/discussions/14
    """
    text = Text()
    for token, mask in zip(tokenizer.convert_ids_to_tokens(input_ids), loss_mask):
        text.append(token.replace("Ġ", " ").replace("Ċ", "\n"), style="cyan" if mask else "white")
    rich_print(text)


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
        "perf/mfu": "MFU",
        "perf/throughput": "Throughput",
        "time/step": "Step Time",
        "perf/peak_memory": "Peak Memory",
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
    formatted_df["MFU"] = df["MFU"].apply(lambda x: f"{format_num(x, precision=2)}%")
    formatted_df["Throughput"] = df["Throughput"].apply(lambda x: format_num(x, precision=2))
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Peak Memory"] = df["Peak Memory"].apply(lambda x: f"{format_num(x, precision=1)} GiB")
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    table.add_row(*([""] * len(formatted_df.columns)))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame()
    formatted_mean_df["MFU"] = mean_df["MFU"].apply(lambda x: f"{format_num(x, precision=2)}%")
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    mean_row = (
        ["Overall"]
        + formatted_mean_df.T.apply(
            lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
        ).tolist()
        + [
            f"{format_num(mean_df['Peak Memory']['mean'], precision=1)} GiB ({mean_df['Peak Memory']['mean'] / (torch.cuda.mem_get_info()[1] / 1024**3) * 100:.1f}%)"
        ]
    )
    table.add_row(*mean_row)

    # Display table
    console.print(table)


def flexible_all_gather(tensor: Tensor) -> Tensor:
    """
    All-gather a 1D tensor between all ranks, with potentially different numbr of element per rank.
    Returns a tensor of shape (world_size * max_numel, dtype=tensor.dtype, device=tensor.device)
    """

    assert tensor.ndim == 1, "Can only flexibly all-gather 1D tensors"

    if dist.get_world_size() == 1:
        return tensor

    # Find the tensor with the most elements
    local_numel = tensor.numel()
    local_numel_tensor = torch.tensor(local_numel, device=tensor.device)
    all_numel_tensors = [torch.tensor(0, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(all_numel_tensors, local_numel_tensor)
    all_numels = [numel.item() for numel in all_numel_tensors]
    max_numel = int(max(all_numels))

    # Pad the tensor with zeros if it has less elements than the maximum
    if local_numel < max_numel:
        tensor = torch.cat([tensor, torch.zeros(max_numel - local_numel, dtype=tensor.dtype, device=tensor.device)])

    # All-gather the tensors
    all_tensors = [
        torch.zeros(max_numel, dtype=tensor.dtype, device=tensor.device) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_tensors, tensor)
    all_tensors_unpadded = torch.cat([tensor[:numel] for tensor, numel in zip(all_tensors, all_numels)])

    return all_tensors_unpadded


class Tensors(defaultdict):
    """A class to accumulate tensors and compute statistics (mean, median, std, min, max) across multiple steps and ranks."""

    def __init__(self):
        assert dist.is_initialized(), "Tensors requires a distributed environment"
        super().__init__(list)

    def compute_stats(self) -> dict[str, float | int]:
        """Synchronize the tensor statistic across all ranks for each key and compute relevant statistics."""

        metrics = {}
        for key in list(self.keys()):
            # All-gather tensors across steps and ranks (get global distribution)
            tensors = torch.cat(self.pop(key), dim=0).to("cuda")
            assert tensors.ndim == 1, "Can only aggregate 1D tensors"
            tensors = flexible_all_gather(tensors)
            assert tensors.ndim == 1, "Can only aggregate 1D tensors"

            # Compute relevant tensor statistics
            metrics[f"{key}/mean"] = tensors.mean().item()
            metrics[f"{key}/median"] = torch.median(tensors).item()
            metrics[f"{key}/std"] = tensors.std().item()
            metrics[f"{key}/min"] = tensors.min().item()
            metrics[f"{key}/max"] = tensors.max().item()

            # Add back all-gathered tensors to self
            self[key].append(tensors.tolist())

        return metrics


MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


class MemoryProfiler:
    def __init__(self, step_num: int, snapshot_path: Path):
        torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
        self.logger = get_logger()
        snapshot_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = snapshot_path
        self.step_num = step_num

    def step(self):
        self.logger.info(f"Dumping memory snapshot at step {self.step_num} at {self.snapshot_path}")
        begin = time.monotonic()
        step_folder = self.snapshot_path / f"step_{self.step_num}"
        step_folder.mkdir(parents=True, exist_ok=True)
        file_path = step_folder / f"rank_{get_world().rank}.pickle"
        with open(file_path, "wb") as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)
        self.logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds, load {file_path} at https://docs.pytorch.org/memory_viz to visualize the memory usage"
        )
        self.step_num += 1
