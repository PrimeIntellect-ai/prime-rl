import gc
import pickle
import shutil
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from rich import print as rich_print
from rich.text import Text
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from torchtitan.distributed.utils import clip_grad_norm_ as torch_clip_grad_norm_
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_ckpt_dir
from prime_rl.utils.utils import get_step_path

if TYPE_CHECKING:
    from prime_rl.trainer.optim import GradientOffloadManager

DEFAULT_TIMEOUT = timedelta(seconds=600)


class GarbageCollection:
    """Controls Python garbage collection to avoid stragglers in distributed training.

    In multi-GPU training, Python's automatic GC can trigger unpredictably on one rank
    while others wait at a synchronization point, stalling the entire step. This class
    disables automatic GC and runs deterministic collections every `interval` steps so
    all ranks collect simultaneously.

    Based on the approach from torchtitan (https://arxiv.org/abs/2505.05713).
    """

    def __init__(self, interval: int = 50):
        assert interval > 0, "gc interval must be a positive integer"
        self.interval = interval
        gc.disable()
        self._collect()

    def run(self, step: int):
        if step > 0 and step % self.interval == 0:
            self._collect()

    def _collect(self, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        get_logger().info(f"[GC] collection took {time.monotonic() - begin:.2f}s")


def prepare_gradient_offload(
    manager: "GradientOffloadManager | None",
    gradient_scale: float,
    *,
    overlap_optimizer: bool,
) -> None:
    if manager is not None:
        manager.begin_step(gradient_scale, overlap_optimizer=overlap_optimizer)


def begin_backward(manager: "GradientOffloadManager | None", *, final_backward: bool) -> None:
    if manager is not None:
        manager.begin_backward(final_backward=final_backward)


def finish_backward(manager: "GradientOffloadManager | None", *, wait_for_copies: bool = False) -> None:
    if manager is not None:
        manager.finish_backward(wait_for_copies=wait_for_copies)


@torch.no_grad()
def scale_gradients_(manager: "GradientOffloadManager | None", model: nn.Module, factor: float) -> None:
    if manager is not None:
        manager.scale_(factor)
        return
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(factor)


def clip_grad_norm_(
    manager: "GradientOffloadManager | None",
    model: nn.Module,
    max_norm: float,
    ep_enabled: bool,
) -> Tensor:
    if manager is not None:
        grad_norm = manager.clip_grad_norm_(max_norm)
    else:
        grad_norm = torch_clip_grad_norm_(model.parameters(), max_norm=max_norm, ep_enabled=ep_enabled)
    return grad_norm.cuda() if grad_norm.device.type == "cpu" else grad_norm
def get_ckpt_disk_metrics(output_dir: Path) -> dict[str, float]:
    """
    Disk usage metrics for the checkpoint directory (<output_dir>/checkpoints).

    Intended to be called by trainer(s) on rank 0 and included in an existing
    monitor.log(...) call (once per step).
    """
    ckpt_dir = get_ckpt_dir(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(ckpt_dir))
    total = float(usage.total) if usage.total else 0.0
    return {
        "system/ckpt_disk_free_gib": usage.free / 1024**3,
        "system/ckpt_disk_used_gib": usage.used / 1024**3,
        "system/ckpt_disk_total_gib": usage.total / 1024**3,
        "system/ckpt_disk_free_ratio": (usage.free / total) if total else 0.0,
    }


def setup_torch_distributed(timeout: timedelta = DEFAULT_TIMEOUT, enable_gloo: bool = False):
    device_id = get_world().local_rank
    torch.cuda.set_device(device_id)
    # Use Gloo backend for CPU and NCCL for GPU when CPU offloading is enabled
    # Otherwise use NCCL for better GPU performance
    backend = None  # by default nccl
    if enable_gloo:
        get_logger().info("Using Gloo backend for CPU and NCCL backend for GPU")
        backend = "cpu:gloo,cuda:nccl"

    # init_process_group applies `timeout` only to the default PG; device-mesh
    # sub-groups (dp/cp/ep) are created via new_group without a timeout and fall
    # back to torch's 10-minute module default — dist_timeout_seconds silently
    # never reached the PGs doing the real work (watchdogs at 600s). Patch the
    # module default so every subsequently created PG inherits it.
    dist.distributed_c10d.default_pg_timeout = timeout

    dist.init_process_group(backend=backend, timeout=timeout, device_id=device_id)


def print_sample(input_ids: list[int], loss_mask: list[bool], tokenizer: PreTrainedTokenizer):
    """
    Visualize the loss mask of a tokenized sample using rich.
    Reference: https://huggingface.co/Qwen/Qwen3-8B/discussions/14
    """
    text = Text()
    for token, mask in zip(tokenizer.convert_ids_to_tokens(input_ids), loss_mask):
        text.append(token.replace("Ġ", " ").replace("Ċ", "\n"), style="cyan" if mask else "white")
    rich_print(text)


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

        local_keys = list(self.keys())
        gathered_keys: list[list[str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_keys, local_keys)
        keys = sorted({key for rank_keys in gathered_keys if rank_keys is not None for key in rank_keys})

        metrics = {}
        for key in keys:
            # All-gather tensors across steps and ranks (get global distribution)
            values = self.pop(key, [])
            tensors = torch.cat(values, dim=0).to("cuda") if values else torch.empty(0, device="cuda")
            assert tensors.ndim == 1, "Can only aggregate 1D tensors"
            tensors = flexible_all_gather(tensors)
            assert tensors.ndim == 1, "Can only aggregate 1D tensors"

            # Handle empty tensors (can happen when all rollouts in a batch fail)
            if tensors.numel() == 0:
                metrics[f"{key}/mean"] = float("nan")
                metrics[f"{key}/median"] = float("nan")
                metrics[f"{key}/std"] = float("nan")
                metrics[f"{key}/min"] = float("nan")
                metrics[f"{key}/max"] = float("nan")
                continue

            # Compute relevant tensor statistics
            metrics[f"{key}/mean"] = tensors.mean().item()
            metrics[f"{key}/median"] = torch.median(tensors).item()
            metrics[f"{key}/std"] = tensors.std().item()
            metrics[f"{key}/min"] = tensors.min().item()
            metrics[f"{key}/max"] = tensors.max().item()

            # Add back all-gathered tensors to self
            self[key].append(tensors.tolist())

        return metrics


def _is_env_tensor_stat(key: str, allowed_stats: set[str]) -> bool:
    parts = key.split("/")
    return len(parts) >= 3 and parts[-1] in allowed_stats


def filter_rl_trainer_tensor_stats_for_wandb(metrics: dict[str, float | int]) -> dict[str, float | int]:
    """Drop noisy per-token distribution keys before sending RL trainer stats to W&B."""
    skip_prefixes = ("trainer_probs/", "inference_probs/")
    mean_max_only_prefixes = (
        "is_masked/",
        "is_masked_low/",
        "is_masked_high/",
        "masked_advantage_positive/",
        "masked_advantage_negative/",
        "mismatch_kl/",
        "masked_mismatch_kl/",
        "unmasked_mismatch_kl/",
        "ref_kl/is_masked/",
        "ref_kl/masked_mismatch_kl/",
        "ref_kl/unmasked_mismatch_kl/",
    )
    out: dict[str, float | int] = {}
    for k, v in metrics.items():
        if k == "step":
            out[k] = v
            continue
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        if k.startswith("entropy/") and not _is_env_tensor_stat(k, {"mean", "std", "max"}):
            continue
        if any(k.startswith(p) for p in mean_max_only_prefixes):
            if _is_env_tensor_stat(k, {"mean", "std", "max"}):
                out[k] = v
                continue
            if not (k.endswith("/mean") or k.endswith("/max")):
                continue
        out[k] = v
    return out


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


def maybe_clean(path: Path, step: int, interval_to_keep: int | None) -> None:
    """Delete the broadcast dir from 2 trainer steps ago.

    With a 1-step async barrier, the orchestrator at trainer step ``step`` is still consuming the
    ckpt from ``step - 1``; ``step - 2`` is therefore safe to remove unless it falls on a
    checkpoint interval that we want to preserve.
    """
    logger = get_logger()
    candidate_step = max(step - 2, 0)
    candidate_path = get_step_path(path, candidate_step)
    if interval_to_keep and candidate_step % interval_to_keep == 0:
        logger.debug(f"Keeping path {candidate_path} (on ckpt interval)")
        return
    logger.debug(f"Removing path {candidate_path}")
    shutil.rmtree(candidate_path, ignore_errors=True)
