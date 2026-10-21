import gc
import math
import os
import pickle
import shutil
import time
from collections import defaultdict
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from rich import print as rich_print
from rich.text import Text
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.control.controller import Controller
from prime_rl.trainer.world import get_world
from prime_rl.utils.device import DEVICE_TYPE, DIST_BACKEND, device_module
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir

if TYPE_CHECKING:
    from prime_rl.trainer.optim import GradientOffloadManager

DEFAULT_TIMEOUT = timedelta(seconds=600)
DCP_TIMEOUT = timedelta(hours=4)


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
        get_logger().debug(f"Collected garbage in {format_time(time.monotonic() - begin)}")


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
) -> Tensor:
    if manager is not None:
        grad_norm = manager.clip_grad_norm_(max_norm)
    else:
        # Norm reductions must complete on each parameter mesh before combining
        # dense, expert-parallel, and other model-parallel gradients.
        mesh_parameters = defaultdict(list)
        for param in model.parameters():
            if param.grad is not None:
                mesh = param.grad.device_mesh if isinstance(param.grad, DTensor) else None
                mesh_parameters[mesh].append(param)
        norms = []
        for parameters in mesh_parameters.values():
            norm = torch.nn.utils.get_total_norm([param.grad for param in parameters])
            norms.append(norm.full_tensor() if isinstance(norm, DTensor) else norm)
        grad_norm = torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.tensor(0.0)
        for parameters in mesh_parameters.values():
            torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, grad_norm)
    return grad_norm.to(DEVICE_TYPE) if grad_norm.device.type == "cpu" else grad_norm


def get_ckpt_disk_metrics(output_dir: Path) -> dict[str, float]:
    """
    Disk usage metrics for the checkpoint directory (<output_dir>/checkpoints).

    Intended to be called by trainer(s) on rank 0 and included in an existing
    monitors.log(...) call (once per step).
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


def bind_process_to_gpu_numa_node() -> None:
    """Pin this rank's CPUs to its GPU's NUMA node.

    Offloaded optimizer state is pageable host memory placed by first-touch, and the
    CPU optimizer pipeline is DRAM-bandwidth-bound; binding before the state is
    allocated keeps slabs, OMP threads, and pinned rings local to the socket the
    GPU hangs off. Must run before CPU optimizer state allocation and before the
    OMP thread pool spins up.
    """
    logger = get_logger()
    if DEVICE_TYPE == "xpu":
        logger.warning("GPU NUMA lookup uses NVML, which has no Intel XPU equivalent; skipping NUMA binding")
        return

    import pynvml

    device_id = torch.cuda.current_device()
    pynvml.nvmlInit()
    try:
        bus_id = pynvml.nvmlDeviceGetPciInfo(pynvml.nvmlDeviceGetHandleByIndex(device_id)).busId
    finally:
        pynvml.nvmlShutdown()
    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode()
    domain, rest = bus_id.split(":", 1)
    sysfs_bus_id = f"{int(domain, 16):04x}:{rest}".lower()
    numa_node = int(Path(f"/sys/bus/pci/devices/{sysfs_bus_id}/numa_node").read_text())
    if numa_node < 0:
        logger.warning(f"GPU {device_id} ({sysfs_bus_id}) reports no NUMA node; skipping NUMA binding")
        return
    cpus: set[int] = set()
    for part in Path(f"/sys/devices/system/node/node{numa_node}/cpulist").read_text().strip().split(","):
        if "-" in part:
            start, end = part.split("-")
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    os.sched_setaffinity(0, cpus)
    logger.info(f"Bound rank with GPU {device_id} to NUMA node {numa_node} ({len(cpus)} CPUs)")


def configure_cpu_optimizer_threads() -> None:
    available = os.sched_getaffinity(0)
    fair_share = (os.cpu_count() or len(available)) // get_world().local_world_size
    threads = max(1, min(len(available), fair_share))
    torch.set_num_threads(threads)
    get_logger().info(
        f"CPU optimizer uses {threads} intra-op threads "
        f"({len(available)} CPUs in this rank's affinity mask, {get_world().local_world_size} local ranks)"
    )


def setup_full_cpu_optimizer_offload() -> None:
    bind_process_to_gpu_numa_node()
    configure_cpu_optimizer_threads()


def setup_torch_distributed(timeout: timedelta = DEFAULT_TIMEOUT, enable_gloo: bool = False):
    get_logger().info(f"Initializing torch distributed (timeout={int(timeout.total_seconds())}s)")
    t0 = time.perf_counter()
    device_id = get_world().local_rank
    device_module.set_device(device_id)
    # Use Gloo backend for CPU and NCCL for GPU when CPU offloading is enabled
    # Otherwise use NCCL for better GPU performance
    backend = None  # by default nccl
    if enable_gloo:
        get_logger().info(f"Using Gloo backend for CPU and {DIST_BACKEND} backend for GPU")
        backend = f"cpu:gloo,{DEVICE_TYPE}:{DIST_BACKEND}"

    # init_process_group applies `timeout` only to the default PG; device-mesh
    # sub-groups (dp/cp/ep) are created via new_group without a timeout and fall
    # back to torch's 10-minute module default — dist_timeout_seconds silently
    # never reached the PGs doing the real work (watchdogs at 600s). Patch the
    # module default so every subsequently created PG inherits it.
    dist.distributed_c10d.default_pg_timeout = timeout

    # Binding the default group to a device makes every later new_group look up a backend for that
    # device. Gloo handles CUDA tensors but not XPU ones, so CPU-only gloo groups (e.g. the DCP group)
    # fail on XPU when the default group is bound; bind on CUDA only.
    bound_device = torch.device(DEVICE_TYPE, device_id) if DEVICE_TYPE == "cuda" else None
    dist.init_process_group(backend=backend, timeout=timeout, device_id=bound_device)
    get_logger().debug(f"Initialized torch distributed in {format_time(time.perf_counter() - t0)}")


@cache
def get_dcp_group() -> dist.ProcessGroup:
    """A gloo group for DCP checkpoint save/load. DCP's collectives wait for the slowest rank's
    writes or reads (and, on the first HF load, its download), which can outlast
    ``dist_timeout_seconds``; on this group they wait up to ``DCP_TIMEOUT`` instead. Every rank must
    make its first call at the same point."""
    return dist.new_group(backend="gloo", timeout=DCP_TIMEOUT)


def broadcast_pause(controller: Controller, step: int) -> bool:
    """Whether the master's controller pauses at ``step``, applied by every rank."""
    paused = torch.tensor(controller.update(step), device=DEVICE_TYPE)
    dist.broadcast(paused, src=0)
    return bool(paused.item())


def print_sample(input_ids: list[int], loss_mask: list[bool], tokenizer: PreTrainedTokenizer):
    """
    Visualize the loss mask of a tokenized sample using rich.
    Reference: https://huggingface.co/Qwen/Qwen3-8B/discussions/14
    """
    text = Text()
    for token, mask in zip(tokenizer.convert_ids_to_tokens(input_ids), loss_mask):
        text.append(token.replace("Ġ", " ").replace("Ċ", "\n"), style="cyan" if mask else "white")
    rich_print(text)


class TensorStats:
    """Accumulates a step's tensor metrics on the GPU and reduces them across ranks once per step.

    Values are folded into a running count, sum, sum of squares, min and max per key, so memory
    stays constant however many tokens a step trains on. Each key reports mean, std, min and max
    over all its values on all ranks. Keys without any value on any rank are left out."""

    def __init__(self):
        assert dist.is_initialized(), "TensorStats requires a distributed environment"
        # Per key: count, sum, sum of squares, -min, max
        self.moments: dict[str, Tensor] = {}

    def add(
        self,
        key: str,
        values: Tensor,
        mask: Tensor | None = None,
        env_ids: Tensor | None = None,
        env_names: list[str] | None = None,
    ) -> None:
        """Fold ``values`` (where ``mask`` is set) into ``key``. With ``env_ids`` (each value's index into
        ``env_names``), fold them into ``{key}/{env}`` per env and into ``{key}/all`` instead.

        Uses masked reductions instead of boolean indexing, so it never waits for the GPU."""
        # A custom loss may emit a metric on the CPU
        values = values.detach().reshape(-1).to(device=DEVICE_TYPE, dtype=torch.float64)
        if values.numel() == 0:
            return
        mask = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.reshape(-1).to(values.device)
        if env_ids is None:
            self._fold(key, _moments(values, mask[:, None])[0])
            return
        env_ids = env_ids.reshape(-1).to(values.device)
        members = (env_ids[:, None] == torch.arange(len(env_names), device=values.device)) & mask[:, None]
        rows = _moments(values, members)
        for env_name, row in zip(env_names, rows):
            self._fold(f"{key}/{env_name}", row)
        self._fold(f"{key}/all", torch.cat([rows[:, :3].sum(0), rows[:, 3:].amax(0)]))

    def _fold(self, key: str, row: Tensor) -> None:
        if (moments := self.moments.get(key)) is None:
            self.moments[key] = row
        else:
            moments[:3] += row[:3]
            torch.maximum(moments[3:], row[3:], out=moments[3:])

    def compute_stats(self) -> dict[str, float]:
        """Reduce every key across all ranks and compute its statistics. Collective: every rank must call it."""
        gathered_keys: list[list[str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_keys, list(self.moments))
        keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
        if not keys:
            return {}

        empty = torch.tensor([0, 0, 0, -math.inf, -math.inf], dtype=torch.float64, device=DEVICE_TYPE)
        rows = torch.stack([self.moments.get(key, empty) for key in keys])
        sums, extremes = rows[:, :3].contiguous(), rows[:, 3:].contiguous()
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(extremes, op=dist.ReduceOp.MAX)
        self.moments.clear()

        metrics = {}
        for key, (count, total, total_sq), (neg_min, max_) in zip(keys, sums.tolist(), extremes.tolist()):
            if count == 0:
                continue
            mean = total / count
            metrics[f"{key}/mean"] = mean
            # Bessel-corrected like torch.std
            variance = max(total_sq - total * mean, 0.0) / (count - 1) if count > 1 else math.nan
            metrics[f"{key}/std"] = math.sqrt(variance)
            metrics[f"{key}/min"] = -neg_min
            metrics[f"{key}/max"] = max_
        return metrics


def _moments(values: Tensor, members: Tensor) -> Tensor:
    """Count, sum, sum of squares, -min and max of ``values`` for each column of the boolean
    ``members`` [values, groups]. Plain reductions, so no atomics contend on a few addresses."""
    values = values[:, None]
    member_values = torch.where(members, values, 0.0)
    return torch.stack(
        [
            members.sum(0, dtype=torch.float64),
            member_values.sum(0),
            member_values.square().sum(0),
            torch.where(members, -values, -math.inf).amax(0),
            torch.where(members, values, -math.inf).amax(0),
        ],
        dim=1,
    )


def _is_env_tensor_stat(key: str, allowed_stats: set[str]) -> bool:
    parts = key.split("/")
    return len(parts) >= 3 and parts[-1] in allowed_stats


def filter_rl_trainer_tensor_stats_for_wandb(metrics: dict[str, float | int]) -> dict[str, float | int]:
    """Drop noisy per-token distribution keys before sending RL trainer stats to W&B."""
    skip_prefixes = ("trainer_probs/", "inference_probs/")
    mean_max_only_prefixes = (
        "is_masked/",
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
        device_module.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
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
            pickle.dump(device_module.memory._snapshot(), output)
        self.logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds, load {file_path} at https://docs.pytorch.org/memory_viz to visualize the memory usage"
        )
        self.step_num += 1
