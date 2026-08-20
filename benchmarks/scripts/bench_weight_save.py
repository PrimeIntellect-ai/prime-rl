#!/usr/bin/env python3
"""Benchmark HF weight-save implementations: master gather+write vs rank-parallel.

Builds a randomly initialized FSDP-sharded model (no weight download) and times
the full weight-save path of each implementation:

  master:   all-gather -> master copies all weights to CPU -> master converts
            to HF format -> master writes all safetensors shards
  parallel: all-gather -> each rank keeps a layer-aligned slice in pinned CPU
            buffers -> per-rank convert -> all ranks write their shards

Run with e.g.:

  uv run torchrun --nproc-per-node 2 benchmarks/scripts/bench_weight_save.py \
      --model-name poolside/Laguna-XS.2 --output-dir outputs/bench-weight-save

With --check, both implementations save the same in-memory model and the outputs
are compared tensor-by-tensor (use a small model; the check loads both
checkpoints into CPU memory).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Annotated, Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import Field
from torch import Tensor
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    load_state_dict,
    save_state_dict,
    save_state_dict_parallel,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger

IMPLS = ("master", "parallel")


def gather_weights_on_master(
    model: nn.Module, is_master: bool, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Tensor]:
    """Baseline: gather all weights on CPU on the master rank with blocking copies."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                value = cast(DTensor, value.to(dtype)).full_tensor()

            if is_master:
                key = get_fqns(model, key)
                assert len(key) == 1
                key = next(iter(key))
                cpu_state[key] = value.to("cpu", non_blocking=False)
        torch.distributed.barrier()

    return cpu_state


class BenchWeightSaveConfig(BaseConfig):
    model_name: Annotated[str, Field(description="HF model name or local config path (weights are random-init)")] = (
        "poolside/Laguna-XS.2"
    )
    output_dir: Annotated[Path, Field(description="Directory to write checkpoints into (needs model-size space)")] = (
        Path("outputs/bench-weight-save")
    )
    impls: Annotated[list[Literal["master", "parallel"]], Field(description="Implementations to benchmark")] = [
        "master",
        "parallel",
    ]
    reps: Annotated[int, Field(ge=1, description="Timed repetitions per implementation")] = 2
    fsync: Annotated[bool, Field(description="fsync all written files as a separate timed phase")] = False
    check: Annotated[bool, Field(description="Compare both implementations' outputs tensor-by-tensor")] = False
    results_path: Annotated[Path | None, Field(description="Write results JSON to this path (master only)")] = None


def _barrier_timed(start: float) -> float:
    torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter() - start


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def _fsync_dir(path: Path) -> None:
    for f in sorted(path.iterdir()):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _save(impl: str, model, save_dir: Path, fsync: bool, is_master: bool) -> dict[str, float]:
    """Run one full weight save, returning per-phase wall times (rank-0 view)."""
    times: dict[str, float] = {}

    start = time.perf_counter()
    if impl == "master":
        state_dict = gather_weights_on_master(model, is_master=is_master, dtype=torch.bfloat16)
    else:
        state_dict = gather_weights_parallel(model, dtype=torch.bfloat16)
    times["gather"] = _barrier_timed(start)

    start = time.perf_counter()
    state_dict = convert_state_dict_to_hf(model, state_dict)
    times["convert"] = _barrier_timed(start)

    start = time.perf_counter()
    if impl == "master":
        if is_master:
            save_state_dict(state_dict, save_dir)
    else:
        save_state_dict_parallel(state_dict, save_dir)
    times["write"] = _barrier_timed(start)

    if fsync:
        start = time.perf_counter()
        if is_master:
            _fsync_dir(save_dir)
        times["fsync"] = _barrier_timed(start)

    times["total"] = sum(times.values())
    return times


def _check_equivalence(dir_a: Path, dir_b: Path) -> None:
    state_a = load_state_dict(dir_a)
    state_b = load_state_dict(dir_b)
    assert state_a.keys() == state_b.keys(), (
        f"Key mismatch: only in {dir_a}: {sorted(state_a.keys() - state_b.keys())[:5]}, "
        f"only in {dir_b}: {sorted(state_b.keys() - state_a.keys())[:5]}"
    )
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"Tensor mismatch for {key}"
    index_b = dir_b / "model.safetensors.index.json"
    if index_b.exists():
        index = json.loads(index_b.read_text())
        assert index["weight_map"].keys() == state_b.keys(), "Index does not cover all tensors"
        for shard in set(index["weight_map"].values()):
            assert (dir_b / shard).exists(), f"Index references missing shard {shard}"
    print(f"CHECK PASSED: {len(state_a)} tensors identical across {dir_a.name} and {dir_b.name}", flush=True)


def main(config: BenchWeightSaveConfig):
    world = get_world()
    logger = setup_logger("debug" if world.is_master else "warning")
    setup_torch_distributed()

    model_config = ModelConfig(name=config.model_name)
    model_config.debug.random_init = True
    resolve_ep(model_config)
    parallel_dims = get_parallel_dims(model_config)
    logger.info(f"Setting up model {config.model_name} (random init, world_size={world.world_size})")
    model = setup_model(model_config, parallel_dims)

    model_bytes = sum(
        value.numel() * torch.bfloat16.itemsize if value.is_floating_point() else value.numel() * value.element_size()
        for value in model.state_dict().values()
    )
    if world.is_master:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(config.output_dir).free
        needed = int(model_bytes * (2.2 if config.check else 1.1))
        if free_bytes < needed:
            raise RuntimeError(
                f"Not enough disk space in {config.output_dir}: need ~{needed / 1e9:.0f}GB, have {free_bytes / 1e9:.0f}GB"
            )
        logger.info(f"Model size ~{model_bytes / 1e9:.1f}GB, free disk {free_bytes / 1e9:.0f}GB")
    dist.barrier()

    results: dict[str, list[dict[str, float]]] = {impl: [] for impl in config.impls}
    check_dirs: dict[str, Path] = {}
    for impl in config.impls:
        for rep in range(config.reps):
            save_dir = config.output_dir / f"{impl}-rep{rep}"
            if world.is_master:
                save_dir.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            torch.cuda.reset_peak_memory_stats()
            times = _save(impl, model, save_dir, config.fsync, world.is_master)
            times["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30

            keep = config.check and rep == config.reps - 1
            if world.is_master:
                bytes_written = _dir_bytes(save_dir)
                times["gbps"] = bytes_written / 1e9 / times["write"]
                logger.info(
                    f"[{impl} rep {rep}] "
                    + " ".join(f"{phase}={value:.2f}" for phase, value in times.items())
                    + f" (wrote {bytes_written / 1e9:.1f}GB)"
                )
                if keep:
                    check_dirs[impl] = save_dir
                else:
                    shutil.rmtree(save_dir)
            results[impl].append(times)
            dist.barrier()

    if world.is_master:
        print(f"\n=== weight save benchmark: {config.model_name}, {world.world_size} ranks ===", flush=True)
        phases = ["gather", "convert", "write", "fsync", "total", "gbps", "peak_gpu_gib"]
        header = f"{'impl':<10}{'rep':<5}" + "".join(f"{phase:>14}" for phase in phases)
        print(header, flush=True)
        for impl in config.impls:
            for rep, times in enumerate(results[impl]):
                row = f"{impl:<10}{rep:<5}" + "".join(f"{times.get(phase, 0):>14.2f}" for phase in phases)
                print(row, flush=True)

        if config.results_path is not None:
            payload = {
                "model_name": config.model_name,
                "world_size": world.world_size,
                "model_bytes": model_bytes,
                "fsync": config.fsync,
                "results": results,
            }
            config.results_path.parent.mkdir(parents=True, exist_ok=True)
            config.results_path.write_text(json.dumps(payload, indent=2) + "\n")
            logger.info(f"Wrote results to {config.results_path}")

        if config.check:
            assert set(check_dirs) == set(IMPLS), "--check needs both impls in --impls"
            _check_equivalence(check_dirs["master"], check_dirs["parallel"])
            for path in check_dirs.values():
                shutil.rmtree(path)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(cli(BenchWeightSaveConfig))
