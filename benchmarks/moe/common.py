#!/usr/bin/env python3
"""Shared helpers for Prime-RL MoE microbenchmarks."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from pydantic import Field

from prime_rl.utils.config import BaseConfig, cli

BenchmarkDType = Literal["bfloat16", "float16", "float32"]

_DTYPE_MAP: dict[BenchmarkDType, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class BaseMoEBenchmarkConfig(BaseConfig):
    token_num: Annotated[int, Field(ge=0, description="Number of tokens to benchmark")] = 8192
    hidden: Annotated[int, Field(ge=1, description="Hidden dimension")] = 256
    topk: Annotated[int, Field(ge=1, description="Experts per token")] = 2
    num_experts: Annotated[int, Field(ge=1, description="Number of experts")] = 8
    warmup: Annotated[int, Field(ge=0, description="Warmup iterations")] = 10
    iters: Annotated[int, Field(ge=1, description="Timed iterations")] = 50
    dtype: Annotated[BenchmarkDType, Field(description="Activation dtype")] = "bfloat16"
    allow_dupe: Annotated[bool, Field(description="Allow duplicate expert ids within a token")] = False
    invalid_frac: Annotated[float, Field(ge=0.0, le=1.0, description="Fraction of invalid -1 indices")] = 0.0
    check: Annotated[bool, Field(description="Run a correctness check before timing")] = True


@dataclass
class BenchmarkResult:
    name: str
    fw_ms: float
    bw_ms: float
    total_ms: float
    fw_peak_mb: float
    total_peak_mb: float
    fw_ok: bool | None = None
    bw_ok: bool | None = None


def resolve_dtype(name: BenchmarkDType) -> torch.dtype:
    return _DTYPE_MAP[name]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Prime-RL MoE microbenchmarks")
    return torch.device("cuda")


def make_topk_ids(
    token_num: int,
    topk: int,
    num_experts: int,
    device: torch.device,
    *,
    allow_dupe: bool,
    invalid_frac: float,
) -> torch.Tensor:
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int64)
    if topk > 1 and not allow_dupe:
        for i in range(1, topk):
            clash = topk_ids[:, i] == topk_ids[:, 0]
            topk_ids[clash, i] = (topk_ids[clash, i] + i) % num_experts
    if invalid_frac > 0:
        mask = torch.rand((token_num, topk), device=device) < invalid_frac
        topk_ids = topk_ids.masked_fill(mask, -1)
    return topk_ids


def make_token_weights(token_num: int, topk: int, device: torch.device) -> torch.Tensor:
    token_weights = torch.rand((token_num, topk), device=device, dtype=torch.float32)
    if token_num == 0:
        return token_weights
    return token_weights / token_weights.sum(dim=1, keepdim=True)


def scatter_index_to_token_indices(scatter_index: torch.Tensor) -> torch.Tensor:
    token_num, topk = scatter_index.shape
    flat_scatter_index = scatter_index.reshape(-1)
    valid = flat_scatter_index >= 0
    out_size = int(valid.sum().item())
    if out_size == 0:
        return torch.zeros((0,), device=scatter_index.device, dtype=torch.long)

    scatter_pos = flat_scatter_index[valid].to(torch.int64)
    flat_token_indices = torch.arange(token_num, device=scatter_index.device, dtype=torch.long).repeat_interleave(topk)
    token_indices = torch.empty((out_size,), device=scatter_index.device, dtype=torch.long)
    token_indices[scatter_pos] = flat_token_indices[valid]
    return token_indices


def benchmark_variant(
    name: str,
    run_once,
    reset_grads,
    *,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    for _ in range(warmup):
        with torch.no_grad():
            _ = run_once()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = run_once()
    torch.cuda.synchronize()
    fw_ms = (time.perf_counter() - start) * 1000.0 / iters
    fw_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    for _ in range(warmup):
        out = run_once()
        out.sum().backward()
        reset_grads()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        out = run_once()
        out.sum().backward()
        reset_grads()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    total_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return BenchmarkResult(
        name=name,
        fw_ms=fw_ms,
        bw_ms=total_ms - fw_ms,
        total_ms=total_ms,
        fw_peak_mb=fw_peak_mb,
        total_peak_mb=total_peak_mb,
    )


def print_result_table(
    label: str,
    params: dict[str, object],
    results: list[BenchmarkResult],
    *,
    include_checks: bool,
) -> None:
    print(f"[{label}] " + " ".join(f"{key}={value}" for key, value in params.items()))
    header = "name, fw_ms, bw_ms, total_ms, speedup_vs_base"
    if include_checks:
        header += ", fw_ok, bw_ok"
    header += ", fw_peak_mb, total_peak_mb"
    print(header)

    base = next(result for result in results if result.name == "baseline")
    for result in results:
        speedup = base.total_ms / result.total_ms if result.total_ms else 0.0
        row = f"{result.name}, {result.fw_ms:.3f}, {result.bw_ms:.3f}, {result.total_ms:.3f}, {speedup:.2f}x"
        if include_checks:
            row += f", {result.fw_ok}, {result.bw_ok}"
        row += f", {result.fw_peak_mb:.1f}MB, {result.total_peak_mb:.1f}MB"
        print(row)
