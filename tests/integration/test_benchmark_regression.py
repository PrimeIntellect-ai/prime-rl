"""
Integration test for benchmark regression.

This test runs the benchmark and compares results against a baseline to ensure:
- Peak memory usage is exactly the same
- Other metrics (mfu, throughput, step_time) are within 5% margin
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 15 * 60  # 15 minutes
METRIC_TOLERANCE = 0.05  # 5% tolerance for mfu, throughput, step_time
MEMORY_TOLERANCE = 0.01  # 1% tolerance for peak memory

# Baseline files for the Qwen3-0.6B RL benchmark
BASELINE_FILE_1GPU = Path(
    "benchmarks/baselines/benchmark-1xa6000-Qwen--Qwen3-0.6B-rl-full-1gpu-Recompute-flash_attention_2-65536-cp1-ep1.json"
)
BASELINE_FILE_4GPU = Path(
    "benchmarks/baselines/benchmark-4xa6000-Qwen--Qwen3-0.6B-rl-full-4gpu-Recompute-flash_attention_2-65536-cp1-ep1.json"
)


@pytest.fixture(scope="module")
def baseline_metrics_1gpu() -> dict:
    """Load baseline metrics for 1-GPU benchmark."""
    with open(BASELINE_FILE_1GPU) as f:
        baseline = json.load(f)
    return baseline["metrics"]


@pytest.fixture(scope="module")
def baseline_metrics_4gpu() -> dict:
    """Load baseline metrics for 4-GPU benchmark."""
    with open(BASELINE_FILE_4GPU) as f:
        baseline = json.load(f)
    return baseline["metrics"]


@pytest.fixture(scope="module")
def benchmark_output_file_1gpu(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Output file path for 1-GPU benchmark results."""
    tmp_dir = tmp_path_factory.mktemp("benchmark_1gpu")
    return tmp_dir / "benchmark_result.json"


@pytest.fixture(scope="module")
def benchmark_output_file_4gpu(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Output file path for 4-GPU benchmark results."""
    tmp_dir = tmp_path_factory.mktemp("benchmark_4gpu")
    return tmp_dir / "benchmark_result.json"


@pytest.fixture(scope="module")
def benchmark_process_1gpu(
    run_process: Callable[..., ProcessResult],
    benchmark_output_file_1gpu: Path,
) -> ProcessResult:
    """Run the 1-GPU benchmark and return the process result."""
    cmd = [
        "uv",
        "run",
        "python",
        "benchmarks/scripts/run_single_benchmark.py",
        "--type",
        "rl",
        "--num-gpus",
        "1",
        "--model-name",
        "Qwen/Qwen3-0.6B",
        "--seq-len",
        "65536",
        "--ac",
        "Recompute",
        "--attention",
        "flash_attention_2",
        "--output",
        str(benchmark_output_file_1gpu),
        "--timeout",
        str(TIMEOUT - 5 * 60),  # Leave 5 min buffer
    ]
    return run_process(cmd, timeout=TIMEOUT, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})


@pytest.fixture(scope="module")
def benchmark_process_4gpu(
    run_process: Callable[..., ProcessResult],
    benchmark_output_file_4gpu: Path,
) -> ProcessResult:
    """Run the 4-GPU benchmark and return the process result."""
    cmd = [
        "uv",
        "run",
        "python",
        "benchmarks/scripts/run_single_benchmark.py",
        "--type",
        "rl",
        "--num-gpus",
        "4",
        "--model-name",
        "Qwen/Qwen3-0.6B",
        "--seq-len",
        "65536",
        "--ac",
        "Recompute",
        "--attention",
        "flash_attention_2",
        "--output",
        str(benchmark_output_file_4gpu),
        "--timeout",
        str(TIMEOUT - 5 * 60),  # Leave 5 min buffer
    ]
    return run_process(cmd, timeout=TIMEOUT, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})


@pytest.fixture(scope="module")
def benchmark_metrics_1gpu(benchmark_process_1gpu: ProcessResult, benchmark_output_file_1gpu: Path) -> dict:
    """Load the 1-GPU benchmark metrics from the output file."""
    assert benchmark_process_1gpu.returncode == 0, (
        f"Benchmark process failed with return code {benchmark_process_1gpu.returncode}"
    )
    assert benchmark_output_file_1gpu.exists(), f"Benchmark output file not found at {benchmark_output_file_1gpu}"
    with open(benchmark_output_file_1gpu) as f:
        result = json.load(f)
    assert result.get("config", {}).get("success", False), (
        f"Benchmark did not succeed: {result.get('config', {}).get('error_reason', 'unknown error')}"
    )
    print("=== Benchmark metrics (1-GPU) ===")
    print(result["metrics"])
    return result["metrics"]


@pytest.fixture(scope="module")
def benchmark_metrics_4gpu(benchmark_process_4gpu: ProcessResult, benchmark_output_file_4gpu: Path) -> dict:
    """Load the 4-GPU benchmark metrics from the output file."""
    assert benchmark_process_4gpu.returncode == 0, (
        f"Benchmark process failed with return code {benchmark_process_4gpu.returncode}"
    )
    assert benchmark_output_file_4gpu.exists(), f"Benchmark output file not found at {benchmark_output_file_4gpu}"
    with open(benchmark_output_file_4gpu) as f:
        result = json.load(f)
    assert result.get("config", {}).get("success", False), (
        f"Benchmark did not succeed: {result.get('config', {}).get('error_reason', 'unknown error')}"
    )
    print("=== Benchmark metrics (4-GPU) ===")
    print(result["metrics"])
    return result["metrics"]


# =============================================================================
# 1-GPU Benchmark Tests
# =============================================================================


def test_peak_memory_within_tolerance_1gpu(benchmark_metrics_1gpu: dict, baseline_metrics_1gpu: dict):
    """Test that peak memory usage is within 1% of the baseline (1-GPU)."""
    actual_memory = benchmark_metrics_1gpu["peak_memory"]["gib"]
    expected_memory = baseline_metrics_1gpu["peak_memory"]["gib"]

    lower_bound = expected_memory * (1 - MEMORY_TOLERANCE)
    upper_bound = expected_memory * (1 + MEMORY_TOLERANCE)

    assert lower_bound <= actual_memory <= upper_bound, (
        f"Peak memory out of tolerance! Expected {expected_memory:.4f} GiB ± 1%, got {actual_memory:.4f} GiB. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}] GiB"
    )


def test_mfu_within_tolerance_1gpu(benchmark_metrics_1gpu: dict, baseline_metrics_1gpu: dict):
    """Test that MFU (Model FLOPS Utilization) is within 5% of baseline (1-GPU)."""
    actual_mfu = benchmark_metrics_1gpu["mfu"]["mean"]
    expected_mfu = baseline_metrics_1gpu["mfu"]["mean"]

    lower_bound = expected_mfu * (1 - METRIC_TOLERANCE)
    upper_bound = expected_mfu * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_mfu <= upper_bound, (
        f"MFU out of tolerance! Expected {expected_mfu:.4f} ± 5%, got {actual_mfu:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


def test_throughput_within_tolerance_1gpu(benchmark_metrics_1gpu: dict, baseline_metrics_1gpu: dict):
    """Test that throughput is within 5% of baseline (1-GPU)."""
    actual_throughput = benchmark_metrics_1gpu["throughput"]["mean"]
    expected_throughput = baseline_metrics_1gpu["throughput"]["mean"]

    lower_bound = expected_throughput * (1 - METRIC_TOLERANCE)
    upper_bound = expected_throughput * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_throughput <= upper_bound, (
        f"Throughput out of tolerance! Expected {expected_throughput:.4f} ± 5%, got {actual_throughput:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


def test_step_time_within_tolerance_1gpu(benchmark_metrics_1gpu: dict, baseline_metrics_1gpu: dict):
    """Test that step time is within 5% of baseline (1-GPU)."""
    actual_step_time = benchmark_metrics_1gpu["step_time"]["mean"]
    expected_step_time = baseline_metrics_1gpu["step_time"]["mean"]

    lower_bound = expected_step_time * (1 - METRIC_TOLERANCE)
    upper_bound = expected_step_time * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_step_time <= upper_bound, (
        f"Step time out of tolerance! Expected {expected_step_time:.4f} ± 5%, got {actual_step_time:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


# =============================================================================
# 4-GPU Benchmark Tests
# =============================================================================


def test_peak_memory_within_tolerance_4gpu(benchmark_metrics_4gpu: dict, baseline_metrics_4gpu: dict):
    """Test that peak memory usage is within 1% of the baseline (4-GPU)."""
    actual_memory = benchmark_metrics_4gpu["peak_memory"]["gib"]
    expected_memory = baseline_metrics_4gpu["peak_memory"]["gib"]

    lower_bound = expected_memory * (1 - MEMORY_TOLERANCE)
    upper_bound = expected_memory * (1 + MEMORY_TOLERANCE)

    assert lower_bound <= actual_memory <= upper_bound, (
        f"Peak memory out of tolerance! Expected {expected_memory:.4f} GiB ± 1%, got {actual_memory:.4f} GiB. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}] GiB"
    )


def test_mfu_within_tolerance_4gpu(benchmark_metrics_4gpu: dict, baseline_metrics_4gpu: dict):
    """Test that MFU (Model FLOPS Utilization) is within 5% of baseline (4-GPU)."""
    actual_mfu = benchmark_metrics_4gpu["mfu"]["mean"]
    expected_mfu = baseline_metrics_4gpu["mfu"]["mean"]

    lower_bound = expected_mfu * (1 - METRIC_TOLERANCE)
    upper_bound = expected_mfu * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_mfu <= upper_bound, (
        f"MFU out of tolerance! Expected {expected_mfu:.4f} ± 5%, got {actual_mfu:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


def test_throughput_within_tolerance_4gpu(benchmark_metrics_4gpu: dict, baseline_metrics_4gpu: dict):
    """Test that throughput is within 5% of baseline (4-GPU)."""
    actual_throughput = benchmark_metrics_4gpu["throughput"]["mean"]
    expected_throughput = baseline_metrics_4gpu["throughput"]["mean"]

    lower_bound = expected_throughput * (1 - METRIC_TOLERANCE)
    upper_bound = expected_throughput * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_throughput <= upper_bound, (
        f"Throughput out of tolerance! Expected {expected_throughput:.4f} ± 5%, got {actual_throughput:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


def test_step_time_within_tolerance_4gpu(benchmark_metrics_4gpu: dict, baseline_metrics_4gpu: dict):
    """Test that step time is within 5% of baseline (4-GPU)."""
    actual_step_time = benchmark_metrics_4gpu["step_time"]["mean"]
    expected_step_time = baseline_metrics_4gpu["step_time"]["mean"]

    lower_bound = expected_step_time * (1 - METRIC_TOLERANCE)
    upper_bound = expected_step_time * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_step_time <= upper_bound, (
        f"Step time out of tolerance! Expected {expected_step_time:.4f} ± 5%, got {actual_step_time:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )
