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

# Baseline file for the Qwen3-0.6B RL benchmark
BASELINE_FILE = Path(
    "benchmarks/baselines/benchmark-1xa6000-Qwen--Qwen3-0.6B-rl-full-1gpu-Recompute-flash_attention_2-65536-cp1-ep1.json"
)


@pytest.fixture(scope="module")
def baseline_metrics() -> dict:
    """Load baseline metrics from the baseline file."""
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    return baseline["metrics"]


@pytest.fixture(scope="module")
def benchmark_output_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Output file path for benchmark results."""
    tmp_dir = tmp_path_factory.mktemp("benchmark")
    return tmp_dir / "benchmark_result.json"


@pytest.fixture(scope="module")
def benchmark_process(
    run_process: Callable[..., ProcessResult],
    benchmark_output_file: Path,
) -> ProcessResult:
    """Run the benchmark and return the process result."""
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
        str(benchmark_output_file),
        "--timeout",
        str(TIMEOUT - 5 * 60),  # Leave 5 min buffer
    ]
    return run_process(cmd, timeout=TIMEOUT, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})


@pytest.fixture(scope="module")
def benchmark_metrics(benchmark_process: ProcessResult, benchmark_output_file: Path) -> dict:
    """Load the benchmark metrics from the output file."""
    assert benchmark_process.returncode == 0, (
        f"Benchmark process failed with return code {benchmark_process.returncode}"
    )
    assert benchmark_output_file.exists(), f"Benchmark output file not found at {benchmark_output_file}"
    with open(benchmark_output_file) as f:
        result = json.load(f)
    assert result.get("config", {}).get("success", False), (
        f"Benchmark did not succeed: {result.get('config', {}).get('error_reason', 'unknown error')}"
    )
    print("=== Benchmark metrics ===")
    print(result["metrics"])
    return result["metrics"]


def test_peak_memory_within_tolerance(benchmark_metrics: dict, baseline_metrics: dict):
    """Test that peak memory usage is within 1% of the baseline."""
    actual_memory = benchmark_metrics["peak_memory"]["gib"]
    expected_memory = baseline_metrics["peak_memory"]["gib"]

    lower_bound = expected_memory * (1 - MEMORY_TOLERANCE)
    upper_bound = expected_memory * (1 + MEMORY_TOLERANCE)

    assert lower_bound <= actual_memory <= upper_bound, (
        f"Peak memory out of tolerance! Expected {expected_memory:.4f} GiB ± 1%, got {actual_memory:.4f} GiB. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}] GiB"
    )


def test_mfu_within_tolerance(benchmark_metrics: dict, baseline_metrics: dict):
    """Test that MFU (Model FLOPS Utilization) is within 5% of baseline."""
    actual_mfu = benchmark_metrics["mfu"]["mean"]
    expected_mfu = baseline_metrics["mfu"]["mean"]

    lower_bound = expected_mfu * (1 - METRIC_TOLERANCE)
    upper_bound = expected_mfu * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_mfu <= upper_bound, (
        f"MFU out of tolerance! Expected {expected_mfu:.4f} ± 5%, got {actual_mfu:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


def test_throughput_within_tolerance(benchmark_metrics: dict, baseline_metrics: dict):
    """Test that throughput is within 5% of baseline."""
    actual_throughput = benchmark_metrics["throughput"]["mean"]
    expected_throughput = baseline_metrics["throughput"]["mean"]

    lower_bound = expected_throughput * (1 - METRIC_TOLERANCE)
    upper_bound = expected_throughput * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_throughput <= upper_bound, (
        f"Throughput out of tolerance! Expected {expected_throughput:.4f} ± 5%, got {actual_throughput:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


def test_step_time_within_tolerance(benchmark_metrics: dict, baseline_metrics: dict):
    """Test that step time is within 5% of baseline."""
    actual_step_time = benchmark_metrics["step_time"]["mean"]
    expected_step_time = baseline_metrics["step_time"]["mean"]

    lower_bound = expected_step_time * (1 - METRIC_TOLERANCE)
    upper_bound = expected_step_time * (1 + METRIC_TOLERANCE)

    assert lower_bound <= actual_step_time <= upper_bound, (
        f"Step time out of tolerance! Expected {expected_step_time:.4f} ± 5%, got {actual_step_time:.4f}. "
        f"Acceptable range: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )
