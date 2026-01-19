"""
Integration tests for A6000 benchmark regression testing.

These tests run the benchmarks that are configured for the A6000 GPU and verify
that the metrics (TPS, step_time, MFU, peak_memory) don't regress beyond the
configured threshold compared to the baselines.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.gpu, pytest.mark.benchmark]


# Regression threshold: 5% (same as CI workflow)
REGRESSION_THRESHOLD = 0.05

# Timeout for each benchmark (in seconds)
BENCHMARK_TIMEOUT = 600  # 10 minutes

# Path to baselines directory
BASELINES_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "baselines"


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    model: str
    training_type: str  # "rl" or "sft"
    lora_rank: int | None
    seq_len: int
    attention: str
    ac: str  # Activation checkpointing: "Recompute", "Offload", or "None"
    num_gpus: int = 1
    cp: int = 1  # Context parallelism
    ep: int = 1  # Expert parallelism
    micro_batches: int = 2

    @property
    def baseline_filename(self) -> str:
        """Generate the baseline filename for this config."""
        model_safe = self.model.replace("/", "--")
        lora_str = str(self.lora_rank) if self.lora_rank else "full"
        return f"benchmark-{self.num_gpus}xa6000-{model_safe}-{self.training_type}-{lora_str}-{self.num_gpus}gpu-{self.ac}-{self.attention}-{self.seq_len}-cp{self.cp}-ep{self.ep}.json"

    @property
    def description(self) -> str:
        """Human readable description of the config."""
        lora_str = f"LoRA(r={self.lora_rank})" if self.lora_rank else "Full"
        return f"{self.model} {self.training_type.upper()} {lora_str} seq={self.seq_len}"


@dataclass
class BenchmarkBaseline:
    """Baseline metrics for a benchmark."""

    mfu_mean: float
    throughput_mean: float  # TPS
    step_time_mean: float
    peak_memory_gib: float
    peak_memory_pct: float

    @classmethod
    def from_json(cls, data: dict) -> "BenchmarkBaseline":
        """Load baseline from JSON data."""
        metrics = data["metrics"]
        return cls(
            mfu_mean=metrics["mfu"]["mean"],
            throughput_mean=metrics["throughput"]["mean"],
            step_time_mean=metrics["step_time"]["mean"],
            peak_memory_gib=metrics["peak_memory"]["gib"],
            peak_memory_pct=metrics["peak_memory"]["pct"],
        )


@dataclass
class BenchmarkResult:
    """Result metrics from a benchmark run."""

    mfu_mean: float
    throughput_mean: float  # TPS
    step_time_mean: float
    peak_memory_gib: float
    peak_memory_pct: float
    success: bool
    error_reason: str | None = None

    @classmethod
    def from_json(cls, data: dict) -> "BenchmarkResult":
        """Load result from JSON data."""
        config = data["config"]
        metrics = data["metrics"]
        return cls(
            mfu_mean=metrics["mfu"]["mean"],
            throughput_mean=metrics["throughput"]["mean"],
            step_time_mean=metrics["step_time"]["mean"],
            peak_memory_gib=metrics["peak_memory"]["gib"],
            peak_memory_pct=metrics["peak_memory"]["pct"],
            success=config.get("success", True),
            error_reason=config.get("error_reason"),
        )


def check_regression(
    name: str,
    actual: float,
    baseline: float,
    threshold: float,
    higher_is_better: bool = True,
) -> tuple[bool, str]:
    """
    Check if a metric has regressed beyond the threshold.

    Returns:
        Tuple of (passed, message)
    """
    if baseline == 0:
        return True, f"{name}: baseline is 0, skipping check"

    pct_change = (actual - baseline) / baseline * 100

    if higher_is_better:
        # For metrics where higher is better (MFU, TPS), regression is when actual < baseline * (1 - threshold)
        is_regression = actual < baseline * (1 - threshold)
    else:
        # For metrics where lower is better (step_time, peak_memory), regression is when actual > baseline * (1 + threshold)
        is_regression = actual > baseline * (1 + threshold)

    if is_regression:
        return False, f"{name}: REGRESSION - actual={actual:.4f}, baseline={baseline:.4f}, change={pct_change:+.2f}%"
    else:
        return True, f"{name}: OK - actual={actual:.4f}, baseline={baseline:.4f}, change={pct_change:+.2f}%"


# A6000 benchmark configurations (from .github/workflows/benchmarks.yaml)
A6000_CONFIGS = [
    BenchmarkConfig(
        model="Qwen/Qwen3-0.6B",
        training_type="rl",
        lora_rank=None,
        seq_len=16384,
        attention="flash_attention_2",
        ac="Recompute",
    ),
    BenchmarkConfig(
        model="Qwen/Qwen3-0.6B",
        training_type="rl",
        lora_rank=None,
        seq_len=65536,
        attention="flash_attention_2",
        ac="Recompute",
    ),
    BenchmarkConfig(
        model="Qwen/Qwen3-0.6B",
        training_type="rl",
        lora_rank=16,
        seq_len=16384,
        attention="flash_attention_2",
        ac="Recompute",
    ),
    BenchmarkConfig(
        model="Qwen/Qwen3-0.6B",
        training_type="rl",
        lora_rank=16,
        seq_len=65536,
        attention="flash_attention_2",
        ac="Recompute",
    ),
    BenchmarkConfig(
        model="Qwen/Qwen3-0.6B",
        training_type="sft",
        lora_rank=None,
        seq_len=8192,
        attention="flash_attention_2",
        ac="Recompute",
    ),
    BenchmarkConfig(
        model="Qwen/Qwen3-4B-Instruct-2507",
        training_type="rl",
        lora_rank=16,
        seq_len=16384,
        attention="flash_attention_2",
        ac="Recompute",
    ),
]


def get_device_name() -> str:
    """Get the current GPU device name."""
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def is_a6000() -> bool:
    """Check if the current GPU is an A6000."""
    device_name = get_device_name()
    return "A6000" in device_name


def load_baseline(config: BenchmarkConfig) -> BenchmarkBaseline | None:
    """Load the baseline for a given config."""
    baseline_path = BASELINES_DIR / config.baseline_filename
    if not baseline_path.exists():
        return None
    with open(baseline_path) as f:
        data = json.load(f)
    return BenchmarkBaseline.from_json(data)


def run_benchmark(config: BenchmarkConfig, output_path: Path) -> BenchmarkResult:
    """Run a single benchmark and return the results."""
    cmd = [
        "uv",
        "run",
        "python",
        "benchmarks/scripts/run_single_benchmark.py",
        "--type",
        config.training_type,
        "--num-gpus",
        str(config.num_gpus),
        "--model-name",
        config.model,
        "--seq-len",
        str(config.seq_len),
        "--ac",
        config.ac,
        "--attention",
        config.attention,
        "--cp",
        str(config.cp),
        "--ep",
        str(config.ep),
        "--micro-batches",
        str(config.micro_batches),
        "--output",
        str(output_path),
        "--timeout",
        str(BENCHMARK_TIMEOUT),
    ]

    if config.lora_rank is not None:
        cmd.extend(["--lora-rank", str(config.lora_rank)])

    print(f"Running benchmark: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BENCHMARK_TIMEOUT + 60,  # Extra minute for setup/teardown
    )

    if result.returncode != 0:
        print(f"Benchmark stderr: {result.stderr}")
        print(f"Benchmark stdout: {result.stdout}")

    if not output_path.exists():
        return BenchmarkResult(
            mfu_mean=0,
            throughput_mean=0,
            step_time_mean=0,
            peak_memory_gib=0,
            peak_memory_pct=0,
            success=False,
            error_reason=f"Output file not created. Return code: {result.returncode}",
        )

    with open(output_path) as f:
        data = json.load(f)

    return BenchmarkResult.from_json(data)


def validate_metrics(
    config: BenchmarkConfig,
    result: BenchmarkResult,
    baseline: BenchmarkBaseline,
    threshold: float = REGRESSION_THRESHOLD,
) -> tuple[bool, list[str]]:
    """
    Validate benchmark metrics against baseline.

    Returns:
        Tuple of (all_passed, list of messages)
    """
    messages = []
    all_passed = True

    # Check MFU (higher is better)
    passed, msg = check_regression("MFU", result.mfu_mean, baseline.mfu_mean, threshold, higher_is_better=True)
    messages.append(msg)
    all_passed = all_passed and passed

    # Check Throughput/TPS (higher is better)
    passed, msg = check_regression(
        "Throughput (TPS)", result.throughput_mean, baseline.throughput_mean, threshold, higher_is_better=True
    )
    messages.append(msg)
    all_passed = all_passed and passed

    # Check Step Time (lower is better)
    passed, msg = check_regression(
        "Step Time", result.step_time_mean, baseline.step_time_mean, threshold, higher_is_better=False
    )
    messages.append(msg)
    all_passed = all_passed and passed

    # Check Peak Memory (lower is better - we don't want memory usage to increase)
    passed, msg = check_regression(
        "Peak Memory (GiB)", result.peak_memory_gib, baseline.peak_memory_gib, threshold, higher_is_better=False
    )
    messages.append(msg)
    all_passed = all_passed and passed

    return all_passed, messages


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for benchmark tests."""
    return f"test-benchmark-a6000-{branch_name}"


@pytest.fixture(scope="module")
def check_a6000_available():
    """Skip tests if A6000 is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not is_a6000():
        pytest.skip(f"A6000 GPU required, found: {get_device_name()}")


# Generate test parameters from configs
def get_benchmark_test_params():
    """Generate pytest parameters for each benchmark config."""
    params = []
    for config in A6000_CONFIGS:
        params.append(
            pytest.param(
                config,
                id=config.description,
            )
        )
    return params


@pytest.mark.parametrize("config", get_benchmark_test_params())
def test_benchmark_no_regression(config: BenchmarkConfig, check_a6000_available, tmp_path: Path):
    """
    Test that benchmark metrics don't regress compared to baseline.

    This test:
    1. Loads the baseline for the given config
    2. Runs the benchmark
    3. Validates that MFU, TPS, step_time, and peak_memory don't regress
    """
    # Load baseline
    baseline = load_baseline(config)
    if baseline is None:
        pytest.skip(f"No baseline found for {config.description} at {config.baseline_filename}")

    # Run benchmark
    output_path = tmp_path / "benchmark_result.json"
    result = run_benchmark(config, output_path)

    # Check benchmark succeeded
    assert result.success, f"Benchmark failed: {result.error_reason}"

    # Validate metrics
    all_passed, messages = validate_metrics(config, result, baseline)

    # Print all messages for visibility
    print(f"\n{'='*60}")
    print(f"Benchmark: {config.description}")
    print(f"{'='*60}")
    for msg in messages:
        print(msg)
    print(f"{'='*60}\n")

    # Assert all metrics pass
    assert all_passed, f"Benchmark regression detected:\n" + "\n".join(messages)


@pytest.mark.parametrize("config", get_benchmark_test_params())
def test_baseline_exists(config: BenchmarkConfig):
    """
    Test that baseline files exist for all A6000 configurations.

    This is a quick sanity check that doesn't require GPU.
    """
    baseline_path = BASELINES_DIR / config.baseline_filename
    assert baseline_path.exists(), f"Missing baseline file: {baseline_path}"

    # Also verify the baseline file has valid structure
    with open(baseline_path) as f:
        data = json.load(f)

    assert "config" in data, f"Baseline missing 'config' key: {baseline_path}"
    assert "metrics" in data, f"Baseline missing 'metrics' key: {baseline_path}"

    metrics = data["metrics"]
    assert "mfu" in metrics, f"Baseline missing 'mfu' metric: {baseline_path}"
    assert "throughput" in metrics, f"Baseline missing 'throughput' metric: {baseline_path}"
    assert "step_time" in metrics, f"Baseline missing 'step_time' metric: {baseline_path}"
    assert "peak_memory" in metrics, f"Baseline missing 'peak_memory' metric: {baseline_path}"
