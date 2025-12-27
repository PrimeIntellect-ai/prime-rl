"""
Nightly trainer performance benchmarking tests.

These tests run the RL trainer in benchmark mode to measure throughput, MFU,
step time, and peak memory usage for different model configurations.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


# Model configurations for benchmarking
BENCHMARK_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "nproc_per_node": 4,
        "model_impl": "custom",
        "model_ac": True,
        "model_compile": True,
        "model_cp": 4,
        "lora_rank": 16,
        "adapter_only": True,
        "max_concurrent_runs": 2,
        "optimization_dtype": "bfloat16",
        "batch_size": 16,
        "seq_len": 65536,
    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "nproc_per_node": 4,
        "model_impl": "custom",
        "model_ac": True,
        "model_compile": True,
        "model_cp": 4,
        "lora_rank": 16,
        "adapter_only": True,
        "max_concurrent_runs": 2,
        "optimization_dtype": "bfloat16",
        "batch_size": 16,
        "seq_len": 65536,
    },
]


def parse_benchmark_table(output: str) -> dict[str, Any]:
    """
    Parse the benchmark table output from the trainer.

    The benchmark output looks like:
                         Benchmark
    ┏━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
    ┃    Step ┃  MFU ┃   Throughput ┃  Step Time ┃  Peak Memory ┃
    ┡━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
    │       1 │ 1.5% │       10,000 │      5.00s │     10.0 GiB │
    │       2 │ 1.5% │       10,000 │      5.00s │     10.0 GiB │
    │       3 │ 1.5% │       10,000 │      5.00s │     10.0 GiB │
    │         │      │              │            │              │
    │ Overall │ ...  │         ...  │       ...  │         ...  │
    └─────────┴──────┴──────────────┴────────────┴──────────────┘

    Returns:
        Dictionary with parsed benchmark metrics.
    """
    result: dict[str, Any] = {"steps": [], "overall": {}}

    # Find the Overall row and parse it
    # The Overall row format is like: "1.5% ± 0.1% [1.4%, 1.6%]"
    overall_pattern = r"Overall\s*│\s*([^│]+)│\s*([^│]+)│\s*([^│]+)│\s*([^│]+)"
    overall_match = re.search(overall_pattern, output, re.MULTILINE)

    if overall_match:
        mfu_str = overall_match.group(1).strip()
        throughput_str = overall_match.group(2).strip()
        step_time_str = overall_match.group(3).strip()
        peak_memory_str = overall_match.group(4).strip()

        # Parse MFU (format: "1.5% ± 0.1% [1.4%, 1.6%]")
        mfu_match = re.search(r"([\d.]+)%\s*±\s*([\d.]+)%", mfu_str)
        if mfu_match:
            result["overall"]["mfu_mean"] = float(mfu_match.group(1))
            result["overall"]["mfu_std"] = float(mfu_match.group(2))

        # Parse throughput (format: "10,000 ± 100 [9,900, 10,100]")
        throughput_match = re.search(r"([\d,]+)\s*±\s*([\d,]+)", throughput_str)
        if throughput_match:
            result["overall"]["throughput_mean"] = float(throughput_match.group(1).replace(",", ""))
            result["overall"]["throughput_std"] = float(throughput_match.group(2).replace(",", ""))

        # Parse step time (format: "5.00s ± 0.10s [4.90s, 5.10s]")
        step_time_match = re.search(r"([\d.]+)s\s*±\s*([\d.]+)s", step_time_str)
        if step_time_match:
            result["overall"]["step_time_mean"] = float(step_time_match.group(1))
            result["overall"]["step_time_std"] = float(step_time_match.group(2))

        # Parse peak memory (format: "10.0 GiB (50.0%)")
        peak_memory_match = re.search(r"([\d.]+)\s*GiB", peak_memory_str)
        if peak_memory_match:
            result["overall"]["peak_memory_gib"] = float(peak_memory_match.group(1))

        peak_memory_pct_match = re.search(r"\(([\d.]+)%\)", peak_memory_str)
        if peak_memory_pct_match:
            result["overall"]["peak_memory_pct"] = float(peak_memory_pct_match.group(1))

    return result


def run_trainer_benchmark(config: dict[str, Any], output_dir: Path) -> tuple[int, str, str]:
    """
    Run the trainer benchmark with the given configuration.

    Args:
        config: Benchmark configuration dictionary.
        output_dir: Directory to write outputs to.

    Returns:
        Tuple of (return_code, stdout, stderr).
    """
    cmd = [
        "uv",
        "run",
        "torchrun",
        "--nproc-per-node",
        str(config["nproc_per_node"]),
        "src/prime_rl/trainer/rl/train.py",
        "--model.name",
        config["name"],
        "--model.impl",
        config["model_impl"],
        "--model.optimization_dtype",
        config["optimization_dtype"],
        "--model.cp",
        str(config["model_cp"]),
        "--model.seq-len",
        str(config["seq_len"]),
        "--max-concurrent-runs",
        str(config["max_concurrent_runs"]),
        "--data.fake.batch-size",
        str(config["batch_size"]),
        "--log.level",
        "debug",
        "--output-dir",
        str(output_dir),
        "--bench",
    ]

    # Add optional flags
    if config.get("model_ac"):
        cmd.append("--model.ac")
    if config.get("model_compile"):
        cmd.append("--model.compile")
    if config.get("lora_rank"):
        cmd.extend(["--model.lora.rank", str(config["lora_rank"])])
    if config.get("adapter_only"):
        cmd.append("--weight_broadcast.adapter_only")

    print(f"Running benchmark command: {' '.join(cmd)}")

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=1800,  # 30 minute timeout per benchmark
    )

    return process.returncode, process.stdout, process.stderr


def write_benchmark_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write benchmark results to a markdown file.

    Args:
        results: List of benchmark results.
        output_path: Path to write the markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    branch_name = os.environ.get("GITHUB_REF_NAME", "unknown")

    lines = [
        "# Trainer Performance Benchmark Results",
        "",
        f"**Last Updated:** {timestamp}",
        f"**Branch:** {branch_name}",
        "",
        "## Overview",
        "",
        "This document contains performance benchmark results for the Prime RL trainer.",
        "The benchmarks are run nightly on the research cluster with the following configuration:",
        "",
        "- **GPUs:** 4x NVIDIA GPUs",
        "- **Context Parallelism:** 4",
        "- **LoRA Rank:** 16",
        "- **Sequence Length:** 65,536",
        "- **Batch Size:** 16",
        "- **Max Concurrent Runs:** 2",
        "",
        "## Results",
        "",
    ]

    # Add results table
    lines.extend(
        [
            "| Model | MFU (%) | Throughput (tok/s) | Step Time (s) | Peak Memory (GiB) |",
            "|-------|---------|-------------------|---------------|-------------------|",
        ]
    )

    for result in results:
        model_name = result.get("model", "Unknown")
        overall = result.get("metrics", {}).get("overall", {})

        mfu = overall.get("mfu_mean", "N/A")
        mfu_str = f"{mfu:.1f}" if isinstance(mfu, (int, float)) else str(mfu)

        throughput = overall.get("throughput_mean", "N/A")
        throughput_str = f"{throughput:,.0f}" if isinstance(throughput, (int, float)) else str(throughput)

        step_time = overall.get("step_time_mean", "N/A")
        step_time_str = f"{step_time:.2f}" if isinstance(step_time, (int, float)) else str(step_time)

        peak_mem = overall.get("peak_memory_gib", "N/A")
        peak_mem_str = f"{peak_mem:.1f}" if isinstance(peak_mem, (int, float)) else str(peak_mem)

        lines.append(f"| {model_name} | {mfu_str} | {throughput_str} | {step_time_str} | {peak_mem_str} |")

    lines.extend(
        [
            "",
            "## Detailed Configuration",
            "",
            "```toml",
            "# Benchmark Configuration",
            "[model]",
            'impl = "custom"',
            "ac = true",
            "compile = true",
            "cp = 4",
            'optimization_dtype = "bfloat16"',
            "seq_len = 65536",
            "",
            "[model.lora]",
            "rank = 16",
            "",
            "[weight_broadcast]",
            "adapter_only = true",
            "",
            "[data.fake]",
            "batch_size = 16",
            "```",
            "",
            "## Notes",
            "",
            "- MFU (Model FLOPS Utilization) measures how efficiently the hardware is being used.",
            "- Throughput is measured in tokens per second across all GPUs.",
            "- Step time is the wall-clock time for one training step.",
            "- Peak memory is the maximum GPU memory reserved during training.",
            "- Results exclude the first warmup step.",
            "",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote benchmark results to {output_path}")


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture for temporary output directory for benchmark tests."""
    return tmp_path_factory.mktemp("trainer_perf_benchmark")


@pytest.fixture(scope="module")
def benchmark_results(output_dir: Path) -> list[dict[str, Any]]:
    """
    Fixture that runs all benchmark configurations and collects results.
    This fixture is module-scoped so benchmarks run once and results are shared.
    """
    results = []

    for config in BENCHMARK_CONFIGS:
        model_name = config["name"]
        print(f"\n{'=' * 60}")
        print(f"Running benchmark for: {model_name}")
        print(f"{'=' * 60}")

        model_output_dir = output_dir / model_name.replace("/", "_")
        model_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            return_code, stdout, stderr = run_trainer_benchmark(config, model_output_dir)

            # Combine stdout and stderr for parsing
            combined_output = stdout + "\n" + stderr

            if return_code != 0:
                print(f"Benchmark failed for {model_name}")
                print(f"STDOUT:\n{stdout}")
                print(f"STDERR:\n{stderr}")
                results.append(
                    {
                        "model": model_name,
                        "config": config,
                        "success": False,
                        "error": f"Return code: {return_code}",
                        "metrics": {},
                    }
                )
                continue

            # Parse benchmark results
            metrics = parse_benchmark_table(combined_output)

            results.append(
                {
                    "model": model_name,
                    "config": config,
                    "success": True,
                    "metrics": metrics,
                }
            )

            print(f"Benchmark completed for {model_name}")
            print(f"Metrics: {json.dumps(metrics, indent=2)}")

        except subprocess.TimeoutExpired:
            print(f"Benchmark timed out for {model_name}")
            results.append(
                {
                    "model": model_name,
                    "config": config,
                    "success": False,
                    "error": "Timeout",
                    "metrics": {},
                }
            )
        except Exception as e:
            print(f"Benchmark error for {model_name}: {e}")
            results.append(
                {
                    "model": model_name,
                    "config": config,
                    "success": False,
                    "error": str(e),
                    "metrics": {},
                }
            )

    return results


def test_benchmarks_complete(benchmark_results: list[dict[str, Any]]) -> None:
    """Test that all benchmarks completed successfully."""
    failed = [r for r in benchmark_results if not r.get("success", False)]

    if failed:
        error_msgs = [f"{r['model']}: {r.get('error', 'Unknown error')}" for r in failed]
        pytest.fail("Some benchmarks failed:\n" + "\n".join(error_msgs))


def test_write_benchmark_markdown(benchmark_results: list[dict[str, Any]]) -> None:
    """Test that writes benchmark results to markdown file."""
    # Write to the docs directory in the repo
    output_path = Path("docs/benchmark_results.md")
    write_benchmark_results(benchmark_results, output_path)

    # Verify the file was created
    assert output_path.exists(), f"Benchmark results file not created at {output_path}"


def test_qwen3_4b_throughput(benchmark_results: list[dict[str, Any]]) -> None:
    """Test that Qwen3-4B achieves reasonable throughput."""
    for result in benchmark_results:
        if "4B" in result["model"] and result.get("success"):
            throughput = result.get("metrics", {}).get("overall", {}).get("throughput_mean", 0)
            # Minimum expected throughput - adjust based on actual hardware
            assert throughput > 0, f"Qwen3-4B throughput should be positive, got {throughput}"


def test_qwen3_30b_throughput(benchmark_results: list[dict[str, Any]]) -> None:
    """Test that Qwen3-30B achieves reasonable throughput."""
    for result in benchmark_results:
        if "30B" in result["model"] and result.get("success"):
            throughput = result.get("metrics", {}).get("overall", {}).get("throughput_mean", 0)
            # Minimum expected throughput - adjust based on actual hardware
            assert throughput > 0, f"Qwen3-30B throughput should be positive, got {throughput}"
