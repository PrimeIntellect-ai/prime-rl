#!/usr/bin/env python3
"""
Run a single benchmark configuration and output JSON results.

This script wraps the prime-rl training with --bench.output-json to get
metrics directly without parsing console output.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import Field

from prime_rl.utils.pydantic_config import BaseSettings, parse_argv

MAX_LORAS = 4


def get_commit_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def extract_oom_error_reason(output: str) -> str | None:
    """
    Extract a human-readable error reason from process output.

    Currently we only special-case CUDA OOM to make failures easier to triage in CI.
    """
    needle = "torch.OutOfMemoryError: CUDA out of memory."
    for line in output.splitlines():
        if needle in line:
            return line.strip()
    return None


class BenchmarkConfig(BaseSettings):
    """Configuration for running a single benchmark."""

    type: Annotated[
        Literal["sft", "rl"],
        Field(description="Training type"),
    ] = "rl"

    num_gpus: Annotated[int, Field(ge=1, description="Number of GPUs")] = 2

    model_name: Annotated[str, Field(description="Model name (e.g., Qwen/Qwen3-0.6B)")] = "Qwen/Qwen3-0.6B"

    lora_rank: Annotated[int | None, Field(description="LoRA rank (None for full fine-tuning)")] = None

    seq_len: Annotated[int, Field(ge=1, description="Sequence length")] = 512

    ac: Annotated[Literal["Recompute", "Offload", "None"], Field(description="Activation checkpointing type")] = (
        "Recompute"
    )

    attention: Annotated[
        Literal["sdpa", "flash_attention_2", "flash_attention_3"],
        Field(description="Attention implementation"),
    ] = "flash_attention_2"

    output: Annotated[Path, Field(description="Output JSON file path")] = Path("benchmark_result.json")

    dry_run: Annotated[bool, Field(description="Print command without executing")] = False

    timeout: Annotated[int, Field(description="Timeout in seconds")] = 3600

    micro_batches: Annotated[int, Field(ge=1, description="Number of micro batches")] = 2

    docker_image: Annotated[str | None, Field(description="Docker image used for the benchmark")] = None

    # Metadata set by the script
    device_name: Annotated[str, Field(description="Device name. This is set automatically by the script.")] = (
        torch.cuda.get_device_name()
    )
    commit_sha: Annotated[str, Field(description="Commit SHA. This is set automatically by the script.")] = (
        get_commit_sha()
    )
    timestamp: Annotated[str, Field(description="Timestamp. This is set automatically by the script.")] = datetime.now(
        timezone.utc
    ).isoformat()
    success: Annotated[bool, Field(description="Success. This is set automatically by the script.")] = True
    error_reason: Annotated[str | None, Field(description="Error reason. This is set automatically by the script.")] = (
        None
    )
    time_taken: Annotated[
        float | None, Field(description="Time taken in seconds. This is set automatically by the script.")
    ] = None


def build_command(config: BenchmarkConfig) -> list[str]:
    """Build the benchmark command from config."""
    # Determine training script
    if config.type == "rl":
        script = "src/prime_rl/trainer/rl/train.py"
    elif config.type == "sft":
        script = "src/prime_rl/trainer/sft/train.py"
    else:
        raise ValueError(f"Invalid training type: {config.type}")

    cmd = [
        "uv",
        "run",
        "torchrun",
        f"--nproc-per-node={config.num_gpus}",
        script,
        "--model.name",
        config.model_name,
        "--model.seq-len",
        str(config.seq_len),
        "--model.attn",
        config.attention,
        "--bench.output-json",
        str(config.output),
        "--model.compile",
        "--dist-timeout-seconds",
        str(config.timeout),
    ]

    # Add activation checkpointing if enabled
    if config.ac == "Recompute":
        cmd.append("--model.ac")
    elif config.ac == "Offload":
        cmd.append("--model.ac-offloading")

    # Add LoRA configuration if applicable
    if config.lora_rank is not None:
        cmd.extend(["--model.lora.rank", str(config.lora_rank)])
        cmd.extend(["--max-concurrent-runs", str(MAX_LORAS)])

    # Data configuration differs between RL and SFT
    if config.type.startswith("rl"):
        cmd.extend(["--data.fake.batch-size", str(config.micro_batches * config.num_gpus)])
    else:
        cmd.extend(
            [
                "--data.type",
                "fake",
                "--data.batch-size",
                str(config.micro_batches * config.num_gpus),
                "--data.seq-len",
                str(config.seq_len),
            ]
        )

    return cmd


def dummy_metrics() -> dict:
    return {
        "mfu": {"mean": 0, "std": 0, "min": 0, "max": 0},
        "throughput": {"mean": 0, "std": 0, "min": 0, "max": 0},
        "step_time": {"mean": 0, "std": 0, "min": 0, "max": 0},
        "peak_memory": {"gib": 0, "pct": 0},
    }


def run_benchmark(config: BenchmarkConfig) -> None:
    """Run a single benchmark and write results to output path."""
    cmd = build_command(config)
    print(f"Running: {' '.join(cmd)}")

    if config.dry_run:
        return

    start_time = time.perf_counter()
    try:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )
        output = result.stdout + result.stderr

        if result.returncode != 0:
            config.success = False
            config.error_reason = extract_oom_error_reason(output) or f"Non-zero exit code: {result.returncode}"
            print(f"Process exited with code {result.returncode}: {output}")
        if not config.output.exists():
            config.success = False
            config.error_reason = config.error_reason or extract_oom_error_reason(output) or "No JSON output written"
            print(f"Process exited with code {result.returncode}: {output}")
            print("Benchmark completed but no JSON output was written")
        else:
            with open(config.output) as f:
                metrics = json.load(f)
            lines = output.splitlines()
            print("\n".join(lines))
    except subprocess.TimeoutExpired:
        config.success = False
        config.error_reason = "Timeout"
        print(f"Benchmark timed out after {config.timeout} seconds")
    except Exception as e:
        config.success = False
        config.error_reason = str(e)
        print(f"Error: {e}")
    finally:
        config.time_taken = time.perf_counter() - start_time

    if not config.success:
        metrics = dummy_metrics()

    # Write final result with config and metadata
    final_result = {
        "config": config.model_dump(mode="json"),
        "metrics": metrics,
    }

    with open(config.output, "w") as f:
        json.dump(final_result, f, indent=2)

    print(f"Results written to {config.output}", file=sys.stderr)


def main():
    config = parse_argv(BenchmarkConfig)
    run_benchmark(config)


if __name__ == "__main__":
    main()
