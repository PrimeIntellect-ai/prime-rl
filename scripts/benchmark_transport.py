#!/usr/bin/env python3
"""
Benchmark script for comparing transport implementations:
- FileSystem
- ZMQ
- TCPStore

Measures latency and throughput for micro batch transport with realistic
and 4x scaled batch sizes based on nightly CI configs.

Reference batch sizes from nightly configs:
- hendrycks_math: batch_size=1024, seq_len=2048
- acereason_math: batch_size=1024, seq_len=8192

Usage:
    python scripts/benchmark_transport.py [--iterations N] [--warmup N]
"""

import argparse
import multiprocessing as mp
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Ensure clean imports without side effects
os.environ.setdefault("PRIME_LOG_LEVEL", "WARNING")


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    num_samples: int  # Number of samples in the batch
    seq_len: int  # Sequence length per sample
    data_world_size: int  # Number of data parallel ranks
    description: str


# Realistic configs based on nightly CI (and 4x scaled versions)
BENCHMARK_CONFIGS = [
    # Realistic configs
    BenchmarkConfig(
        name="small",
        num_samples=128,
        seq_len=2048,
        data_world_size=2,
        description="Small batch (integration test size)",
    ),
    BenchmarkConfig(
        name="medium",
        num_samples=1024,
        seq_len=2048,
        data_world_size=4,
        description="Medium batch (hendrycks_math nightly)",
    ),
    BenchmarkConfig(
        name="large",
        num_samples=1024,
        seq_len=8192,
        data_world_size=4,
        description="Large batch (acereason_math nightly)",
    ),
    # 4x scaled configs
    BenchmarkConfig(
        name="4x_medium",
        num_samples=4096,
        seq_len=2048,
        data_world_size=4,
        description="4x medium batch",
    ),
    BenchmarkConfig(
        name="4x_large",
        num_samples=4096,
        seq_len=8192,
        data_world_size=8,
        description="4x large batch",
    ),
]


def generate_micro_batch(seq_len: int, num_sequences: int = 1) -> "MicroBatch":
    """Generate a realistic micro batch with the given sequence length."""
    from prime_rl.transport.types import MicroBatch

    total_len = seq_len * num_sequences
    return MicroBatch(
        input_ids=list(range(total_len)),  # Simulated token IDs
        loss_mask=[i % 2 == 0 for i in range(total_len)],  # Alternating mask
        advantages=[0.1 * (i % 10) for i in range(total_len)],  # Simulated advantages
        inference_logprobs=[-0.5 - 0.01 * (i % 100) for i in range(total_len)],  # Simulated logprobs
        position_ids=list(range(total_len)),  # Position IDs
        temperature=0.6,
    )


def generate_micro_batch_grid(num_samples: int, seq_len: int, data_world_size: int) -> list[list["MicroBatch"]]:
    """Generate a grid of micro batches for all data ranks.

    The grid structure is: grid[data_rank][micro_batch_idx] = MicroBatch
    Each MicroBatch contains packed sequences totaling seq_len tokens.
    """
    samples_per_rank = num_samples // data_world_size
    # Assume each micro batch contains multiple packed sequences
    sequences_per_micro_batch = max(1, samples_per_rank // 4)  # 4 micro batches per rank
    num_micro_batches = max(1, samples_per_rank // sequences_per_micro_batch)

    grid = []
    for _ in range(data_world_size):
        rank_batches = []
        for _ in range(num_micro_batches):
            batch = generate_micro_batch(seq_len, sequences_per_micro_batch)
            rank_batches.append(batch)
        grid.append(rank_batches)
    return grid


def estimate_batch_size_mb(grid: list[list["MicroBatch"]]) -> float:
    """Estimate the size of the micro batch grid in MB."""
    import msgspec

    encoder = msgspec.msgpack.Encoder()
    total_bytes = 0
    for rank_batches in grid:
        data = encoder.encode(rank_batches)
        total_bytes += len(data)
    return total_bytes / (1024 * 1024)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    transport_type: str
    config_name: str
    iterations: int
    total_time_seconds: float
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_mb_per_sec: float
    batch_size_mb: float


def run_filesystem_benchmark(
    config: BenchmarkConfig,
    iterations: int,
    warmup: int,
) -> BenchmarkResult:
    """Benchmark the filesystem transport."""
    from prime_rl.transport.filesystem import FileSystemMicroBatchReceiver, FileSystemMicroBatchSender

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Generate test data
        grid = generate_micro_batch_grid(config.num_samples, config.seq_len, config.data_world_size)
        batch_size_mb = estimate_batch_size_mb(grid)

        latencies = []

        for i in range(warmup + iterations):
            step = i
            sender = FileSystemMicroBatchSender(output_dir, config.data_world_size, step)

            # Measure send + receive latency
            start = time.perf_counter()
            sender.send(grid)

            # Simulate receivers reading
            for rank in range(config.data_world_size):
                receiver = FileSystemMicroBatchReceiver(output_dir, rank, step)
                receiver.wait()
                _ = receiver.receive()

            end = time.perf_counter()

            if i >= warmup:
                latencies.append((end - start) * 1000)  # Convert to ms

            # Cleanup step directory
            shutil.rmtree(output_dir / "rollouts", ignore_errors=True)

        total_time = sum(latencies) / 1000  # Convert back to seconds
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        throughput = (batch_size_mb * iterations) / total_time if total_time > 0 else 0

        return BenchmarkResult(
            transport_type="filesystem",
            config_name=config.name,
            iterations=iterations,
            total_time_seconds=total_time,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            throughput_mb_per_sec=throughput,
            batch_size_mb=batch_size_mb,
        )


def _zmq_receiver_worker(
    rank: int,
    data_world_size: int,
    output_dir: Path,
    host: str,
    port: int,
    iterations: int,
    warmup: int,
    ready_event: mp.Event,
    done_event: mp.Event,
) -> None:
    """Worker process for ZMQ receiver."""
    os.environ.setdefault("PRIME_LOG_LEVEL", "WARNING")
    from prime_rl.transport.config import ZMQTransportConfig
    from prime_rl.transport.zmq import ZMQMicroBatchReceiver

    transport = ZMQTransportConfig(host=host, port=port)
    receiver = ZMQMicroBatchReceiver(output_dir, rank, 0, transport)

    # Signal ready
    ready_event.set()

    # Receive all batches
    for _ in range(warmup + iterations):
        receiver.wait()
        _ = receiver.receive()

    receiver.close()
    done_event.set()


def run_zmq_benchmark(
    config: BenchmarkConfig,
    iterations: int,
    warmup: int,
) -> BenchmarkResult:
    """Benchmark the ZMQ transport."""
    from prime_rl.transport.config import ZMQTransportConfig
    from prime_rl.transport.zmq import ZMQMicroBatchSender

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        host = "127.0.0.1"
        port = 15555  # Use non-standard port to avoid conflicts

        transport = ZMQTransportConfig(host=host, port=port)

        # Generate test data
        grid = generate_micro_batch_grid(config.num_samples, config.seq_len, config.data_world_size)
        batch_size_mb = estimate_batch_size_mb(grid)

        # Start receiver workers
        ready_events = []
        done_events = []
        workers = []

        for rank in range(config.data_world_size):
            ready_event = mp.Event()
            done_event = mp.Event()
            ready_events.append(ready_event)
            done_events.append(done_event)

            worker = mp.Process(
                target=_zmq_receiver_worker,
                args=(
                    rank,
                    config.data_world_size,
                    output_dir,
                    host,
                    port,
                    iterations,
                    warmup,
                    ready_event,
                    done_event,
                ),
            )
            worker.start()
            workers.append(worker)

        # Wait for all receivers to be ready
        for event in ready_events:
            event.wait(timeout=30)

        # Create sender after receivers are ready
        sender = ZMQMicroBatchSender(output_dir, config.data_world_size, 0, transport)

        latencies = []

        for i in range(warmup + iterations):
            start = time.perf_counter()
            sender.send(grid)
            end = time.perf_counter()

            if i >= warmup:
                latencies.append((end - start) * 1000)

        sender.close()

        # Wait for workers to finish
        for event in done_events:
            event.wait(timeout=30)

        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        total_time = sum(latencies) / 1000
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        throughput = (batch_size_mb * iterations) / total_time if total_time > 0 else 0

        return BenchmarkResult(
            transport_type="zmq",
            config_name=config.name,
            iterations=iterations,
            total_time_seconds=total_time,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            throughput_mb_per_sec=throughput,
            batch_size_mb=batch_size_mb,
        )


def _tcpstore_receiver_worker(
    rank: int,
    data_world_size: int,
    output_dir: Path,
    host: str,
    port: int,
    iterations: int,
    warmup: int,
    ready_event: mp.Event,
    done_event: mp.Event,
) -> None:
    """Worker process for TCPStore receiver."""
    os.environ.setdefault("PRIME_LOG_LEVEL", "WARNING")
    from prime_rl.transport.config import TCPStoreTransportConfig
    from prime_rl.transport.tcpstore import TCPStoreMicroBatchReceiver

    transport = TCPStoreTransportConfig(host=host, port=port)
    receiver = TCPStoreMicroBatchReceiver(output_dir, rank, 0, transport)

    # Signal ready
    ready_event.set()

    # Receive all batches
    for _ in range(warmup + iterations):
        receiver.wait()
        _ = receiver.receive()

    receiver.close()
    done_event.set()


def run_tcpstore_benchmark(
    config: BenchmarkConfig,
    iterations: int,
    warmup: int,
) -> BenchmarkResult:
    """Benchmark the TCPStore transport."""
    from prime_rl.transport.config import TCPStoreTransportConfig
    from prime_rl.transport.tcpstore import TCPStoreMicroBatchSender

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        host = "127.0.0.1"
        port = 29700  # Use non-standard port to avoid conflicts

        transport = TCPStoreTransportConfig(host=host, port=port)

        # Generate test data
        grid = generate_micro_batch_grid(config.num_samples, config.seq_len, config.data_world_size)
        batch_size_mb = estimate_batch_size_mb(grid)

        # Create sender first (it's the master for TCPStore)
        sender = TCPStoreMicroBatchSender(output_dir, config.data_world_size, 0, transport)

        # Start receiver workers
        ready_events = []
        done_events = []
        workers = []

        for rank in range(config.data_world_size):
            ready_event = mp.Event()
            done_event = mp.Event()
            ready_events.append(ready_event)
            done_events.append(done_event)

            worker = mp.Process(
                target=_tcpstore_receiver_worker,
                args=(
                    rank,
                    config.data_world_size,
                    output_dir,
                    host,
                    port,
                    iterations,
                    warmup,
                    ready_event,
                    done_event,
                ),
            )
            worker.start()
            workers.append(worker)

        # Wait for all receivers to be ready
        for event in ready_events:
            event.wait(timeout=30)

        latencies = []

        for i in range(warmup + iterations):
            start = time.perf_counter()
            sender.send(grid)
            end = time.perf_counter()

            if i >= warmup:
                latencies.append((end - start) * 1000)

        sender.close()

        # Wait for workers to finish
        for event in done_events:
            event.wait(timeout=30)

        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        total_time = sum(latencies) / 1000
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        throughput = (batch_size_mb * iterations) / total_time if total_time > 0 else 0

        return BenchmarkResult(
            transport_type="tcpstore",
            config_name=config.name,
            iterations=iterations,
            total_time_seconds=total_time,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            throughput_mb_per_sec=throughput,
            batch_size_mb=batch_size_mb,
        )


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as an ASCII table."""
    # Group by config
    configs = {}
    for r in results:
        if r.config_name not in configs:
            configs[r.config_name] = {}
        configs[r.config_name][r.transport_type] = r

    lines = []
    lines.append("=" * 120)
    lines.append("MICRO BATCH TRANSPORT BENCHMARK RESULTS")
    lines.append("=" * 120)
    lines.append("")

    header = f"{'Config':<15} {'Transport':<12} {'Batch MB':<10} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'MB/s':<10}"
    lines.append(header)
    lines.append("-" * 120)

    for config_name in configs:
        for transport_type in ["filesystem", "zmq", "tcpstore"]:
            if transport_type in configs[config_name]:
                r = configs[config_name][transport_type]
                line = f"{r.config_name:<15} {r.transport_type:<12} {r.batch_size_mb:<10.2f} {r.mean_latency_ms:<12.2f} {r.std_latency_ms:<10.2f} {r.min_latency_ms:<10.2f} {r.max_latency_ms:<10.2f} {r.throughput_mb_per_sec:<10.2f}"
                lines.append(line)
        lines.append("-" * 120)

    lines.append("")
    lines.append("Legend:")
    lines.append("  - Batch MB: Size of micro batch grid in megabytes")
    lines.append("  - Mean/Std/Min/Max (ms): Latency statistics in milliseconds")
    lines.append("  - MB/s: Throughput in megabytes per second")
    lines.append("")

    return "\n".join(lines)


def format_comparison_summary(results: list[BenchmarkResult]) -> str:
    """Format a comparison summary showing speedups."""
    # Group by config
    configs = {}
    for r in results:
        if r.config_name not in configs:
            configs[r.config_name] = {}
        configs[r.config_name][r.transport_type] = r

    lines = []
    lines.append("=" * 80)
    lines.append("SPEEDUP COMPARISON (vs filesystem baseline)")
    lines.append("=" * 80)
    lines.append("")

    header = f"{'Config':<15} {'ZMQ Speedup':<15} {'TCPStore Speedup':<18}"
    lines.append(header)
    lines.append("-" * 80)

    for config_name, transports in configs.items():
        if "filesystem" not in transports:
            continue

        fs_latency = transports["filesystem"].mean_latency_ms

        zmq_speedup = "N/A"
        if "zmq" in transports:
            zmq_latency = transports["zmq"].mean_latency_ms
            zmq_speedup = f"{fs_latency / zmq_latency:.2f}x"

        tcp_speedup = "N/A"
        if "tcpstore" in transports:
            tcp_latency = transports["tcpstore"].mean_latency_ms
            tcp_speedup = f"{fs_latency / tcp_latency:.2f}x"

        line = f"{config_name:<15} {zmq_speedup:<15} {tcp_speedup:<18}"
        lines.append(line)

    lines.append("-" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark transport implementations")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["small", "medium", "large", "4x_medium", "4x_large", "all"],
        default=["all"],
        help="Which configs to benchmark",
    )
    parser.add_argument(
        "--transports",
        nargs="+",
        choices=["filesystem", "zmq", "tcpstore", "all"],
        default=["all"],
        help="Which transports to benchmark",
    )
    args = parser.parse_args()

    # Determine which configs to run
    if "all" in args.configs:
        configs_to_run = BENCHMARK_CONFIGS
    else:
        configs_to_run = [c for c in BENCHMARK_CONFIGS if c.name in args.configs]

    # Determine which transports to run
    if "all" in args.transports:
        transports_to_run = ["filesystem", "zmq", "tcpstore"]
    else:
        transports_to_run = args.transports

    transport_runners: dict[str, Callable[[BenchmarkConfig, int, int], BenchmarkResult]] = {
        "filesystem": run_filesystem_benchmark,
        "zmq": run_zmq_benchmark,
        "tcpstore": run_tcpstore_benchmark,
    }

    print(f"\nRunning benchmarks with {args.iterations} iterations and {args.warmup} warmup iterations")
    print(f"Configs: {[c.name for c in configs_to_run]}")
    print(f"Transports: {transports_to_run}")
    print("")

    results = []

    for config in configs_to_run:
        print(f"\n{'=' * 60}")
        print(f"Config: {config.name} - {config.description}")
        print(f"  Samples: {config.num_samples}, Seq len: {config.seq_len}, DP size: {config.data_world_size}")
        print(f"{'=' * 60}")

        for transport in transports_to_run:
            print(f"\n  Running {transport} benchmark...", end=" ", flush=True)
            try:
                result = transport_runners[transport](config, args.iterations, args.warmup)
                results.append(result)
                print(
                    f"Done. Mean latency: {result.mean_latency_ms:.2f}ms, Throughput: {result.throughput_mb_per_sec:.2f} MB/s"
                )
            except Exception as e:
                print(f"FAILED: {e}")

    # Print summary
    print("\n\n")
    print(format_results_table(results))
    print(format_comparison_summary(results))


if __name__ == "__main__":
    # Use spawn method for multiprocessing to avoid issues with fork
    mp.set_start_method("spawn", force=True)
    main()
