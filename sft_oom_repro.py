"""CPU-only host-memory reproducer for the prime-rl SFT dataloader (main vs PackedDataLoader).

Run from a prime-rl checkout (main or feat/sft-pack-before-shard) with identical args on both:

    uv run torchrun --nproc-per-node 8 sft_oom_repro.py --subsets 4 --num-workers 8

Experiment matrix (runs 1-3 on both checkouts, 4-6 on the branch only; compare the printed peak):

    1. baseline:          --subsets 4 --num-workers 8
    2. no interleave:     --subsets 1 --num-workers 8
    3. one worker:        --subsets 4 --num-workers 1
    4. fork:              --subsets 4 --num-workers 8 --start-method fork
    5. no flatten:        --subsets 4 --num-workers 8 --sources config
    6. on-disk shuffle:   --subsets 4 --num-workers 8 --shuffle disk

See sft_oom_repro.md for what each run isolates.
"""

import argparse
import os
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

GIB = 1024**3
SOURCES_ENV = "SFT_OOM_REPRO_SOURCES"
SHUFFLE_ENV = "SFT_OOM_REPRO_SHUFFLE"
_patched = False
_logged_patch_calls = set()


def log_patch_call(name: str, detail: str) -> None:
    if name in _logged_patch_calls:
        return
    _logged_patch_calls.add(name)
    from torch.utils.data import get_worker_info

    worker = get_worker_info()
    where = f"dataloader worker {worker.id}" if worker is not None else "main process"
    print(f"[patch] {name} active in {where} (pid {os.getpid()}): {detail}", flush=True)


def apply_patches() -> None:
    global _patched
    if _patched:
        return
    _patched = True
    import datasets

    if os.environ.get(SOURCES_ENV) == "config":

        def base_table_unique(self, column: str) -> list:
            log_patch_call("sources=config", f"Dataset.unique({column!r}) read the base table")
            return self._data.column(column).unique().to_pylist()

        datasets.Dataset.unique = base_table_unique

    if os.environ.get(SHUFFLE_ENV) == "disk":
        original_shuffle = datasets.Dataset.shuffle

        def on_disk_shuffle(self, *args, **kwargs):
            requested = kwargs.get("keep_in_memory")
            kwargs["keep_in_memory"] = False
            shuffled = original_shuffle(self, *args, **kwargs)
            log_patch_call(
                "shuffle=disk",
                f"keep_in_memory {requested} -> False, indices table {type(shuffled._indices).__name__}",
            )
            return shuffled

        datasets.Dataset.shuffle = on_disk_shuffle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--subsets", type=int, default=4)
    parser.add_argument("--payload-chars", type=int, default=4000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--start-method", choices=["default", "spawn", "fork"], default="default")
    parser.add_argument("--sources", choices=["unique", "config"], default="unique")
    parser.add_argument("--shuffle", choices=["memory", "disk"], default="memory")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--tokenizer", default="PrimeIntellect/Qwen3-0.6B")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--print-interval", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = Path.home() / "tmp" / "sft_oom_data" / f"{args.rows}_{args.subsets}_{args.payload_chars}"
    return args


def subset_names(num_subsets: int) -> list[str]:
    return [f"subset_{k}" for k in range(num_subsets)]


def build_dataset(data_dir: Path, rows: int, num_subsets: int, payload_chars: int, seed: int) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    weights = range(1, num_subsets + 1)
    sizes = [rows * weight // sum(weights) for weight in weights]
    sizes[-1] += rows - sum(sizes)
    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    schema = pa.schema([("prompt", pa.list_(message)), ("completion", pa.list_(message))])
    alphabet = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz      ", dtype=np.uint8)
    rng = np.random.default_rng(seed)
    half = payload_chars // 2
    staging = data_dir.with_name(data_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    readme = ["---", "configs:"]
    for name, size in zip(subset_names(num_subsets), sizes):
        readme += [f"- config_name: {name}", "  data_files:", "  - split: train", f"    path: {name}.parquet"]
        with pq.ParquetWriter(staging / f"{name}.parquet", schema) as writer:
            for start in range(0, size, 4096):
                count = min(4096, size - start)
                chars = alphabet[rng.integers(0, len(alphabet), size=(count, payload_chars), dtype=np.uint8)]
                texts = [row.tobytes().decode() for row in chars]
                table = pa.table(
                    {
                        "prompt": [[{"role": "user", "content": text[:half]}] for text in texts],
                        "completion": [[{"role": "assistant", "content": text[half:]}] for text in texts],
                    },
                    schema=schema,
                )
                writer.write_table(table)
        print(f"[data] wrote {name} with {size} rows", flush=True)
    readme.append("---")
    (staging / "README.md").write_text("\n".join(readme) + "\n")
    staging.rename(data_dir)


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for file in files:
            try:
                total += os.lstat(os.path.join(root, file)).st_size
            except FileNotFoundError:
                pass
    return total


def read_meminfo() -> dict[str, int]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {}
    values = {}
    for line in meminfo.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("Dirty", "Shmem"):
            values[key] = int(rest.split()[0]) * 1024
    return values


class MemoryMonitor(threading.Thread):
    def __init__(self, sample_interval: float, print_interval: float, cache_dir: Path):
        import psutil

        super().__init__(daemon=True)
        self.psutil = psutil
        self.root = psutil.Process(os.getppid())
        self.metric = "pss" if hasattr(psutil.Process().memory_full_info(), "pss") else "rss"
        self.sample_interval = sample_interval
        self.print_interval = print_interval
        self.cache_dir = cache_dir
        self.cache_start = dir_size(cache_dir)
        self.start_time = time.perf_counter()
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.phase = "startup"
        self.phase_peak = 0
        self.peak = (0, 0.0, "startup")

    def sample(self) -> dict:
        memory = 0
        processes = 0
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                info = process.memory_full_info() if self.metric == "pss" else process.memory_info()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied, self.psutil.ZombieProcess):
                continue
            memory += getattr(info, self.metric)
            processes += 1
        elapsed = time.perf_counter() - self.start_time
        with self.lock:
            self.phase_peak = max(self.phase_peak, memory)
            if memory > self.peak[0]:
                self.peak = (memory, elapsed, self.phase)
        return {"elapsed": elapsed, "memory": memory, "processes": processes, **read_meminfo()}

    def report(self, event: str, sample: dict, phase: str, phase_peak: int) -> None:
        extras = "".join(f" {key.lower()}={sample[key] / GIB:.2f}GiB" for key in ("Dirty", "Shmem") if key in sample)
        cache_delta = (dir_size(self.cache_dir) - self.cache_start) / GIB
        print(
            f"[mem] t={sample['elapsed']:7.1f}s {event:<5} phase={phase:<12} procs={sample['processes']:3d} "
            f"{self.metric}={sample['memory'] / GIB:.2f}GiB phase_peak={phase_peak / GIB:.2f}GiB"
            f"{extras} hf_cache_delta={cache_delta:.2f}GiB",
            flush=True,
        )

    def run(self) -> None:
        last_print = time.perf_counter()
        while not self.finished.wait(self.sample_interval):
            sample = self.sample()
            if time.perf_counter() - last_print >= self.print_interval:
                last_print = time.perf_counter()
                self.report("tick", sample, self.phase, self.phase_peak)

    def begin(self, phase: str) -> None:
        sample = self.sample()
        with self.lock:
            self.phase = phase
            self.phase_peak = sample["memory"]
        self.report("begin", sample, phase, sample["memory"])

    def end(self) -> None:
        sample = self.sample()
        self.report("end", sample, self.phase, self.phase_peak)

    def summary(self) -> None:
        self.finished.set()
        self.join()
        memory, elapsed, phase = self.peak
        print(
            f"[mem] PEAK {self.metric}={memory / GIB:.2f}GiB at t={elapsed:.1f}s during phase={phase} "
            f"(summed over torchrun agent pid {self.root.pid} and its descendants)",
            flush=True,
        )


def main():
    args = parse_args()
    if args.sources == "config":
        os.environ[SOURCES_ENV] = "config"
    if args.shuffle == "disk":
        os.environ[SHUFFLE_ENV] = "disk"
    apply_patches()

    import torch.distributed as dist

    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout_seconds))

    import datasets
    from renderers import AutoRendererConfig
    from transformers import AutoTokenizer

    from prime_rl.configs.sft import SFTDataConfig
    from prime_rl.trainer.world import get_world

    try:
        from prime_rl.trainer.sft.data import broker
    except ImportError:
        broker = None

    world = get_world()
    monitor = None
    if world.local_rank == 0:
        monitor = MemoryMonitor(args.sample_interval, args.print_interval, Path(datasets.config.HF_DATASETS_CACHE))
        monitor.start()

    @contextmanager
    def phase(name: str):
        if monitor is not None:
            monitor.begin(name)
        start = time.perf_counter()
        yield
        print(f"[rank {world.rank}] {name}: {time.perf_counter() - start:.1f}s", flush=True)
        if monitor is not None:
            monitor.end()

    if broker is None and (args.start_method, args.sources, args.shuffle) != ("default", "unique", "memory"):
        raise SystemExit("--start-method, --sources and --shuffle apply only to the PackedDataLoader branch")

    if world.rank == 0:
        api = "branch (PackedDataLoader)" if broker is not None else "main (StatefulDataLoader)"
        print(f"[setup] api={api} world_size={world.world_size} args={vars(args)}", flush=True)
        print(f"[setup] hf_datasets_cache={datasets.config.HF_DATASETS_CACHE}", flush=True)
        if monitor is not None:
            print(f"[setup] memory metric={monitor.metric}", flush=True)

    with phase("build_data"):
        if world.rank == 0 and not args.data_dir.exists():
            build_dataset(args.data_dir, args.rows, args.subsets, args.payload_chars, args.seed)
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = SFTDataConfig(
        name=str(args.data_dir),
        subsets=subset_names(args.subsets),
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    micro_batches_per_step = args.batch_size // (world.world_size * args.micro_batch_size)
    if micro_batches_per_step * world.world_size * args.micro_batch_size != args.batch_size:
        raise SystemExit("--batch-size must be divisible by world_size * --micro-batch-size")

    if broker is not None:
        from prime_rl.trainer.sft.data import setup_dataloader
        from prime_rl.trainer.sft.data.dataset import load_sft_dataset

        if args.start_method != "default":
            original_dataloader = broker.DataLoader

            def forced_context_dataloader(*loader_args, **loader_kwargs):
                loader_kwargs["multiprocessing_context"] = args.start_method
                return original_dataloader(*loader_args, **loader_kwargs)

            broker.DataLoader = forced_context_dataloader

        with phase("load_dataset"):
            raw_dataset = load_sft_dataset(config)
        with phase("build_loader"):
            dataloader = setup_dataloader(
                tokenizer,
                config,
                1,
                timeout_seconds=args.timeout_seconds,
                raw_dataset=raw_dataset,
                renderer_config=AutoRendererConfig(),
            )
        if not isinstance(dataloader, broker.PackedDataLoader):
            raise SystemExit(f"expected PackedDataLoader, got {type(dataloader).__name__}")
    else:
        from prime_rl.trainer.sft.data import load_sft_dataset, setup_dataloader, setup_dataset

        with phase("load_dataset"):
            raw_dataset = load_sft_dataset(config)
        with phase("build_loader"):
            dataset = setup_dataset(
                tokenizer, config, 1, raw_dataset=raw_dataset, renderer_config=AutoRendererConfig()
            )
            dataloader = setup_dataloader(dataset, config)

    dataiter = iter(dataloader)
    with phase("first_batch"):
        next(dataiter)
    for step in range(1, args.steps + 1):
        remaining = micro_batches_per_step - 1 if step == 1 else micro_batches_per_step
        with phase(f"step_{step}"):
            for _ in range(remaining):
                next(dataiter)

    if monitor is not None:
        monitor.summary()
    if broker is not None:
        dataloader.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
else:
    apply_patches()
