import copy
import pickle
import resource
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import torch.distributed as dist
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.trainer.sft.data import CatDataset, RendererResolver, SFTDataset, cat_collate, setup_dataset
from prime_rl.trainer.sft.packing import OnlinePacker, SampleDescriptor, schedule_rows
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


def prepare_sample(sample: dict, capacity: int) -> dict:
    """Apply the local packer's text cutoff before choosing a bucket."""
    length = len(sample["input_ids"])
    if sample.get("mm_kwargs") is not None and length > capacity:
        raise ValueError("Multimodal samples must be placeholder-safely truncated by the renderer")
    result = dict(sample)
    for key in ("input_ids", "target_ids", "position_ids", "loss_mask"):
        if len(sample[key]) != length:
            raise ValueError(f"Misaligned {key} in processed sample")
        result[key] = sample[key][:capacity]
    if sample.get("mm_token_type_ids") is not None:
        if len(sample["mm_token_type_ids"]) != length:
            raise ValueError("Misaligned multimodal token types")
        result["mm_token_type_ids"] = sample["mm_token_type_ids"][:capacity]
    result["seq_lens"] = [min(length, capacity)]
    if not length or not any(result["loss_mask"]):
        raise ValueError("Processed samples must contain trainable tokens after truncation")
    return result


def compatibility_key(sample: dict) -> tuple:
    sidecars = sample.get("mm_kwargs")
    if sidecars is None:
        return ("text",)
    return (
        "multimodal",
        sample.get("mm_token_type_ids") is not None,
        tuple((key, str(value.dtype), tuple(value.shape[1:])) for key, value in sorted(sidecars.items())),
    )


def materialize_rows(
    buckets: list[list[SampleDescriptor]], samples: dict[tuple[int, int], dict], capacity: int, dp_size: int
) -> list[list[dict]]:
    lanes = schedule_rows(buckets, dp_size)
    result: list[list[dict]] = [[] for _ in range(dp_size)]
    for micro_step in range(len(lanes[0])):
        template = next((samples[row[0].sample_id] for lane in lanes if (row := lane[micro_step])), None)
        for lane, rows in enumerate(lanes):
            descriptors = rows[micro_step]
            row_samples = [samples[descriptor.sample_id] for descriptor in descriptors]
            if not row_samples:
                if template is None:
                    alignment = dict(
                        input_ids=[0],
                        target_ids=[0],
                        position_ids=[0],
                        loss_mask=[False],
                        seq_lens=[1],
                        mm_kwargs=None,
                        mm_token_type_ids=None,
                    )
                else:
                    alignment = {**template, "loss_mask": [False] * len(template["input_ids"])}
                row_samples = [alignment]
            packed = next(iter(CatDataset(row_samples, capacity)))
            batch = cat_collate([packed])
            batch["num_tokens"] = sum(descriptor.length for descriptor in descriptors)
            batch["sample_ids"] = [descriptor.sample_id for descriptor in descriptors]
            assert batch["input_ids"].shape == (1, capacity)
            assert int(batch["seq_lens"].sum()) == capacity
            assert int(batch["loss_mask"].sum()) == sum(sum(sample["loss_mask"]) for sample in row_samples)
            result[lane].append(batch)
    return result


class GlobalPackedDataset(IterableDataset):
    """One ordered source cursor; worker prefetch is checkpointed by torchdata."""

    def __init__(self, dataset: SFTDataset, config: SFTDataConfig, dp_size: int):
        self.dataset = dataset
        self.config = config
        self.capacity = config.seq_len * config.micro_batch_size
        self.dp_size = dp_size
        self.num_rows = config.batch_size // config.micro_batch_size
        self.raw = dataset.dataset.add_column("__packing_row_id", list(range(dataset.num_examples)))
        self.cursor = 0
        self.filtered = 0
        self.truncated = 0
        self.epoch_valid = 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.samples: dict[tuple[int, int], dict] = {}
        self.render_seconds = 0.0
        self.source_seconds = 0.0
        self.packer = OnlinePacker(
            iter(()), self.capacity, self.num_rows, dp_size, **config.global_packing.model_dump()
        )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["dataset"] = copy.copy(self.dataset)
        state["dataset"].logger = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dataset.logger = get_logger()

    def state_dict(self) -> dict:
        return {
            "cursor": self.cursor,
            "filtered": self.filtered,
            "truncated": self.truncated,
            "epoch_valid": self.epoch_valid,
            "num_samples": dict(self.num_samples),
            "num_tokens": dict(self.num_tokens),
            "samples": dict(self.samples),
            "packer": self.packer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        for name in ("cursor", "filtered", "truncated", "epoch_valid", "samples"):
            setattr(self, name, state[name])
        self.num_samples = defaultdict(int, state["num_samples"])
        self.num_tokens = defaultdict(int, state["num_tokens"])
        self.packer.load_state_dict(state["packer"])

    def progress(self) -> dict:
        return {
            "step": self.cursor,
            "epoch": max(0, self.cursor - 1) // self.dataset.num_examples,
            "num_samples": dict(self.num_samples),
            "num_tokens": dict(self.num_tokens),
        }

    def _ordered_samples(self) -> Iterator[SampleDescriptor]:
        if not self.dataset.num_examples:
            raise ValueError("Global packing requires a nonempty dataset")
        worker_local = threading.local()

        def process(example):
            if not hasattr(worker_local, "dataset"):
                worker_local.dataset = copy.copy(self.dataset)
                resolver = self.dataset.renderers
                if isinstance(resolver, RendererResolver):
                    worker_local.dataset.renderers = RendererResolver(
                        resolver.tokenizer, resolver.config, processor=resolver.processor, columns=resolver.columns
                    )
            start = time.perf_counter()
            sample = worker_local.dataset._process(example)
            return sample, time.perf_counter() - start

        submitted = self.cursor
        shuffled_epoch = None
        shuffled = None
        futures = deque()
        finite_size = (
            self.dataset.max_epochs * self.dataset.num_examples if self.dataset.max_epochs is not None else None
        )
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            while True:
                source_start = time.perf_counter()
                while len(futures) < self.config.num_workers and (finite_size is None or submitted < finite_size):
                    epoch, row_index = divmod(submitted, self.dataset.num_examples)
                    if epoch != shuffled_epoch:
                        shuffled = (
                            self.raw.shuffle(seed=self.dataset.seed + epoch) if self.dataset.shuffle else self.raw
                        )
                        shuffled_epoch = epoch
                    example = shuffled[row_index]
                    sample_id = (epoch, example["__packing_row_id"])
                    source = example.get("__subset") or example.get("__split")
                    futures.append((sample_id, source, executor.submit(process, example)))
                    submitted += 1
                if not futures:
                    return
                sample_id, source, future = futures.popleft()
                sample, render_seconds = future.result()
                self.render_seconds += render_seconds
                self.cursor += 1
                if sample is None:
                    self.filtered += 1
                else:
                    self.epoch_valid += 1
                    self.truncated += int(len(sample["input_ids"]) > self.capacity)
                    sample = prepare_sample(sample, self.capacity)
                    num_bytes = len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                    if num_bytes > self.config.global_packing.max_sample_bytes:
                        raise ValueError(f"Sample {sample_id} exceeds max_sample_bytes ({num_bytes} bytes)")
                    self.samples[sample_id] = sample
                    self.num_samples[source] += 1
                    self.num_tokens[source] += len(sample["input_ids"])
                if self.cursor % self.dataset.num_examples == 0:
                    if not self.epoch_valid and self.dataset.max_epochs is None:
                        raise ValueError("A full dataset epoch produced no trainable samples")
                    self.epoch_valid = 0
                self.source_seconds += time.perf_counter() - source_start
                if sample is not None:
                    yield SampleDescriptor(
                        sample_id, len(sample["input_ids"]), compatibility_key(sample), num_bytes, self.packer.step
                    )

    def __iter__(self):
        source = self._ordered_samples()
        self.packer.source = source
        try:
            while True:
                start = time.perf_counter()
                self.render_seconds = 0.0
                self.source_seconds = 0.0
                cpu_before = resource.getrusage(resource.RUSAGE_SELF)
                buckets = self.packer.next_step()
                pack_seconds = time.perf_counter() - start
                if buckets is None:
                    return
                start = time.perf_counter()
                lanes = materialize_rows(buckets, self.samples, self.capacity, self.dp_size)
                for bucket in buckets:
                    for descriptor in bucket:
                        del self.samples[descriptor.sample_id]
                cpu_after = resource.getrusage(resource.RUSAGE_SELF)
                yield {
                    "step": self.packer.step - 1,
                    "lanes": lanes,
                    "progress": self.progress(),
                    "metrics": {
                        **{f"packing/{key}": value for key, value in self.packer.metrics.items()},
                        "packing/filtered_samples": self.filtered,
                        "packing/truncated_samples": self.truncated,
                        "time/broker_produce": pack_seconds,
                        "time/broker_pack": max(0.0, pack_seconds - self.source_seconds),
                        "time/broker_preprocess": self.source_seconds,
                        "time/broker_render_workers": self.render_seconds,
                        "time/broker_materialize": time.perf_counter() - start,
                        "time/broker_cpu": (
                            cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime
                        ),
                        "packing/broker_peak_rss_bytes": cpu_after.ru_maxrss * 1024,
                    },
                }
        finally:
            source.close()


class GlobalDataLoader:
    """Synchronous step delivery acknowledges consumption at checkpoint boundaries."""

    def __init__(
        self,
        dataset: GlobalPackedDataset | None,
        config: SFTDataConfig,
        cp_size: int,
        group=None,
        timeout_seconds: int = 300,
    ):
        self.world = get_world()
        self.group = group
        self.cp_size = cp_size
        self.dp_size = self.world.world_size // cp_size
        if config.batch_size % (self.dp_size * config.micro_batch_size):
            raise ValueError("Global batch size must be divisible by DP size times micro batch size")
        self.grad_accum_steps = config.batch_size // (self.dp_size * config.micro_batch_size)
        self.signature = {"config": config.model_dump(mode="json"), "world_size": self.world.world_size, "cp": cp_size}
        self.producer = (
            StatefulDataLoader(
                dataset,
                batch_size=None,
                num_workers=1,
                prefetch_factor=1,
                multiprocessing_context="spawn",
                timeout=timeout_seconds,
            )
            if self.world.is_master
            else None
        )
        self.producer_iter = None
        self.rows: deque[dict] = deque()
        self.next_step = 0
        self.dataset_progress = {"step": 0, "epoch": 0, "num_samples": {}, "num_tokens": {}}
        self.metrics = {}

    def __iter__(self):
        return self

    def __next__(self):
        if not self.rows:
            start = time.perf_counter()
            payloads = None
            if self.world.is_master:
                try:
                    if self.producer_iter is None:
                        self.producer_iter = iter(self.producer)
                    produced = next(self.producer_iter, None)
                    payloads = [
                        None
                        if produced is None
                        else {
                            "step": produced["step"],
                            "rows": produced["lanes"][rank // self.cp_size],
                            "progress": produced["progress"],
                            "metrics": produced["metrics"],
                        }
                        for rank in range(self.world.world_size)
                    ]
                except Exception as error:
                    payloads = [{"error": f"{type(error).__name__}: {error}"}] * self.world.world_size
            wait_seconds = time.perf_counter() - start
            start = time.perf_counter()
            received = [None]
            if self.world.world_size > 1:
                dist.scatter_object_list(received, payloads, src=0, group=self.group)
            else:
                received[0] = payloads[0]
            payload = received[0]
            if payload is None:
                raise StopIteration
            if "error" in payload:
                raise RuntimeError(f"Global SFT broker failed: {payload['error']}")
            if payload["step"] != self.next_step:
                raise RuntimeError(f"Expected broker step {self.next_step}, received {payload['step']}")
            if len(payload["rows"]) != self.grad_accum_steps:
                raise RuntimeError("Broker delivered an inconsistent microstep count")
            self.rows.extend(payload["rows"])
            self.next_step += 1
            self.dataset_progress = payload["progress"]
            self.metrics = {
                **payload["metrics"],
                "time/broker_wait": wait_seconds,
                "time/broker_transport": time.perf_counter() - start,
            }
        return self.rows.popleft()

    def state_dict(self) -> dict:
        if self.rows:
            raise RuntimeError("Global SFT checkpoints require a completed optimizer step")
        return {
            "global_packing": self.signature,
            "next_step": self.next_step,
            "progress": self.dataset_progress,
            "producer": self.producer.state_dict() if self.producer is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.producer_iter is not None or self.rows:
            raise RuntimeError("Restore global SFT state before starting iteration")
        if state.get("global_packing") != self.signature:
            raise ValueError("Global packing resume requires the same data config and DP/CP topology")
        self.next_step = state["next_step"]
        self.dataset_progress = state["progress"]
        if self.producer is not None:
            if state["producer"] is None:
                raise ValueError("Rank 0 requires the broker's rank-0 dataloader checkpoint")
            self.producer.load_state_dict(state["producer"])

    def close(self):
        self.producer_iter = None
        self.producer = None


def setup_global_dataloader(
    tokenizer, config: SFTDataConfig, cp_size: int, group=None, timeout_seconds: int = 300, **dataset_kwargs
):
    world = get_world()
    dataset = None
    status = [None]
    if world.is_master:
        try:
            dataset = setup_dataset(tokenizer, config, cp_size, **dataset_kwargs)
            dataset = GlobalPackedDataset(dataset, config, world.world_size // cp_size)
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if world.world_size > 1:
        dist.broadcast_object_list(status, src=0, group=group)
    if status[0] is not None:
        raise RuntimeError(f"Global SFT dataset setup failed: {status[0]}")
    return GlobalDataLoader(dataset, config, cp_size, group, timeout_seconds)
