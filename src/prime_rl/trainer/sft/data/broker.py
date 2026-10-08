import copy
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from itertools import count
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.trainer.sft.data.dataset import SFTDataset
from prime_rl.trainer.sft.data.packing import OnlinePacker, SampleDescriptor, schedule_rows
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class RenderedSample(NamedTuple):
    position: int
    source: int
    tokens: torch.Tensor


class StridedShard(Sampler[int]):
    """Yield this rank's global stream positions, starting at the resume cursor."""

    def __init__(self, rank: int, world_size: int, stop: int | None):
        self.rank = rank
        self.world_size = world_size
        self.stop = stop
        self.position = 0

    def __iter__(self):
        start = self.position + (self.rank - self.position) % self.world_size
        return count(start, self.world_size) if self.stop is None else iter(range(start, self.stop, self.world_size))


class RenderedDataset(Dataset):
    """Render global shuffled-stream positions, retaining filtered examples as empty records."""

    def __init__(self, dataset: SFTDataset, capacity: int):
        self.dataset = dataset
        self.capacity = capacity
        self.cached_epoch = None
        self.shuffled = None
        sources = {
            value
            for column in ("__subset", "__split")
            if column in dataset.dataset.column_names
            for value in dataset.dataset.data.column(column).unique().to_pylist()
            if value is not None
        }
        self.sources = [None, *sorted(sources)]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["dataset"] = copy.copy(self.dataset)
        state["dataset"].logger = None
        return state

    def __getitem__(self, position: int) -> RenderedSample:
        self.dataset.logger = get_logger()
        epoch, index = divmod(position, self.dataset.num_examples)
        if epoch != self.cached_epoch:
            self.shuffled = (
                self.dataset.dataset.shuffle(seed=self.dataset.seed + epoch, keep_in_memory=True)
                if self.dataset.shuffle
                else self.dataset.dataset
            )
            self.cached_epoch = epoch
        example = self.shuffled[index]
        source = self.sources.index(example.get("__subset") or example.get("__split"))
        sample = self.dataset._process(example)
        tokens = torch.empty((4, 0), dtype=torch.int64)
        if sample is not None:
            if sample["mm_kwargs"] is not None or sample["mm_token_type_ids"] is not None:
                raise ValueError("Distributed packing requires text samples")
            tokens = torch.tensor(
                [sample[key][: self.capacity] for key in ("input_ids", "target_ids", "position_ids", "loss_mask")],
                dtype=torch.int64,
            )
            if not tokens[3].any():
                tokens = torch.empty((4, 0), dtype=torch.int64)
        return RenderedSample(position, source, tokens)


def materialize_rows(rows: list[list[SampleDescriptor]], samples: dict[int, torch.Tensor], capacity: int) -> list[dict]:
    batches = []
    for row in rows:
        packed = torch.zeros((4, capacity), dtype=torch.int64)
        offset = 0
        lengths = []
        for sample in row:
            packed[:, offset : offset + sample.length] = samples[sample.position]
            offset += sample.length
            lengths.append(sample.length)
        packed[2, offset:] = torch.arange(capacity - offset)
        if lengths:
            lengths[-1] += capacity - offset
        else:
            lengths = [capacity]
        batches.append(
            dict(
                input_ids=packed[0].unsqueeze(0),
                target_ids=packed[1].unsqueeze(0),
                position_ids=packed[2].unsqueeze(0),
                loss_mask=packed[3].bool().unsqueeze(0),
                seq_lens=torch.tensor(lengths, dtype=torch.int64),
                mm_kwargs=None,
                mm_token_type_ids=None,
                num_tokens=offset,
                sample_ids=[sample.position for sample in row],
            )
        )
    return batches


class PackedDataLoader:
    """One CPU-prefetched global step, on a loader-owned Gloo group independent of model collectives.

    Construct and close on the main thread, in the same order on every rank.
    Only the producer thread uses this group; workers never perform collectives.
    No producer operation uses CUDA or a model process group. Closing drains the
    in-flight step before destroying the group. Train and validation loaders own
    different groups, so validation cannot reorder the training prefetch stream.
    Checkpoints use the last fully consumed step's cursor, never the producer's
    speculative cursor. Rendering buffers are reconstructed on resume.
    Validation rounds packing capacity up to complete DP groups, including its padded tail.
    """

    def __init__(
        self,
        dataset: SFTDataset,
        config: SFTDataConfig,
        cp_size: int = 1,
        timeout_seconds: int = 300,
        *,
        validation: bool = False,
    ):
        world = get_world()
        self.rank, self.world_size = world.rank, world.world_size
        if self.world_size % cp_size:
            raise ValueError("World size must be divisible by CP size")
        self.cp_size = cp_size
        self.dp_size = self.world_size // cp_size
        self.num_rows = config.batch_size // config.micro_batch_size
        if validation:
            self.num_rows = (self.num_rows + self.dp_size - 1) // self.dp_size * self.dp_size
        elif self.num_rows % self.dp_size:
            raise ValueError("Global batch size must be divisible by DP size times micro batch size")
        if not dataset.num_examples and dataset.max_epochs is None:
            raise ValueError("Training requires a nonempty dataset")
        self.config = config
        self.capacity = config.seq_len * config.micro_batch_size
        self.source = RenderedDataset(dataset, self.capacity)
        stop = None if dataset.max_epochs is None else dataset.num_examples * dataset.max_epochs
        self.sampler = StridedShard(self.rank, self.world_size, stop)
        self.group = (
            dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds)) if self.world_size > 1 else None
        )
        self.loader = DataLoader(
            self.source,
            sampler=self.sampler,
            batch_size=None,
            num_workers=config.num_workers,
            prefetch_factor=2,
            in_order=True,
            multiprocessing_context="spawn",
            timeout=timeout_seconds,
            generator=torch.Generator().manual_seed(config.seed),
        )
        self.source_iter = None
        self.executor = None
        self.future = None
        self.closed = False
        self.exhausted = False
        self.position = 0
        self.epoch_started = True
        self.epoch_has_samples = False
        self.chunk_start = 0
        self.metadata = []
        self.samples: dict[int, torch.Tensor] = {}
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.rows = deque()
        self.step_progress = None
        self.dataset_progress = {"step": 0, "epoch": 0, "num_samples": {}, "num_tokens": {}}
        self.metrics = {}

    def __iter__(self):
        return self

    def _read_chunk(self):
        self.chunk_start = self.position
        chunk_size = self.config.packing.chunk_size
        metadata = torch.full((chunk_size, 2), -1, dtype=torch.int64)
        first = self.position + (self.rank - self.position) % self.world_size
        for index in range(chunk_size):
            sample = next(self.source_iter, None)
            if sample is None:
                break
            if sample.position != first + index * self.world_size:
                raise RuntimeError("Rendered shard is not in global stream order")
            length = sample.tokens.shape[1]
            metadata[index] = torch.tensor([length, sample.source])
            if length:
                self.samples[sample.position] = sample.tokens
        gathered = [torch.empty_like(metadata) for _ in range(self.world_size)]
        if self.group is not None:
            dist.all_gather(gathered, metadata, group=self.group)
        else:
            gathered[0] = metadata
        gathered = [part.tolist() for part in gathered]
        self.metadata = [
            gathered[(self.position + offset) % self.world_size][offset // self.world_size]
            for offset in range(chunk_size * self.world_size)
        ]

    def _exchange(self, rows: list[list[list[SampleDescriptor]]]) -> list[dict]:
        outgoing = [
            [
                sample
                for row in rows[rank // self.cp_size]
                for sample in row
                if sample.position % self.world_size == self.rank
            ]
            for rank in range(self.world_size)
        ]
        own_rows = rows[self.rank // self.cp_size]
        incoming = [
            [sample for row in own_rows for sample in row if sample.position % self.world_size == rank]
            for rank in range(self.world_size)
        ]
        send_splits = [sum(sample.length * 4 for sample in part) for part in outgoing]
        recv_splits = [sum(sample.length * 4 for sample in part) for part in incoming]
        payloads = [self.samples[sample.position].flatten() for part in outgoing for sample in part]
        send = torch.cat(payloads) if payloads else torch.empty(0, dtype=torch.int64)
        received = torch.empty(sum(recv_splits), dtype=torch.int64)
        if self.group is not None:
            dist.all_to_all_single(received, send, recv_splits, send_splits, group=self.group)
        else:
            received.copy_(send)
        samples = {}
        offset = 0
        for part in incoming:
            for sample in part:
                end = offset + 4 * sample.length
                samples[sample.position] = received[offset:end].view(4, sample.length)
                offset = end
        for position in {sample.position for part in outgoing for sample in part}:
            del self.samples[position]
        return materialize_rows(own_rows, samples, self.capacity)

    def _produce_step(self):
        start = time.perf_counter()
        gather_seconds = 0.0
        packer = OnlinePacker(self.capacity, self.num_rows)
        size = self.source.dataset.num_examples
        limit = None if self.source.dataset.max_epochs is None else size * self.source.dataset.max_epochs
        while not packer.full and (limit is None or self.position < limit):
            if self.position >= self.chunk_start + len(self.metadata):
                gather_start = time.perf_counter()
                self._read_chunk()
                gather_seconds += time.perf_counter() - gather_start
            length, source = self.metadata[self.position - self.chunk_start]
            if length < 0:
                raise RuntimeError("Rendered shard ended before the global stream")
            if length and not packer.add(SampleDescriptor(self.position, length)):
                break
            if length:
                name = self.source.sources[source]
                self.num_samples[name] += 1
                self.num_tokens[name] += length
                self.epoch_has_samples = True
            self.position += 1
            if self.position % size == 0:
                if limit is None and self.epoch_started and not self.epoch_has_samples:
                    raise ValueError("A full dataset epoch produced no trainable samples")
                self.epoch_started = True
                self.epoch_has_samples = False
        progress = {
            "step": self.position,
            "epoch": max(0, self.position - 1) // size if size else 0,
            "num_samples": dict(self.num_samples),
            "num_tokens": dict(self.num_tokens),
        }
        transport_start = time.perf_counter()
        batches = self._exchange(schedule_rows(packer.rows, self.dp_size)) if any(packer.rows) else []
        return (
            batches,
            progress,
            {
                "time/broker_produce": time.perf_counter() - start,
                "time/broker_gather": gather_seconds,
                "time/broker_transport": time.perf_counter() - transport_start,
            },
        )

    def __next__(self):
        if self.closed or self.exhausted:
            raise StopIteration
        if self.executor is None:
            self.source_iter = iter(self.loader)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sft-packing")
            self.future = self.executor.submit(self._produce_step)
        if not self.rows:
            start = time.perf_counter()
            rows, self.step_progress, metrics = self.future.result()
            self.metrics = {**metrics, "time/broker_wait": time.perf_counter() - start}
            if not rows:
                self.dataset_progress = self.step_progress
                self.exhausted = True
                raise StopIteration
            self.rows.extend(rows)
            self.future = self.executor.submit(self._produce_step)
        batch = self.rows.popleft()
        if torch.cuda.is_available():
            batch = {
                key: value.pin_memory() if isinstance(value, torch.Tensor) else value for key, value in batch.items()
            }
        if not self.rows:
            self.dataset_progress = self.step_progress
        return batch

    def state_dict(self) -> dict:
        if self.rows:
            raise RuntimeError("Packing checkpoints require a completed optimizer step")
        return copy.deepcopy({"progress": self.dataset_progress})

    def load_state_dict(self, state: dict) -> None:
        if self.executor is not None or self.closed:
            raise RuntimeError("Restore packing state before starting iteration")
        self.dataset_progress = copy.deepcopy(state["progress"])
        self.position = self.dataset_progress["step"]
        self.sampler.position = self.position
        self.epoch_started = self.position % max(1, self.source.dataset.num_examples) == 0
        self.num_samples.update(self.dataset_progress["num_samples"])
        self.num_tokens.update(self.dataset_progress["num_tokens"])

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.future.result()
        finally:
            if self.source_iter is not None:
                self.source_iter._shutdown_workers()
            if self.group is not None:
                dist.destroy_process_group(self.group)
