from collections import defaultdict
from typing import Any, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Int
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.supervised import BaseDataConfig
from prime_rl.utils.logger import get_logger


class Sample(TypedDict):
    input_ids: Int[np.ndarray, "seq"]
    position_ids: Int[np.ndarray, "seq"]
    loss_mask: Bool[np.ndarray, "seq"]
    target_ids: Int[np.ndarray, "seq"]
    seq_lens: list[int]
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[np.ndarray, "seq"] | None


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    seq_lens: Int[Tensor, "packed"]
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[Tensor, "batch seq"] | None


class StatefulIterableDataset(Stateful, IterableDataset):
    """Iterable (infinite) and stateful (checkpointable) dataset, read by data rank ``dp_rank * num_workers +
    worker_id`` of ``dp_size * num_workers`` (`ParallelDims.data_parallel`)."""

    def __init__(self, dp_rank: int = 0, dp_size: int = 1):
        self.step, self.epoch = 0, 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.fast_forward = False
        self.dp_rank, self.dp_size = dp_rank, dp_size
        self._setup_world_info()

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.fast_forward = True
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]

    def _setup_world_info(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        self.data_rank = self.dp_rank * num_workers + worker_id
        self.data_world_size = self.dp_size * num_workers


class CatDataset(StatefulIterableDataset):
    """Pack samples into rows of at most `seq_len` tokens, padded to a multiple of `pad_to_multiple_of`."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int, pad_to_multiple_of: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pending_sample: Sample | None = None

    def state_dict(self) -> dict:
        state = {
            "dataset": self.dataset.state_dict(),
            "progress": {
                "num_samples": dict(self.dataset.num_samples),
                "num_tokens": dict(self.dataset.num_tokens),
            },
        }
        if self.pending_sample is not None:
            state["pending_sample"] = self.pending_sample
        return state

    def load_state_dict(self, state_dict: dict):
        progress = state_dict.get("progress", {})
        self.dataset.num_samples.update(progress.get("num_samples", {}))
        self.dataset.num_tokens.update(progress.get("num_tokens", {}))
        self.dataset.load_state_dict(state_dict["dataset"])
        self.pending_sample = state_dict.get("pending_sample")

    def __iter__(self):
        packed_samples = defaultdict(list)
        packed_samples["mm_kwargs"] = None
        packed_samples["mm_token_type_ids"] = None
        seq_len = 0

        pending_sample = self.pending_sample
        self.pending_sample = None

        def samples():
            if pending_sample is not None:
                yield pending_sample
            yield from self.dataset

        for sample in samples():
            sample_len = len(sample["input_ids"])
            would_overflow = seq_len + sample_len > self.seq_len
            if seq_len > 0 and would_overflow:
                self.pending_sample = sample
                yield self._finalize_pack(packed_samples, self.seq_len)
                self.pending_sample = None
                packed_samples = defaultdict(list)
                packed_samples["mm_kwargs"] = None
                packed_samples["mm_token_type_ids"] = None
                seq_len = 0

            for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):
                packed_samples[key].append(sample[key])
            packed_samples["seq_lens"].extend(sample["seq_lens"])

            sample_mm_kwargs = sample.get("mm_kwargs")
            sample_mm_type_ids = sample.get("mm_token_type_ids")
            if sample_mm_kwargs is None:
                if packed_samples["mm_token_type_ids"] is not None:
                    packed_samples["mm_token_type_ids"].append(np.zeros(sample_len, dtype=np.int64))
            else:
                if packed_samples["mm_kwargs"] is not None and (
                    (packed_samples["mm_token_type_ids"] is None) != (sample_mm_type_ids is None)
                ):
                    raise ValueError("Cannot pack multimodal samples with mixed mm_token_type_ids")

                if packed_samples["mm_kwargs"] is None:
                    packed_samples["mm_kwargs"] = dict(sample_mm_kwargs)
                else:
                    if packed_samples["mm_kwargs"].keys() != sample_mm_kwargs.keys():
                        raise ValueError("Cannot pack multimodal samples with different mm_kwargs keys")
                    for key, value in sample_mm_kwargs.items():
                        packed_samples["mm_kwargs"][key] = torch.cat([packed_samples["mm_kwargs"][key], value], dim=0)

                if packed_samples["mm_token_type_ids"] is None and sample_mm_type_ids is not None:
                    packed_samples["mm_token_type_ids"] = [np.zeros(seq_len, dtype=np.int64)]
                if packed_samples["mm_token_type_ids"] is not None:
                    packed_samples["mm_token_type_ids"].append(sample_mm_type_ids)

            seq_len += sample_len

            if seq_len >= self.seq_len:
                yield self._finalize_pack(packed_samples, self.seq_len)
                packed_samples = defaultdict(list)
                packed_samples["mm_kwargs"] = None
                packed_samples["mm_token_type_ids"] = None
                seq_len = 0

        if seq_len > 0:
            yield self._finalize_pack(packed_samples, self.seq_len)

    def _finalize_pack(self, packed: dict[str, Any], seq_len: int) -> dict:
        result: dict[str, Any] = {
            k: np.concatenate(packed[k])[:seq_len] for k in ("input_ids", "position_ids", "loss_mask", "target_ids")
        }
        result["seq_lens"] = []
        remaining = len(result["input_ids"])
        for sample_len in packed["seq_lens"]:
            if remaining <= 0:
                break
            kept = min(sample_len, remaining)
            if kept > 0:
                result["seq_lens"].append(kept)
            remaining -= kept
        pad_len = -len(result["input_ids"]) % self.pad_to_multiple_of
        if pad_len > 0:
            # Identical pad tokens route to the same MoE experts and can overload one EP rank, so text packs pad with
            # their own tokens. Multimodal packs keep a constant id: a repeated image placeholder would have no feature
            # to receive.
            if packed["mm_kwargs"] is None:
                padding = np.resize(result["input_ids"], pad_len)
            else:
                padding = np.zeros(pad_len, dtype=np.int64)
            result["input_ids"] = np.concatenate([result["input_ids"], padding])
            result["position_ids"] = np.concatenate([result["position_ids"], np.arange(pad_len)])
            result["loss_mask"] = np.concatenate([result["loss_mask"], np.zeros(pad_len, dtype=bool)])
            result["target_ids"] = np.concatenate([result["target_ids"], np.zeros(pad_len, dtype=np.int64)])
            result["seq_lens"][-1] += pad_len
        result["mm_kwargs"] = packed["mm_kwargs"]
        if packed["mm_token_type_ids"] is not None:
            type_ids = np.concatenate(packed["mm_token_type_ids"])[:seq_len]
            result["mm_token_type_ids"] = np.concatenate([type_ids, np.zeros(pad_len, dtype=np.int64)])
        else:
            result["mm_token_type_ids"] = None
        return result


def cat_collate(samples: list[Sample]) -> Batch:
    # CPU tensors only: this runs in dataloader workers then the trainer moves batches to the GPU with async copies from pinned memory
    (sample,) = samples
    mm_kwargs = sample.get("mm_kwargs")
    mm_token_type_ids = sample.get("mm_token_type_ids")
    return {
        "input_ids": torch.from_numpy(sample["input_ids"]).unsqueeze(0),
        "position_ids": torch.from_numpy(sample["position_ids"]).unsqueeze(0),
        "loss_mask": torch.from_numpy(sample["loss_mask"]).unsqueeze(0),
        "target_ids": torch.from_numpy(sample["target_ids"]).unsqueeze(0),
        "seq_lens": torch.tensor(sample["seq_lens"], dtype=torch.long),
        "mm_kwargs": dict(mm_kwargs) if mm_kwargs is not None else None,
        "mm_token_type_ids": torch.from_numpy(mm_token_type_ids).unsqueeze(0)
        if mm_token_type_ids is not None
        else None,
    }


def setup_dataloader(dataset: StatefulIterableDataset, config: BaseDataConfig, cp: int) -> StatefulDataLoader:
    row_len = config.seq_len * config.micro_batch_size
    # Rows hold at most `row_len` tokens, so padding to a multiple of it pads every row to exactly `row_len`
    pad_to_multiple_of = row_len if config.pad_to_seq_len else cp
    packing_dataset = CatDataset(dataset, row_len, pad_to_multiple_of=pad_to_multiple_of)
    return StatefulDataLoader(
        packing_dataset,
        batch_size=1,
        collate_fn=cat_collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def get_dataset_state(dataloader: StatefulDataLoader) -> dict:
    """Dataset position per worker for the startup log, parsed from ``StatefulDataLoader.state_dict()``.

    The loader is the only source that is correct in every case: after a resume's
    ``load_state_dict`` the restored position exists solely in the loader's stashed
    state (it reaches the dataset copies inside workers when the iterator forks them;
    the main-process dataset object stays at position zero). The keys are torchdata's
    private worker-snapshot layout."""
    snapshots = dataloader.state_dict()["_snapshot"]["_worker_snapshots"]
    return {wid: snap["dataset_state"]["dataset"] for wid, snap in sorted(snapshots.items())}


def get_dataset_progress(dataloader: StatefulDataLoader) -> dict:
    """Dataset position and aggregate counters from dataloader workers."""
    snapshot = dataloader.state_dict()["_snapshot"]
    worker_snapshots = snapshot["_worker_snapshots"]
    positions = [worker_snapshot["dataset_state"]["dataset"] for worker_snapshot in worker_snapshots.values()]
    furthest = max(positions, key=lambda position: position["step"])
    num_samples = defaultdict(int)
    num_tokens = defaultdict(int)
    for worker_snapshot in worker_snapshots.values():
        progress = worker_snapshot["dataset_state"].get("progress", {})
        for name, count in progress.get("num_samples", {}).items():
            num_samples[name] += count
        for name, count in progress.get("num_tokens", {}).items():
            num_tokens[name] += count
    return {
        **furthest,
        "num_samples": dict(num_samples),
        "num_tokens": dict(num_tokens),
    }
