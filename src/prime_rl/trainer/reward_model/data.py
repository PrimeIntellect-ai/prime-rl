from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

import torch
from datasets import Dataset, load_dataset
from renderers.base import Renderer, build_training_sample
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.reward_model import BradleyTerryDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import normalize_messages


class BradleyTerryBatch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    num_pairs: int
    pair_weights: Tensor


def load_bradley_terry_dataset(config: BradleyTerryDataConfig) -> Dataset:
    path = Path(config.name)
    if path.is_file():
        return cast(Dataset, load_dataset("json", data_files=str(path), split="train"))
    return cast(Dataset, load_dataset(config.name, split=config.split))


def _messages(value, default_role: str) -> list[dict]:
    return normalize_messages(value, default_role=default_role)


def render_bradley_terry_pair(example: dict, renderer: Renderer, seq_len: int) -> tuple[list[int], list[int]]:
    if example.get("prompt") is None or example.get("chosen") is None or example.get("rejected") is None:
        raise ValueError("Bradley-Terry rows require non-null 'prompt', 'chosen', and 'rejected' fields.")

    prompt = _messages(example["prompt"], "user")
    chosen = prompt + _messages(example["chosen"], "assistant")
    rejected = prompt + _messages(example["rejected"], "assistant")

    chosen_ids = list(build_training_sample(renderer, chosen, ensure_final_stop=True).token_ids)
    rejected_ids = list(build_training_sample(renderer, rejected, ensure_final_stop=True).token_ids)
    if len(chosen_ids) > seq_len:
        chosen_ids = chosen_ids[-seq_len:]
    if len(rejected_ids) > seq_len:
        rejected_ids = rejected_ids[-seq_len:]
    if not chosen_ids or not rejected_ids:
        raise ValueError("Rendered Bradley-Terry responses must contain at least one token.")
    return chosen_ids, rejected_ids


class BradleyTerryDataset(Stateful, IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        renderer: Renderer,
        config: BradleyTerryDataConfig,
        max_epochs: int | None = None,
        pad_to_data_world_size: bool = False,
    ):
        self.dataset = dataset
        self.renderer = renderer
        self.config = config
        self.max_epochs = max_epochs
        self.pad_to_data_world_size = pad_to_data_world_size
        self.step = 0
        self.epoch = 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        worker_info = get_worker_info()
        worker_id, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)
        self.data_rank = get_world().rank * num_workers + worker_id
        self.data_world_size = get_world().world_size * num_workers

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]

    def __iter__(self):
        num_examples = len(self.dataset)
        if num_examples == 0:
            raise ValueError("Bradley-Terry datasets must contain at least one example.")
        epoch_size = num_examples
        if self.pad_to_data_world_size:
            epoch_size = ((num_examples + self.data_world_size - 1) // self.data_world_size) * self.data_world_size
        dataset = self.dataset.shuffle(seed=self.config.seed + self.epoch) if self.config.shuffle else self.dataset
        while True:
            epoch = self.step // epoch_size
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.config.seed + epoch) if self.config.shuffle else self.dataset
            epoch_index = self.step % epoch_size
            self.step += 1
            if epoch_index % self.data_world_size != self.data_rank:
                continue
            sample_weight = epoch_index < num_examples
            index = epoch_index % num_examples
            chosen_ids, rejected_ids = render_bradley_terry_pair(
                cast(dict, dataset[index]), self.renderer, self.config.seq_len
            )
            if sample_weight:
                self.num_samples["bradley_terry"] += 1
                self.num_tokens["bradley_terry"] += len(chosen_ids) + len(rejected_ids)
            yield {
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
                "sample_weight": sample_weight,
            }


def bradley_terry_collate(samples: list[dict], pad_token_id: int) -> BradleyTerryBatch:
    sequences = [sample[side] for side in ("chosen_ids", "rejected_ids") for sample in samples]
    max_len = max(map(len, sequences))
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "num_pairs": len(samples),
        "pair_weights": torch.tensor([sample.get("sample_weight", True) for sample in samples], dtype=torch.float32),
    }


def setup_dataset(
    config: BradleyTerryDataConfig,
    renderer: Renderer,
    *,
    raw_dataset: Dataset | None = None,
    max_epochs: int | None = None,
    pad_to_data_world_size: bool = False,
) -> BradleyTerryDataset:
    return BradleyTerryDataset(
        raw_dataset if raw_dataset is not None else load_bradley_terry_dataset(config),
        renderer,
        config,
        max_epochs=max_epochs,
        pad_to_data_world_size=pad_to_data_world_size,
    )


def setup_dataloader(
    dataset: BradleyTerryDataset,
    config: BradleyTerryDataConfig,
    pad_token_id: int,
) -> StatefulDataLoader:
    return StatefulDataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        collate_fn=lambda samples: bradley_terry_collate(samples, pad_token_id),
    )
