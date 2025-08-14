from collections import defaultdict
from typing import Iterator, TypedDict

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import DataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class Sample(TypedDict):
    epoch: int # TODO: Argh, can we find a way to export epoch metainformation in a nicer way?
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]


class Batch(TypedDict):
    epoch: int
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]


class FakeDataset(IterableDataset):
    """A dataset of fake tokens"""

    def __init__(self, tokenizer: AutoTokenizer, seq_len: int):
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size

    def __iter__(self) -> Iterator[Sample]:
        while True:
            rand_seq_len = torch.randint(1, self.seq_len + 1, (1,)).item()
            # simulate different sequence lengths
            input_ids = torch.randint(0, self.vocab_size, (rand_seq_len + 1,)).long().tolist()
            position_ids = torch.arange(len(input_ids)).long()
            loss_mask = torch.ones(len(input_ids)).bool()
            loss_mask[-1] = 0
            yield {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "target_ids": input_ids[1:] + [0],
                "epoch": 0,
            }


class SFTDataset(IterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt + completion format."""

    def __init__(self, tokenizer: AutoTokenizer, name: str, split: str):
        self.tokenizer = tokenizer
        self._logger = get_logger()

        # Load dataset
        self.dataset: HFDataset = load_dataset(name, split=split)

        # Assert that the dataset has a 'text' column
        if "prompt" not in self.dataset.column_names or "completion" not in self.dataset.column_names:
            raise ValueError("HF dataset must have a 'prompt' and 'completion' column for SFT")

        # Get the data rank and world size
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        self.data_rank = get_world().rank * num_workers + worker_id
        self.data_world_size = get_world().world_size * num_workers

    def __iter__(self) -> Iterator[Sample]:
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        counter, epoch = -1, -1
        while True:
            epoch += 1
            for example in self.dataset:
                # Increment the counter (0, 1, ...)
                counter += 1

                # Skip samples that don't belong to this data rank
                if counter % self.data_world_size != self.data_rank:
                    continue

                assert "prompt" in example and "completion" in example, (
                    "Prompt and completion must be present in the example"
                )
                assert isinstance(example["prompt"], list) and isinstance(example["completion"], list), (
                    "Prompt and completion must be lists"
                )

                prompt_ids = self.tokenizer.apply_chat_template(
                    example["prompt"],
                    tools=example.get("tools"),
                    **example.get("chat_template_kwargs", {}),
                )
                prompt_completion_ids = self.tokenizer.apply_chat_template(
                    example["prompt"] + example["completion"],
                    tools=example.get("tools"),
                    **example.get("chat_template_kwargs", {}),
                )

                if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                    self._logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )

                # Create sample (with one fake target for the last token)
                sample = {
                    "input_ids": prompt_completion_ids,
                    "position_ids": list(range(len(prompt_completion_ids))),
                    "loss_mask": [False] * len(prompt_ids) + [True] * (len(prompt_completion_ids) - len(prompt_ids) - 1) + [False],
                    "target_ids": prompt_completion_ids[1:] + [0],
                    "epoch": epoch,
                }

                yield sample


class PackingDataset(IterableDataset):
    """A dataset that packs samples into a single sequence."""

    def __init__(self, dataset: SFTDataset, seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[Sample]:
        packed_samples, seq_len = defaultdict(list), 0
        for sample in self.dataset:
            # Add sample to packed samples
            for key, value in sample.items():
                if key == "epoch":
                    packed_samples[key] = min(packed_samples.get(key, float("inf")), value)
                else:
                    packed_samples[key].extend(value)

            # Update sequence length
            seq_len += len(sample["input_ids"])

            # If batch is full, truncate and yield it
            if seq_len >= self.seq_len:
                for key, value in packed_samples.items():
                    if isinstance(value, list):
                        packed_samples[key] = value[: self.seq_len]
                yield packed_samples
                packed_samples, seq_len = defaultdict(list), 0


class PaddingDataset(IterableDataset):
    """A dataset that pads samples to a fixed sequence length."""

    def __init__(self, dataset: SFTDataset, seq_len: int, pad_token_id: int):
        self.dataset = dataset
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def __iter__(self) -> Iterator[Sample]:
        for sample in self.dataset:
            if len(sample["input_ids"]) < self.seq_len: # Pad
                num_padding_tokens = self.seq_len - len(sample["input_ids"])
                sample["input_ids"] = sample["input_ids"] + [self.pad_token_id] * num_padding_tokens
                sample["loss_mask"] = sample["loss_mask"] + [0] * num_padding_tokens
                sample["position_ids"] = sample["position_ids"] + [0] * num_padding_tokens
                sample["target_ids"] = sample["target_ids"] + [self.pad_token_id] * num_padding_tokens

            # Truncate if too long
            sample["input_ids"] = sample["input_ids"][: self.seq_len]
            sample["loss_mask"] = sample["loss_mask"][: self.seq_len]
            sample["position_ids"] = sample["position_ids"][: self.seq_len]
            sample["target_ids"] = sample["target_ids"][: self.seq_len]

            yield sample

def collate(samples: list[Sample]) -> Batch:
    return {
        "input_ids": torch.stack([torch.tensor(sample["input_ids"]) for sample in samples], dim=0).long(),
        "position_ids": torch.stack([torch.tensor(sample["position_ids"]) for sample in samples], dim=0).long(),
        "loss_mask": torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples], dim=0).bool(),
        "target_ids": torch.stack([torch.tensor(sample["target_ids"]) for sample in samples], dim=0).long(),
        "epoch": min([sample["epoch"] for sample in samples]),
    }

def setup_dataset(tokenizer: AutoTokenizer, config: DataConfig) -> IterableDataset:
    if config.fake:
        return FakeDataset(tokenizer, config.seq_len)
    return SFTDataset(tokenizer, name=config.name, split=config.split)

def setup_dataloader(dataset: IterableDataset, tokenizer: AutoTokenizer, config: DataConfig) -> DataLoader:
    seq_len = config.micro_batch_size * config.seq_len if config.collate_mode == "packing" else config.seq_len
    if config.collate_mode == "packing":
        packing_dataset = PackingDataset(dataset, seq_len)
        return DataLoader(packing_dataset, batch_size=1, collate_fn=collate)
    padding_dataset = PaddingDataset(dataset, seq_len, tokenizer.pad_token_id)
    return DataLoader(padding_dataset, batch_size=config.micro_batch_size, collate_fn=collate)
