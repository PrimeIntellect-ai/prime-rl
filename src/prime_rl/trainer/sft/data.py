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
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]


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

        self._init_process_id()

    def _init_process_id(self):
        world = get_world()
        rank = world.rank
        world_size = world.world_size

        # Get dataloader worker info
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        self.world_size = world_size * num_workers
        self.rank = rank * num_workers + worker_id

    def __iter__(self) -> Iterator[Sample]:
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """

        counter = 0
        while True:
            for example in self.dataset:
                counter += 1

                # Skip samples that don't belong to this process
                if counter % self.world_size != self.rank:
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

                # Create sample
                # we still want to keep a power of 2 for the sequence length so adding a fake target for the last token
                sample = {
                    "input_ids": prompt_completion_ids,
                    "position_ids": list(range(len(prompt_completion_ids))),
                    "loss_mask": [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids) - 1) + [0],
                    "target_ids": prompt_completion_ids[1:] + [self.tokenizer.pad_token_id],
                }

                yield sample


class PackingDataset(IterableDataset):
    """A dataset that packs samples into a single sequence."""

    def __init__(self, dataset: SFTDataset, seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[Sample]:
        current_samples = defaultdict(list)
        current_seq_len = 0

        for sample in self.dataset:
            for key, value in sample.items():
                current_samples[key].extend(value)

            current_seq_len += len(sample["input_ids"])

            if current_seq_len >= self.seq_len:
                batch = {}
                for key, value in current_samples.items():
                    batch[key] = torch.tensor(value[: self.seq_len])

                current_samples = defaultdict(list)
                current_seq_len = 0
                yield batch


class PaddingDataset(IterableDataset):
    """A dataset that pads samples to a fixed sequence length."""

    def __init__(self, dataset: SFTDataset, seq_len: int, pad_token_id: int):
        self.dataset = dataset
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def __iter__(self) -> Iterator[Sample]:
        for sample in self.dataset:
            if len(sample["input_ids"]) < self.seq_len:
                padding_len = self.seq_len - len(sample["input_ids"])
                sample["input_ids"] = sample["input_ids"] + [self.pad_token_id] * padding_len
                sample["loss_mask"] = sample["loss_mask"] + [0] * padding_len
                sample["position_ids"] = sample["position_ids"] + [0] * padding_len
                sample["target_ids"] = sample["target_ids"] + [self.pad_token_id] * padding_len

            for key, value in sample.items():
                sample[key] = torch.tensor(value[: self.seq_len])

            yield sample


def get_dataloader(tokenizer: AutoTokenizer, config: DataConfig) -> DataLoader:
    if config.collate_mode == "packing":
        seq_len = config.micro_batch_size * config.seq_len
    else:
        seq_len = config.seq_len

    if config.fake:
        dataset = FakeDataset(tokenizer, seq_len)
    else:
        dataset = SFTDataset(tokenizer, config.name, config.split)

    if config.collate_mode == "packing":
        packing_dataset = PackingDataset(dataset, seq_len)
        return DataLoader(packing_dataset, batch_size=1)

    else:
        padding_dataset = PaddingDataset(dataset, seq_len, tokenizer.pad_token_id)
        return DataLoader(padding_dataset, batch_size=config.micro_batch_size)
