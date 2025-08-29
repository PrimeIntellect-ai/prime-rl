from collections import defaultdict
from typing import Iterator, TypedDict

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.config import DataConfigType, FakeDataConfig, RealDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class Sample(TypedDict):
    epoch: int  # TODO: Argh, can we find a way to export epoch metainformation in a nicer way?
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


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self):
        self.step, self.epoch = -1, 0
        self._setup_world_info()
        self._logger = get_logger()

    def state_dict(self) -> dict:
        # +1 because the stateful dataloader expects uses 1-based counting while we start at 0
        return {"step": self.step + 1, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        # -1 because the stateful dataloader expects uses 1-based counting while we start at 0
        self.step = state_dict["step"] - 1
        self.epoch = state_dict["epoch"]

    def _setup_world_info(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        self.data_rank = get_world().rank * num_workers + worker_id
        self.data_world_size = get_world().world_size * num_workers


class FakeDataset(StatefulIterableDataset):
    """A dataset of fake tokens"""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: FakeDataConfig):
        super().__init__()
        self.config = config
        self.vocab_size = tokenizer.vocab_size
        self.num_examples = config.num_examples

    def __iter__(self) -> Iterator[Sample]:
        while True:
            # Increment the step counter (0, 1, 2, ...)
            # This has to be done before yielding the sample for the dataloader to checkpoint correctly
            self.step += 1

            # Skip samples that don't belong to this data rank
            if self.step % self.data_world_size != self.data_rank:
                continue

            # Update epoch if num_examples is set
            if self.num_examples is not None:
                self.epoch = self.step // self.num_examples

            seq_len = (
                int(torch.randint(1, self.config.seq_len, (1,)).item())
                if self.config.length == "variable"
                else self.config.seq_len
            )
            input_ids = (
                [self.step] * (seq_len + 1)
                if self.config.input_ids == "increasing"
                else torch.randint(0, self.vocab_size, (seq_len + 1,)).long().tolist()
            )
            position_ids = list(range(seq_len))
            loss_mask = [True] * seq_len
            fake_sample = {
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:],
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "epoch": self.epoch,
            }
            yield fake_sample


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt + completion format."""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: RealDataConfig):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Load dataset
        self.dataset: Dataset = concatenate_datasets(
            [load_dataset(config.name, split=split) for split in config.splits]
        )

        # Assert that the dataset has a 'text' column
        if "prompt" not in self.dataset.column_names or "completion" not in self.dataset.column_names:
            raise ValueError("HF dataset must have a 'prompt' and 'completion' column for SFT")

        # If specified, select a subset of the dataset
        if config.num_examples is not None:
            self.dataset = self.dataset.select(range(config.num_examples))

        # Store the number of examples in the dataset
        self.num_examples = len(self.dataset)

    def __iter__(self) -> Iterator[Sample]:
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        dataset = self.dataset.shuffle(seed=self.epoch) if self.config.shuffle else self.dataset
        while True:
            # Increment the step counter (0, 1, 2, ...)
            # This has to be done before yielding the sample for the dataloader to checkpoint correctly
            self.step += 1

            # Get example from dataset
            example = dataset[self.step % self.num_examples]

            # Skip samples that don't belong to this data rank
            if self.step % self.data_world_size != self.data_rank:
                continue

            # Compute current epoch based on step count (total samples seen)
            epoch = self.step // self.num_examples

            # Update stored epoch if new epoch is reached, optionall shuffle
            if epoch > self.epoch:
                dataset = self.dataset.shuffle(seed=self.epoch) if self.config.shuffle else self.dataset
                self.epoch = epoch

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
                "loss_mask": [False] * len(prompt_ids)
                + [True] * (len(prompt_completion_ids) - len(prompt_ids) - 1)
                + [False],
                "target_ids": prompt_completion_ids[1:] + [0],
                "epoch": self.epoch,
            }

            yield sample


class PackingDataset(StatefulIterableDataset):
    """A dataset that packs samples into a single sequence."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len
        # Public state attributes for checkpointing
        self.packed_samples = defaultdict(list)
        self.current_seq_len = 0

    def state_dict(self) -> dict:
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict)

    def __iter__(self) -> Iterator[Sample]:
        packed_samples, seq_len = self.packed_samples, self.current_seq_len
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


def collate(samples: list[Sample]) -> Batch:
    return {
        "input_ids": torch.stack([torch.tensor(sample["input_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "position_ids": torch.stack([torch.tensor(sample["position_ids"]) for sample in samples], dim=0)
        .long()
        .to("cuda"),
        "loss_mask": torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples], dim=0).bool().to("cuda"),
        "target_ids": torch.stack([torch.tensor(sample["target_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "epoch": min([sample["epoch"] for sample in samples]),
    }


def setup_dataset(tokenizer: PreTrainedTokenizer, config: DataConfigType) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(tokenizer, config)
    elif config.type == "real":
        return SFTDataset(tokenizer, config)
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(
    dataset: StatefulIterableDataset, tokenizer: PreTrainedTokenizer, config: DataConfigType
) -> StatefulDataLoader:
    seq_len = config.micro_batch_size * config.seq_len
    packing_dataset = PackingDataset(dataset, seq_len)
    return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=collate)
