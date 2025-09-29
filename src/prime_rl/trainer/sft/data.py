import json
import uuid
from collections import defaultdict
from typing import Iterator, TypedDict, cast

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.config import DataConfigType, FakeDataConfig, LossMaskConfig, SFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

STACKING_DATASET_BUCKET_TIMEOUT = 10


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

    def __iter__(self):
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
                "index": self.step,
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:],
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "epoch": self.epoch,
            }
            yield fake_sample


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt + completion format."""

    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, config: SFTDataConfig, non_dp_size: int = 1):
        super().__init__()
        self.logger = get_logger()
        self.config = config
        self.tokenizer = tokenizer

        # Add dataset index
        self.dataset = dataset.add_column("index", list(range(len(dataset))), new_fingerprint=str(uuid.uuid4()))

        # Assert that the dataset has a 'text' column
        if "prompt" not in self.dataset.column_names or "completion" not in self.dataset.column_names:
            raise ValueError("HF dataset must have a 'prompt' and 'completion' column for SFT")

        # Get the data rank and world size
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        assert get_world().world_size % non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = get_world().rank // non_dp_size * num_workers + worker_id
        self.data_world_size = get_world().world_size // non_dp_size * num_workers

        # If specified, select a subset of the dataset
        if config.num_examples is not None:
            self.dataset = self.dataset.select(range(config.num_examples))

        # Store the number of examples in the dataset
        self.num_examples = len(self.dataset)

    def __iter__(self):
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        dataset = self.dataset.shuffle(seed=self.epoch + self.config.seed) if self.config.shuffle else self.dataset
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
                dataset = self.dataset.shuffle(seed=self.epoch + self.config.seed) if self.config.shuffle else self.dataset
                self.epoch = epoch

            assert "prompt" in example and "completion" in example, (
                "Prompt and completion must be present in the example"
            )
            assert isinstance(example["prompt"], list) and isinstance(example["completion"], list), (
                "Prompt and completion must be lists."
            )

            def deserialize_tool_calls(messages: list[dict]) -> list[dict]:
                """
                Deserialize tool calls in messages, if any are present. Iterates
                over all messages in a message list and tries to find
                "tool_calls" key. If found, assumes it is a OAI format and has
                key "function" with "arguments" key which is stringified. It
                will then deserialize the argument so that chat tmeplates like
                Qwen3's can be used.
                """

                def deserialize_tool_call(tool_call: dict) -> dict:
                    return {
                        **tool_call,
                        "function": {
                            **tool_call["function"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                        },
                    }

                return [
                    {
                        **message,
                        "tool_calls": [
                            deserialize_tool_call(tool_call) for tool_call in message.get("tool_calls", []) or []
                        ],
                    }
                    for message in messages
                ]

            # Deserialize tool call arguments from message list, if present - assumes OAI format
            # Reference: https://platform.openai.com/docs/guides/function-calling#handling-function-calls
            prompt = deserialize_tool_calls(example["prompt"])
            completion = deserialize_tool_calls(example["completion"])

            # Parse available tools, if present - assumes OAI format
            # Reference: https://platform.openai.com/docs/guides/function-calling#function-tool-example
            tools = json.loads(example.get("tools", "[]"))

            def should_mask(message: dict, loss_mask_config: LossMaskConfig) -> bool:
                assert "role" in message, "Message must have a role"
                match message["role"]:
                    case "user":
                        return True if loss_mask_config.user else False
                    case "assistant":
                        return True if loss_mask_config.assistant else False
                    case "system":
                        return True if loss_mask_config.system else False
                    case "tool":
                        return True if loss_mask_config.tool else False
                    case _:
                        raise ValueError(f"Invalid message role: {message['role']}")

            def build_loss_mask(prompt, completion, tokenizer, loss_mask_config: LossMaskConfig) -> list[bool]:
                messages = prompt + completion
                loss_mask: list[bool] = []
                prev_ids, prev_len = [], 0
                for i, message in enumerate(messages):
                    assert "role" in message, "Message must have a role"
                    # Support parallel tool call outputs (treat them as one message for loss mask)
                    if message["role"] == "tool" and i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                        continue
                    cur_ids = tokenizer.apply_chat_template(
                        messages[: i + 1],
                        tools=tools,
                        # This is to mask out the generation prompt after user and tool messages
                        # It leads to us not training on <|im_start|>assistant
                        add_generation_prompt=True
                        if (
                            message["role"] in ["user", "tool"]
                            and i + 1 < len(messages)
                            and messages[i + 1]["role"] == "assistant"
                        )
                        else False,
                        **example.get("chat_template_kwargs", {}),
                    )
                    assert prev_ids == cur_ids[:prev_len], (
                        f"Got mismatch in incremental tokenization with chat template at message {i}. Previous ids: {prev_ids} != {cur_ids[:prev_len]=}.\nDecoded prev_ids:\n{tokenizer.decode(prev_ids)}\nDecoded cur_ids:\n{tokenizer.decode(cur_ids[:prev_len])}"
                    )
                    loss_mask.extend([should_mask(message, loss_mask_config)] * (len(cur_ids) - prev_len))
                    prev_ids, prev_len = cur_ids, len(cur_ids)

                return loss_mask

            # Build input_ids
            input_ids = cast(
                list[int],
                self.tokenizer.apply_chat_template(
                    prompt + completion,
                    tools=tools,
                    **example.get("chat_template_kwargs", {}),
                ),
            )

            # Build loss_mask
            loss_mask = build_loss_mask(prompt, completion, self.tokenizer, self.config.loss_mask)

            # If EOS token is not found, manually append it
            if not self.tokenizer.eos_token_id in input_ids:
                self.logger.warning(
                    f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. Is something wrong with the chat template? Manually appending EOS token..."
                )
                input_ids.append(cast(int, self.tokenizer.eos_token_id))
                loss_mask.append(True)

            # Prepare inputs
            target_ids = input_ids.copy()[1:]
            loss_mask = loss_mask[1:]
            input_ids = input_ids[:-1]

            if sum(loss_mask[: self.config.seq_len]) == 0:
                self.logger.warning(
                    f"Skipping example with index {self.step} because no trainable tokens were found within the context window ({self.config.seq_len}). This is to prevent NaN loss."
                )
                continue

            assert len(input_ids) == len(loss_mask) == len(target_ids), (
                f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
            )
            assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
            assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

            # Create sample (with one fake target for the last token)
            sample = {
                "index": example["index"],
                "input_ids": input_ids,
                "target_ids": target_ids,
                "loss_mask": loss_mask,
                "position_ids": list(range(len(input_ids))),
                "epoch": self.epoch,
            }

            yield sample


class CatDataset(StatefulIterableDataset):
    """A dataset that concatenates samples into a single sequence with a fixed length."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.logger = get_logger()
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
        packed_samples, seq_len, indices = self.packed_samples, self.current_seq_len, []
        for sample in self.dataset:
            # Add sample to packed samples
            for key, value in sample.items():
                if key == "epoch":
                    packed_samples[key] = min(packed_samples.get(key, float("inf")), value)
                elif key == "index":
                    indices.append(value)
                else:
                    packed_samples[key].extend(value)

            # Update sequence length
            seq_len += len(sample["input_ids"])

            # If batch is full, truncate and yield it
            if seq_len >= self.seq_len:
                for key, value in packed_samples.items():
                    if isinstance(value, list):
                        packed_samples[key] = value[: self.seq_len]
                self.logger.debug(f"Yield batch with dataset indices={indices}")
                yield packed_samples
                packed_samples, seq_len, indices = defaultdict(list), 0, []


class StackDataset(StatefulIterableDataset):
    """A dataset that stacks samples into batch with a fixed area"""

    def __init__(self, dataset: StatefulIterableDataset, max_area: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.max_area = max_area
        self.current_seq_len = 0
        # TODO: Can we steal step from dataset?
        self.step = 0
        assert self.max_area % 256 == 0
        self.bucket_sizes = [max_area // 4, max_area // 2, max_area]
        self.buckets = [[] for _ in range(len(self.bucket_sizes))]
        self.bucket_timers: list[int | None] = [None] * len(self.buckets)
        self.bucket_timeout = STACKING_DATASET_BUCKET_TIMEOUT

    def state_dict(self) -> dict:
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict)

    def __iter__(self) -> Iterator[Sample]:
        for sample in self.dataset:
            # Truncate sample if it's longer than max area
            len_sample = len(sample["input_ids"])
            if len_sample > self.max_area:
                for key, value in sample.items():
                    if isinstance(value, list):
                        sample[key] = sample[key][: self.max_area]
                len_sample = self.max_area

            # Add sample to bucket
            bucket_idx = 0 if len_sample <= self.bucket_sizes[0] else 1 if len_sample <= self.bucket_sizes[1] else 2
            self.buckets[bucket_idx].append(sample)

            # Check if bucket has timed out
            bucket_timer = self.bucket_timers[bucket_idx]
            if bucket_timer is not None:
                hit_timeout = bucket_timer + self.bucket_timeout < self.step
            else:
                hit_timeout = False

            # Check if bucket is full
            is_full = self.bucket_sizes[bucket_idx] * len(self.buckets[bucket_idx]) >= self.max_area

            if is_full or hit_timeout:
                if hit_timeout:
                    while bucket_idx < len(self.buckets) - 1:
                        if (
                            self.bucket_sizes[bucket_idx + 1]
                            * (len(self.buckets[bucket_idx]) + len(self.buckets[bucket_idx + 1]))
                            < self.max_area
                        ):
                            self.buckets[bucket_idx + 1].extend(self.buckets[bucket_idx])
                            self.buckets[bucket_idx] = []
                            self.bucket_timers[bucket_idx] = None
                            bucket_idx += 1
                        else:
                            break

                    while self.bucket_sizes[bucket_idx] * len(self.buckets[bucket_idx]) < self.max_area:
                        dummy_sample = {}
                        for key, value in sample.items():
                            if key == "epoch":
                                dummy_sample[key] = value
                            else:
                                dummy_sample[key] = [0]
                        self.buckets[bucket_idx].append(dummy_sample)

                packed_samples = defaultdict(list)
                indices = []
                for bucket_item in self.buckets[bucket_idx]:
                    for key, value in bucket_item.items():
                        if key == "epoch":
                            packed_samples[key] = min(packed_samples.get(key, float("inf")), value)
                        elif key == "index":
                            indices.append(value)
                        else:
                            packed_samples[key].append(value + [0] * (self.bucket_sizes[bucket_idx] - len(value)))
                self.logger.debug(f"Yield batch with dataset indices={indices}")
                yield packed_samples
                self.step += 1
                self.buckets[bucket_idx] = []
                self.bucket_timers[bucket_idx] = None
            else:
                if self.bucket_timers[bucket_idx] is None:
                    self.bucket_timers[bucket_idx] = self.step


def stack_collate(samples: list[Sample]) -> Batch:
    return {
        "input_ids": torch.tensor(samples[0]["input_ids"], dtype=torch.long, device="cuda"),
        "position_ids": torch.tensor(samples[0]["position_ids"], dtype=torch.long, device="cuda"),
        "loss_mask": torch.tensor(samples[0]["loss_mask"], dtype=torch.bool, device="cuda"),
        "target_ids": torch.tensor(samples[0]["target_ids"], dtype=torch.long, device="cuda"),
        "epoch": min([sample["epoch"] for sample in samples]),
    }


def cat_collate(samples: list[Sample]) -> Batch:
    return {
        "input_ids": torch.stack([torch.tensor(sample["input_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "position_ids": torch.stack([torch.tensor(sample["position_ids"]) for sample in samples], dim=0)
        .long()
        .to("cuda"),
        "loss_mask": torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples], dim=0).bool().to("cuda"),
        "target_ids": torch.stack([torch.tensor(sample["target_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "epoch": min([sample["epoch"] for sample in samples]),
    }


def setup_dataset(
    tokenizer: PreTrainedTokenizer, config: DataConfigType, non_dp_size: int = 1
) -> StatefulIterableDataset:
    if config.type == "fake":
        # Shouldnt matter to handle non_dp_size if dataset is random
        return FakeDataset(tokenizer, config)
    elif config.type == "sft":
        dataset = concatenate_datasets(
            [cast(Dataset, load_dataset(config.name, split=split)) for split in config.splits]
        )
        return SFTDataset(dataset, tokenizer, config, non_dp_size)
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(
    dataset: StatefulIterableDataset, tokenizer: PreTrainedTokenizer, config: DataConfigType
) -> StatefulDataLoader:
    seq_len = config.micro_batch_size * config.seq_len
    if config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, seq_len)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        packing_dataset = CatDataset(dataset, seq_len)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
