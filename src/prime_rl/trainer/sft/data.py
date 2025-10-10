import json
from collections import defaultdict
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.config import DataConfigType, LossMaskConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

STACKING_DATASET_BUCKET_TIMEOUT = 10


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self):
        self.step, self.epoch = 0, 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self._setup_world_info()

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.step = state_dict["step"]
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

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        length: Literal["fixed", "variable"] = "fixed",
        input_ids: Literal["increasing", "random"] = "random",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.input_ids = input_ids

    def __iter__(self):
        while True:
            self.step += 1

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            seq_len = int(torch.randint(1, self.seq_len, (1,)).item()) if self.length == "variable" else self.seq_len
            input_ids = (
                [self.step - 1] * (seq_len + 1)
                if self.input_ids == "increasing"
                else torch.randint(0, self.vocab_size, (self.seq_len + 1,)).long().tolist()
            )
            position_ids = list(range(seq_len))
            loss_mask = [True] * seq_len
            fake_sample = {
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:],
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            self.num_samples["fake"] += 1
            self.num_tokens["fake"] += len(input_ids)
            yield fake_sample


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt + completion format."""

    def __init__(
        self,
        dataset: HFIterableDataset,
        num_examples: int,
        tokenizer: PreTrainedTokenizer | None,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
    ):
        super().__init__()
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = num_examples
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.max_examples = max_examples
        self.max_epochs = max_epochs

        if self.tokenizer is None:
            self.logger.warning("No tokenizer provided, will not process examples")

        # If specified, select a subset of the dataset
        if self.max_examples is not None:
            self.num_examples = min(self.num_examples, self.max_examples)
            self.dataset = self.dataset.take(self.max_examples)

        # Add dataset index
        # self.dataset = dataset.add_column("index", list(range(len(dataset))), new_fingerprint=str(uuid.uuid4()))

        # Get the data rank and world size
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        assert get_world().world_size % non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = get_world().rank // non_dp_size * num_workers + worker_id
        self.data_world_size = get_world().world_size // non_dp_size * num_workers

    def _process(self, example: dict) -> dict | None:
        # Skip processing if no tokenizer was provided
        if self.tokenizer is None:
            return example

        # Assert that the example has a 'prompt' and 'completion' column
        if "prompt" not in example or "completion" not in example:
            raise ValueError("All examples in the dataset must have a 'prompt' and 'completion' column for SFT")

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
                    "tool_calls": [deserialize_tool_call(tool_call) for tool_call in message.get("tool_calls") or []],
                }
                for message in messages
            ]

        # Deserialize tool call arguments from message list, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#handling-function-calls
        prompt = deserialize_tool_calls(example["prompt"])
        completion = deserialize_tool_calls(example["completion"])

        # Parse available tools, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#function-tool-example
        tools = json.loads(example.get("tools") or "[]")

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
        loss_mask = build_loss_mask(prompt, completion, self.tokenizer, self.loss_mask_config)

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

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

        # Create sample (with one fake target for the last token)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
        }

    def __iter__(self):
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        while True:
            # Determine eoch from current step
            self.epoch = self.step // self.num_examples

            # Break if max epochs is reached
            if self.max_epochs is not None and self.epoch >= self.max_epochs:
                break

            # Shuffle dataset before each epoch
            dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
            dataset_iter = iter(dataset)

            # If resuming, skip the first few samples in the epoch
            if self.step > 0:
                skip_steps = self.step % self.num_examples
            else:
                skip_steps = 0
            if skip_steps > 0:
                self.logger.info(f"Skipping the first {skip_steps} examples in epoch {self.epoch}")

            # Iterate over dataset (one epoch)
            for i, example in enumerate(dataset_iter):
                # Skip steps
                if skip_steps > 0:
                    skip_steps -= 1
                    continue

                self.step += 1

                # Skip samples that don't belong to this data rank
                if i % self.data_world_size != self.data_rank:
                    continue

                processed_example = self._process(cast(dict, example))

                # If processed example is None, skip it (e.g. if tokenized sample exceeds context window)
                if processed_example is None:
                    continue

                # Yield the example
                example = cast(dict, example)
                subset_or_split = example.get("subset", example.get("split")) or "train"
                self.logger.debug(
                    f"Yield example {i}"
                    + (f" from {subset_or_split} " if subset_or_split else " ")
                    + f"with {len(processed_example.get('input_ids', []))} tokens ({sum(processed_example.get('loss_mask', []))} trainable tokens)"
                )
                self.num_samples[subset_or_split] += 1
                self.num_tokens[subset_or_split] += len(processed_example.get("input_ids", []))
                yield processed_example


class CatDataset(StatefulIterableDataset):
    """A dataset that concatenates samples into a single sequence with a fixed length."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len

    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def __iter__(self):
        packed_samples, seq_len = defaultdict(list), 0
        for sample in self.dataset:
            # Add sample to packed samples
            for key, value in sample.items():
                assert isinstance(value, list), f"Value for key {key} must be a list"
                packed_samples[key].extend(value)

            # Update sequence length
            seq_len += len(sample["input_ids"])

            # If batch is full, truncate and yield it
            if seq_len >= self.seq_len:
                for key, value in packed_samples.items():
                    assert isinstance(value, list), f"Value for key {key} must be a list"
                    packed_samples[key] = value[: self.seq_len]
                yield packed_samples
                packed_samples, seq_len = defaultdict(list), 0


class StackDataset(StatefulIterableDataset):
    """A dataset that stacks samples into batch with a fixed area"""

    def __init__(self, dataset: StatefulIterableDataset, max_area: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.max_area = max_area
        assert self.max_area % 256 == 0
        self.bucket_sizes = [max_area // 4, max_area // 2, max_area]
        # Checkpoint state
        self.step = 0
        self.buckets = [[] for _ in range(len(self.bucket_sizes))]
        self.bucket_timers: list[int | None] = [None] * len(self.buckets)

    def state_dict(self) -> dict:
        return {
            "dataset": self.dataset.state_dict(),
            "step": self.step,
            "buckets": self.buckets,
            "bucket_timers": self.bucket_timers,
        }

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.step = state_dict["step"]
        self.buckets = state_dict["buckets"]
        self.bucket_timers = state_dict["bucket_timers"]

    def __iter__(self):
        for sample in self.dataset:
            # Truncate sample if it's longer than max area
            len_sample = len(sample["input_ids"])
            if len_sample > self.max_area:
                for key, value in sample.items():
                    assert isinstance(value, list)
                    sample[key] = sample[key][: self.max_area]
                len_sample = self.max_area

            # Add sample to bucket
            bucket_idx = 0 if len_sample <= self.bucket_sizes[0] else 1 if len_sample <= self.bucket_sizes[1] else 2
            self.buckets[bucket_idx].append(sample)

            # Check if bucket has timed out
            bucket_timer = self.bucket_timers[bucket_idx]
            if bucket_timer is not None:
                hit_timeout = bucket_timer + STACKING_DATASET_BUCKET_TIMEOUT < self.step
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
                for bucket_item in self.buckets[bucket_idx]:
                    for key, value in bucket_item.items():
                        packed_samples[key].append(value + [0] * (self.bucket_sizes[bucket_idx] - len(value)))
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
    }


def cat_collate(samples: list[Sample]) -> Batch:
    return {
        "input_ids": torch.stack([torch.tensor(sample["input_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "position_ids": torch.stack([torch.tensor(sample["position_ids"]) for sample in samples], dim=0)
        .long()
        .to("cuda"),
        "loss_mask": torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples], dim=0).bool().to("cuda"),
        "target_ids": torch.stack([torch.tensor(sample["target_ids"]) for sample in samples], dim=0).long().to("cuda"),
    }


def setup_dataset(
    tokenizer: PreTrainedTokenizer, config: DataConfigType, non_dp_size: int = 1
) -> StatefulIterableDataset:
    if config.type == "fake":
        # Shouldnt matter to handle non_dp_size if dataset is random
        return FakeDataset(
            vocab_size=tokenizer.vocab_size, seq_len=config.seq_len, length=config.length, input_ids=config.input_ids
        )
    elif config.type == "sft":
        logger = get_logger()
        if config.subsets is None and config.splits is None:
            dataset = cast(HFDataset, load_dataset(config.name, split="train"))
            num_examples = len(dataset)
            iterable_dataset = dataset.to_iterable_dataset()
        elif config.subsets is not None and config.splits is None:
            logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
            datasets = [cast(HFDataset, load_dataset(config.name, subset, split="train")) for subset in config.subsets]
            num_examples = len([len(dataset) for dataset in datasets])
            iterable_dataset = interleave_datasets(
                [dataset.to_iterable_dataset() for dataset in datasets],
                probabilities=config.probabilities,
                stopping_strategy=config.stopping_strategy,
                seed=0,
            )
        elif config.subsets is None and config.splits is not None:
            logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
            datasets = [cast(HFDataset, load_dataset(config.name, split=split)) for split in config.splits]
            num_examples = len([len(dataset) for dataset in datasets])
            iterable_dataset = interleave_datasets(
                [dataset.to_iterable_dataset() for dataset in datasets],
                probabilities=config.probabilities,
                stopping_strategy=config.stopping_strategy,
                seed=0,
            )
        else:
            assert config.subsets is not None and config.splits is not None
            logger.debug(
                f"Loading datasets for subsets {config.subsets} and splits {config.splits} with default probabilities 'None'"
            )
            datasets = [
                cast(HFDataset, load_dataset(config.name, subset, split=split))
                for subset, split in zip(config.subsets, config.splits)
            ]
            num_examples = len([len(dataset) for dataset in datasets])
            iterable_dataset = interleave_datasets(
                [dataset.to_iterable_dataset() for dataset in datasets],
                probabilities=config.probabilities,
                stopping_strategy=config.stopping_strategy,
                seed=0,
            )
        return SFTDataset(
            iterable_dataset,
            num_examples,
            tokenizer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(dataset: StatefulIterableDataset, config: DataConfigType) -> StatefulDataLoader:
    seq_len = config.micro_batch_size * config.seq_len
    if config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, seq_len)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        packing_dataset = CatDataset(dataset, seq_len)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
