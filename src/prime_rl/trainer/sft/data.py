import json
import random
import uuid
from collections import defaultdict
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.config import CurriculumConfig, DataConfigType, LossMaskConfig
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
        self.fast_forward = False
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
        dataset: Dataset,
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
        self.num_examples = len(self.dataset)
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

        def strip_content(messages: list[dict]) -> list[dict]:
            def _strip_content(message: dict) -> dict:
                if isinstance(message.get("content"), str):
                    return {**message, "content": message["content"].strip()}
                return message

            return [_strip_content(message) for message in messages]

        # Deserialize tool call arguments from message list, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#handling-function-calls
        prompt = deserialize_tool_calls(example["prompt"])
        completion = deserialize_tool_calls(example["completion"])

        # Strip content from all messages so that incremental tokenization works
        # NOTE: This has the side effect that we do never train on leading or trailing whitespace
        prompt = strip_content(prompt)
        completion = strip_content(completion)

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
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
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
        dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
        while True:
            self.step += 1

            # Determine epoch from current step
            epoch = (self.step - 1) // self.num_examples

            # Break if max epochs is reached
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break

            # Update stored epoch if new epoch is reached, optionally shuffle
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            # Get example
            example = dataset[(self.step - 1) % self.num_examples]

            # Process example
            processed_example = self._process(cast(dict, example))

            # If processed example is None, skip it (e.g. if tokenized sample exceeds context window)
            if processed_example is None:
                continue

            # Yield the example
            example = cast(dict, example)
            subset_or_split = example.get("__subset") or example.get("__split")
            self.logger.debug(
                f"Yield example {example.get('__index', '')}"
                + (f" from {subset_or_split} " if subset_or_split else " ")
                + f"with {len(processed_example.get('input_ids', []))} tokens ({sum(processed_example.get('loss_mask', []))} trainable tokens)"
            )
            self.num_samples[subset_or_split] += 1
            self.num_tokens[subset_or_split] += len(processed_example.get("input_ids", []))
            yield processed_example


class CurriculumSFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with difficulty-based curriculum learning.

    Samples are organized by difficulty level and sampling probabilities change
    based on training progress, starting with easier samples and gradually
    introducing harder ones.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer | None,
        curriculum_config: CurriculumConfig,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        max_steps: int | None = None,
    ):
        super().__init__()
        self.logger = get_logger()
        self.tokenizer = tokenizer
        self.curriculum_config = curriculum_config
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.max_examples = max_examples
        self.max_epochs = max_epochs
        self.max_steps = max_steps

        if self.tokenizer is None:
            self.logger.warning("No tokenizer provided, will not process examples")

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
        if self.max_examples is not None:
            dataset = dataset.select(range(min(len(dataset), self.max_examples)))

        # Organize examples by difficulty level
        self.difficulty_field = curriculum_config.difficulty_field
        self.difficulty_levels = curriculum_config.difficulty_levels
        self.unknown_difficulty_level = self.difficulty_levels[-1]  # Treat unknown as hardest

        # Validate that the difficulty field exists
        if self.difficulty_field not in dataset.column_names:
            raise ValueError(
                f"Difficulty field '{self.difficulty_field}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        # Organize examples by difficulty level
        self.examples_by_difficulty: dict[str, list[dict]] = {level: [] for level in self.difficulty_levels}
        for i, example in enumerate(dataset):
            example = cast(dict, example)
            example["__index"] = i
            difficulty = example.get(self.difficulty_field, self.unknown_difficulty_level)
            if difficulty not in self.difficulty_levels:
                self.logger.warning(
                    f"Unknown difficulty '{difficulty}' for example {i}, treating as '{self.unknown_difficulty_level}'"
                )
                difficulty = self.unknown_difficulty_level
            self.examples_by_difficulty[difficulty].append(example)

        # Log difficulty distribution
        total_examples = sum(len(examples) for examples in self.examples_by_difficulty.values())
        self.num_examples = total_examples
        for level in self.difficulty_levels:
            count = len(self.examples_by_difficulty[level])
            self.logger.info(
                f"Difficulty level '{level}': {count} examples ({count / total_examples * 100:.1f}%)"
            )

        # Track curriculum metrics
        self.curriculum_samples_by_difficulty = {level: 0 for level in self.difficulty_levels}

        # Initialize random state
        self.rng = random.Random(seed)

    def _get_difficulty_probabilities(self, progress: float) -> dict[str, float]:
        """Compute sampling probabilities for each difficulty level based on training progress.

        Args:
            progress: Training progress as a fraction [0, 1]

        Returns:
            Dictionary mapping difficulty levels to sampling probabilities
        """
        num_levels = len(self.difficulty_levels)
        warmup = self.curriculum_config.warmup_fraction
        full_diff = self.curriculum_config.full_difficulty_fraction
        min_prob = self.curriculum_config.min_difficulty_prob

        if self.curriculum_config.schedule == "step":
            # Step schedule: unlock one difficulty level at each threshold
            # Divide the range [warmup, full_diff] into num_levels - 1 sections
            probs = {level: 0.0 for level in self.difficulty_levels}

            if progress < warmup:
                # Only easiest difficulty
                probs[self.difficulty_levels[0]] = 1.0
            else:
                # Calculate how many levels to unlock
                curriculum_progress = (progress - warmup) / (full_diff - warmup) if full_diff > warmup else 1.0
                curriculum_progress = min(curriculum_progress, 1.0)

                # Number of levels to unlock (at least 1, at most all)
                num_unlocked = max(1, min(num_levels, int(curriculum_progress * num_levels) + 1))

                # Distribute probability evenly among unlocked levels, with min_prob for easiest
                if num_unlocked == 1:
                    probs[self.difficulty_levels[0]] = 1.0
                else:
                    # Ensure minimum probability for easiest level
                    remaining_prob = 1.0 - min_prob
                    prob_per_level = remaining_prob / (num_unlocked - 1)

                    probs[self.difficulty_levels[0]] = min_prob
                    for i in range(1, num_unlocked):
                        probs[self.difficulty_levels[i]] = prob_per_level

        else:  # linear schedule
            # Linear schedule: gradually shift probability towards harder levels
            probs = {level: 0.0 for level in self.difficulty_levels}

            if progress < warmup:
                # Only easiest difficulty during warmup
                probs[self.difficulty_levels[0]] = 1.0
            elif progress >= full_diff:
                # Full difficulty: distribute evenly with minimum for easiest
                if num_levels == 1:
                    probs[self.difficulty_levels[0]] = 1.0
                else:
                    remaining_prob = 1.0 - min_prob
                    prob_per_level = remaining_prob / (num_levels - 1)
                    probs[self.difficulty_levels[0]] = min_prob
                    for i in range(1, num_levels):
                        probs[self.difficulty_levels[i]] = prob_per_level
            else:
                # Transitioning: linearly interpolate between warmup and full difficulty
                transition_progress = (progress - warmup) / (full_diff - warmup)

                # Start state: all probability on easiest
                # End state: evenly distributed with min_prob on easiest
                if num_levels == 1:
                    probs[self.difficulty_levels[0]] = 1.0
                else:
                    # Interpolate easiest level probability from 1.0 to min_prob
                    easiest_prob = 1.0 - transition_progress * (1.0 - min_prob)
                    probs[self.difficulty_levels[0]] = easiest_prob

                    # Distribute remaining probability among other levels progressively
                    remaining_prob = 1.0 - easiest_prob
                    # Weight harder levels by how far we are in transition
                    level_weights = []
                    for i in range(1, num_levels):
                        # Higher index = harder = lower weight at start, increasing
                        # Level i gets unlocked when transition_progress >= (i-1)/(num_levels-1)
                        unlock_threshold = (i - 1) / (num_levels - 1)
                        if transition_progress >= unlock_threshold:
                            weight = min(1.0, (transition_progress - unlock_threshold) * (num_levels - 1))
                        else:
                            weight = 0.0
                        level_weights.append(weight)

                    total_weight = sum(level_weights)
                    if total_weight > 0:
                        for i, weight in enumerate(level_weights):
                            probs[self.difficulty_levels[i + 1]] = remaining_prob * weight / total_weight

        # Normalize probabilities to account for empty difficulty levels
        available_probs = {
            level: prob
            for level, prob in probs.items()
            if len(self.examples_by_difficulty.get(level, [])) > 0
        }

        if not available_probs:
            # Fallback: distribute evenly among all non-empty levels
            non_empty_levels = [
                level for level in self.difficulty_levels
                if len(self.examples_by_difficulty.get(level, [])) > 0
            ]
            if non_empty_levels:
                prob = 1.0 / len(non_empty_levels)
                return {level: prob for level in non_empty_levels}
            else:
                raise ValueError("No examples available in any difficulty level")

        # Normalize
        total = sum(available_probs.values())
        if total > 0:
            return {level: prob / total for level, prob in available_probs.items()}

        return available_probs

    def _sample_difficulty_level(self, progress: float) -> str:
        """Sample a difficulty level based on current progress."""
        probs = self._get_difficulty_probabilities(progress)
        levels = list(probs.keys())
        weights = [probs[level] for level in levels]
        return self.rng.choices(levels, weights=weights, k=1)[0]

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "rng_state": self.rng.getstate(),
            "curriculum_samples_by_difficulty": self.curriculum_samples_by_difficulty,
        }

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.fast_forward = True
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        if "rng_state" in state_dict:
            self.rng.setstate(state_dict["rng_state"])
        if "curriculum_samples_by_difficulty" in state_dict:
            self.curriculum_samples_by_difficulty = state_dict["curriculum_samples_by_difficulty"]

    def _process(self, example: dict) -> dict | None:
        """Process a single example (reuse logic from SFTDataset)."""
        # Skip processing if no tokenizer was provided
        if self.tokenizer is None:
            return example

        # Assert that the example has a 'prompt' and 'completion' column
        if "prompt" not in example or "completion" not in example:
            raise ValueError("All examples in the dataset must have a 'prompt' and 'completion' column for SFT")

        def deserialize_tool_calls(messages: list[dict]) -> list[dict]:
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

        def strip_content(messages: list[dict]) -> list[dict]:
            def _strip_content(message: dict) -> dict:
                if isinstance(message.get("content"), str):
                    return {**message, "content": message["content"].strip()}
                return message

            return [_strip_content(message) for message in messages]

        # Deserialize tool call arguments from message list, if present
        prompt = deserialize_tool_calls(example["prompt"])
        completion = deserialize_tool_calls(example["completion"])

        # Strip content from all messages
        prompt = strip_content(prompt)
        completion = strip_content(completion)

        # Parse available tools, if present
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
                if message["role"] == "tool" and i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                    continue
                cur_ids = tokenizer.apply_chat_template(
                    messages[: i + 1],
                    tools=tools,
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
                    f"Got mismatch in incremental tokenization with chat template at message {i}."
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
                f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. Manually appending EOS token..."
            )
            input_ids.append(cast(int, self.tokenizer.eos_token_id))
            loss_mask.append(True)

        # Prepare inputs
        target_ids = input_ids.copy()[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len})."
            )
            return None

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
        }

    def _compute_progress(self) -> float:
        """Compute training progress as a fraction [0, 1]."""
        if self.max_steps is not None and self.max_steps > 0:
            # Use step-based progress
            return min(1.0, self.step / self.max_steps)
        elif self.max_epochs is not None and self.max_epochs > 0:
            # Use epoch-based progress
            return min(1.0, self.epoch / self.max_epochs)
        else:
            # No progress tracking available, assume full difficulty
            return 1.0

    def __iter__(self):
        """Iterate through examples using curriculum-based sampling."""
        # Shuffle examples within each difficulty level
        if self.shuffle:
            for level in self.difficulty_levels:
                self.rng.shuffle(self.examples_by_difficulty[level])

        # Track indices within each difficulty level
        indices_by_difficulty = {level: 0 for level in self.difficulty_levels}

        while True:
            self.step += 1

            # Compute current progress
            progress = self._compute_progress()

            # Check for max epochs
            if self.max_epochs is not None and self.epoch >= self.max_epochs:
                break

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            # Sample a difficulty level
            difficulty = self._sample_difficulty_level(progress)
            examples = self.examples_by_difficulty[difficulty]

            if not examples:
                continue

            # Get the next example from this difficulty level
            idx = indices_by_difficulty[difficulty]
            if idx >= len(examples):
                # Reshuffle and reset index for this difficulty level
                if self.shuffle:
                    self.rng.shuffle(examples)
                idx = 0
                # Track epochs based on any difficulty level completing
                self.epoch += 1
                if self.shuffle:
                    # Re-seed RNG for reproducibility
                    self.rng.seed(self.seed + self.epoch)

            example = examples[idx]
            indices_by_difficulty[difficulty] = idx + 1

            # Process example
            processed_example = self._process(cast(dict, example))

            if processed_example is None:
                continue

            # Track metrics
            self.curriculum_samples_by_difficulty[difficulty] += 1
            subset_or_split = example.get("__subset") or example.get("__split")
            self.logger.debug(
                f"Yield example {example.get('__index', '')} (difficulty={difficulty}, progress={progress:.2f})"
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
        self.bucket_sizes = []
        while max_area % 256 == 0:
            self.bucket_sizes.insert(0, max_area)
            max_area //= 2
        self.logger.debug(f"Initialized {len(self.bucket_sizes)} buckets (bucket_sizes={self.bucket_sizes})")
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
            def find_bucket_idx(len_sample: int) -> int:
                bucket_idx = 0
                while bucket_idx < len(self.bucket_sizes) - 1 and len_sample > self.bucket_sizes[bucket_idx]:
                    bucket_idx += 1
                return bucket_idx

            bucket_idx = find_bucket_idx(len_sample)
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
                            dummy_sample[key] = [0]
                        self.buckets[bucket_idx].append(dummy_sample)

                packed_samples = defaultdict(list)
                num_samples, num_tokens, num_trainable_tokens, num_pad_tokens = 0, 0, 0, 0
                for bucket_item in self.buckets[bucket_idx]:
                    num_samples += 1
                    for key, value in bucket_item.items():
                        pad_tokens = [0] * (self.bucket_sizes[bucket_idx] - len(value))
                        if key == "loss_mask":
                            num_tokens += len(value)
                            num_trainable_tokens += sum(value)
                            num_pad_tokens += len(pad_tokens)
                        packed_samples[key].append(value + pad_tokens)
                reason = "bucket is full" if is_full else "because bucket timed out"
                reason += " and " if is_full and hit_timeout else ""
                reason += "bucket timed out" if hit_timeout else ""
                self.logger.debug(
                    f"Yield bucket {bucket_idx} because {reason} with {num_samples=}, {num_tokens=}, {num_trainable_tokens=}, {num_pad_tokens=}"
                )
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


def setup_and_interleave_datasets(
    dataset_name: str,
    subsets_and_splits: list[tuple[str | None, str]],
    probabilities: list[float] | None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"],
    seed: int = 0,
) -> Dataset:
    logger = get_logger()
    datasets = []
    for subset, split in subsets_and_splits:
        logger.debug(f"Loading dataset {dataset_name} with {subset=} and {split=}")
        dataset = cast(Dataset, load_dataset(dataset_name, subset, split=split))
        num_examples = len(dataset)
        dataset = dataset.add_column("__subset", [subset] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__split", [split] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__index", list(range(num_examples)), new_fingerprint=str(uuid.uuid4()))
        datasets.append(dataset)
    if len(datasets) > 1:
        logger.debug(f"Interleaving datasets with {probabilities=} and {stopping_strategy=}")
        dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
            seed=seed,
        )
    else:
        dataset = datasets[0]

    return dataset


def setup_dataset(
    tokenizer: PreTrainedTokenizer, config: DataConfigType, non_dp_size: int = 1, max_steps: int | None = None
) -> StatefulIterableDataset:
    if config.type == "fake":
        # Shouldnt matter to handle non_dp_size if dataset is random
        return FakeDataset(
            vocab_size=tokenizer.vocab_size, seq_len=config.seq_len, length=config.length, input_ids=config.input_ids
        )
    elif config.type == "sft":
        logger = get_logger()
        if config.subsets is None and config.splits is None:
            dataset = setup_and_interleave_datasets(
                dataset_name=config.name,
                subsets_and_splits=[(None, "train")],
                probabilities=config.probabilities,
                stopping_strategy=config.stopping_strategy,
            )
        elif config.subsets is not None and config.splits is None:
            logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
            dataset = setup_and_interleave_datasets(
                dataset_name=config.name,
                subsets_and_splits=[(subset, "train") for subset in config.subsets],
                probabilities=config.probabilities,
                stopping_strategy=config.stopping_strategy,
            )
        elif config.subsets is None and config.splits is not None:
            logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
            dataset = setup_and_interleave_datasets(
                dataset_name=config.name,
                subsets_and_splits=[(None, split) for split in config.splits],
                probabilities=config.probabilities,
                stopping_strategy=config.stopping_strategy,
            )
        else:
            assert config.subsets is not None and config.splits is not None
            logger.debug(f"Loading datasets for subsets {config.subsets} with splits {config.splits}")
            dataset = setup_and_interleave_datasets(
                dataset_name=config.name,
                subsets_and_splits=list(zip(config.subsets, config.splits)),
                probabilities=config.probabilities,
                stopping_strategy=config.stopping_strategy,
            )

        # Use CurriculumSFTDataset if curriculum learning is enabled
        if config.curriculum.enabled:
            logger.info(
                f"Using curriculum learning with difficulty levels: {config.curriculum.difficulty_levels}, "
                f"schedule: {config.curriculum.schedule}, warmup: {config.curriculum.warmup_fraction}, "
                f"full_difficulty: {config.curriculum.full_difficulty_fraction}"
            )
            return CurriculumSFTDataset(
                dataset,
                tokenizer,
                curriculum_config=config.curriculum,
                shuffle=config.shuffle,
                seed=config.seed,
                seq_len=config.seq_len,
                loss_mask_config=config.loss_mask,
                non_dp_size=non_dp_size,
                max_steps=max_steps,
            )

        return SFTDataset(
            dataset,
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
    if config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
