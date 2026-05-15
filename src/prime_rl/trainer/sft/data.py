import json
import math
import random
import uuid
from collections import defaultdict
from typing import Literal, NotRequired, TypedDict, cast

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import DataConfig, LossMaskConfig, SFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import (
    IncrementalTokenizationError,
    build_incremental_token_mask,
    deserialize_tool_calls,
    normalize_messages,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.sequence_packing import build_cu_seqlens

STACKING_DATASET_BUCKET_TIMEOUT = 10

# Profile definitions: map HF config names → dimension weights for system prompt sampling.
# Each profile assigns importance weights to prompt dimensions (tags).
SYSTEM_PROMPT_PROFILES: dict[str, dict[str, float]] = {
    "chat": {"natural": 2.0, "pragmatic": 1.0},
    "math": {"analytic": 2.0, "calibrated": 1.0},
    "coding-bare": {"concise": 2.0, "literal": 1.0},
    "coding-explained": {"explanatory": 2.0, "pragmatic": 1.0},
    "precise-short": {"literal": 2.0, "concise": 1.0},
    "science": {"analytic": 1.0, "explanatory": 1.0},
}

# Map HF config (subset) names to profiles.
SUBSET_TO_PROFILE: dict[str, str] = {
    "wildchat": "chat",
    "openassistant": "chat",
    "tulu-3-persona-math": "math",
    "openmathinstruct-2": "math",
    "tulu-3-persona-algebra": "math",
    "dolci-python-algo": "coding-bare",
    "evol-codealpaca": "coding-explained",
    "dolci-precise-if": "precise-short",
    "flan": "precise-short",
    "tablegpt": "precise-short",
    "dolci-openthoughts-sci": "science",
    "sciriff": "science",
}

# Smoothing constant for profile weights to ensure non-zero probability for all prompts.
_PROFILE_EPSILON = 0.01


class SystemPromptSampler:
    """Loads a pool of tagged system prompts and samples from it using profile-weighted distributions.

    `pool_path` accepts either:
      - A local file path to `system_prompts_final.json` (expects a sibling
        `system_prompts_expanded.json` for tags); or
      - A HuggingFace Hub dataset repo ID (e.g. `joanvelja/sft-system-prompts-v1`)
        containing `system_prompts_final.json` + `system_prompts_expanded.json`.

    A string is treated as a local path if the file exists on disk, otherwise it
    is treated as an HF repo ID.
    """

    _FINAL_FILENAME = "system_prompts_final.json"
    _EXPANDED_FILENAME = "system_prompts_expanded.json"

    def __init__(self, pool_path: str):
        final_path, expanded_path = self._resolve_paths(pool_path)

        with open(final_path) as f:
            self.prompts: list[str] = json.load(f)
        if not self.prompts:
            raise ValueError(f"System prompt pool at {pool_path} is empty")

        # Try loading the expanded file for tags (covers all prompts, not just seeds).
        # Fall back to uniform tagging if unavailable.
        self.tags_per_prompt: list[list[str]] = []
        try:
            with open(expanded_path) as f:
                raw = json.load(f)
            # Handle both formats: list-of-dicts or {"prompts": list-of-dicts}
            if isinstance(raw, dict):
                raw = raw.get("prompts", [])
            prompt_to_tags = {item["prompt"]: item.get("tags", []) for item in raw}
            for prompt in self.prompts:
                self.tags_per_prompt.append(prompt_to_tags.get(prompt, []))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            self.tags_per_prompt = [[] for _ in self.prompts]

    @classmethod
    def _resolve_paths(cls, pool_path: str) -> tuple[str, str]:
        """Return (final_path, expanded_path) as local filesystem paths.

        Downloads from HF Hub if `pool_path` looks like a repo ID.
        """
        from pathlib import Path

        p = Path(pool_path)
        if p.is_file():
            final_path = str(p)
            expanded_path = pool_path.replace("_final.json", "_expanded.json")
            return final_path, expanded_path

        # Treat as HF Hub dataset repo ID (e.g. "joanvelja/sft-system-prompts-v1")
        from huggingface_hub import hf_hub_download

        final_path = hf_hub_download(repo_id=pool_path, filename=cls._FINAL_FILENAME, repo_type="dataset")
        expanded_path = hf_hub_download(repo_id=pool_path, filename=cls._EXPANDED_FILENAME, repo_type="dataset")
        return final_path, expanded_path

    def sample(self, profile_name: str | None, seed: int) -> str:
        """Sample a system prompt using profile-weighted distribution with deterministic seed."""
        rng = random.Random(seed)

        if profile_name is None or profile_name not in SYSTEM_PROMPT_PROFILES:
            # Uniform sampling when no profile matches
            return rng.choice(self.prompts)

        dim_weights = SYSTEM_PROMPT_PROFILES[profile_name]
        weights = []
        for tags in self.tags_per_prompt:
            score = sum(dim_weights.get(tag, 0.0) for tag in tags)
            weights.append(_PROFILE_EPSILON + math.exp(score))

        # Weighted sample
        return rng.choices(self.prompts, weights=weights, k=1)[0]


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    sequence_lengths: NotRequired[list[int]]


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    sequence_lengths: NotRequired[Int[Tensor, "num_sequences"]]
    cu_seqlens: NotRequired[Int[Tensor, "num_sequences_plus_one"]]
    max_seqlen: NotRequired[int]


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
                "sequence_lengths": [seq_len],
            }
            self.num_samples["fake"] += 1
            self.num_tokens["fake"] += len(input_ids)
            yield fake_sample


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt/completion or raw messages format."""

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
        system_prompt_sampler: SystemPromptSampler | None = None,
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
        self.system_prompt_sampler = system_prompt_sampler

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

        def resolve_messages(example: dict) -> list[dict]:
            # `messages` takes precedence over explicit split fields and is interpreted
            # as a whole-chat training sample with an empty prompt.
            if "messages" in example:
                messages = normalize_messages(example["messages"], default_role="assistant")
            elif "prompt" in example and "completion" in example:
                messages = normalize_messages(example["prompt"], default_role="user") + normalize_messages(
                    example["completion"], default_role="assistant"
                )
            else:
                raise ValueError(
                    "All examples in the dataset must have either a 'messages' column "
                    "or both 'prompt' and 'completion' columns for SFT"
                )

            # Deserialize tool call arguments from message list, if present - assumes OAI format
            # Reference: https://platform.openai.com/docs/guides/function-calling#handling-function-calls
            messages = deserialize_tool_calls(messages)

            # Strip content from all messages so that incremental tokenization works
            # NOTE: This has the side effect that we do never train on leading or trailing whitespace
            return strip_message_content(messages)

        messages = resolve_messages(example)

        # Multi-turn filter: skip samples with >2 messages (before injection)
        if self.system_prompt_sampler is not None and len(messages) > 2:
            self.logger.debug(
                f"Skipping multi-turn example {example.get('__index', '')} "
                f"({len(messages)} messages) — system prompt injection requires single-turn"
            )
            return None

        # System prompt injection: prepend a profile-weighted system prompt
        if self.system_prompt_sampler is not None:
            subset = example.get("__subset")
            profile = SUBSET_TO_PROFILE.get(subset) if subset else None
            sample_index = example.get("__index", 0)
            prompt = self.system_prompt_sampler.sample(profile, seed=self.seed + sample_index)
            messages = [{"role": "system", "content": prompt}] + messages

        # Parse available tools, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#function-tool-example
        tools = json.loads(example.get("tools") or "[]")

        def should_mask(message: dict) -> bool:
            assert "role" in message, "Message must have a role"
            match message["role"]:
                case "user":
                    return True if self.loss_mask_config.user else False
                case "assistant":
                    return True if self.loss_mask_config.assistant else False
                case "system":
                    return True if self.loss_mask_config.system else False
                case "tool":
                    return True if self.loss_mask_config.tool else False
                case _:
                    raise ValueError(f"Invalid message role: {message['role']}")

        try:
            input_ids, loss_mask = build_incremental_token_mask(
                self.tokenizer,
                messages,
                role_to_mask=should_mask,
                tools=tools,
                chat_template_kwargs=example.get("chat_template_kwargs", {}),
                collapse_consecutive_tool_messages=True,
            )
        except IncrementalTokenizationError as e:
            self.logger.warning(f"Skipping example {example.get('__index', '')}: {e}")
            return None

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
            "sequence_lengths": [len(input_ids)],
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


class CatDataset(StatefulIterableDataset):
    """A dataset that concatenates samples into a single sequence with a fixed length."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len
        self.carry: dict[str, list] = defaultdict(list)
        self.carry_sequence_lengths: list[int] = []

    def state_dict(self) -> dict:
        state = {"dataset": self.dataset.state_dict()}
        if any(self.carry.values()) or self.carry_sequence_lengths:
            state["carry"] = dict(self.carry)
            state["carry_sequence_lengths"] = self.carry_sequence_lengths
        return state

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.carry = defaultdict(list, state_dict.get("carry", {}))
        self.carry_sequence_lengths = list(state_dict.get("carry_sequence_lengths", []))

    @staticmethod
    def _split_sequence_lengths(sequence_lengths: list[int], cutoff: int) -> tuple[list[int], list[int]]:
        emitted, carry = [], []
        remaining = cutoff
        for idx, length in enumerate(sequence_lengths):
            if remaining >= length:
                emitted.append(length)
                remaining -= length
                continue
            if remaining > 0:
                emitted.append(remaining)
            tail = length - remaining
            if tail > 0:
                carry.append(tail)
            carry.extend(sequence_lengths[idx + 1 :])
            break
        return emitted, carry

    def __iter__(self):
        packed_samples = defaultdict(list, self.carry)
        sequence_lengths = list(self.carry_sequence_lengths)
        seq_len = sum(sequence_lengths)
        for sample in self.dataset:
            sample_sequence_lengths = sample.get("sequence_lengths", [len(sample["input_ids"])])
            # Add sample to packed samples
            for key, value in sample.items():
                if key == "sequence_lengths":
                    continue
                assert isinstance(value, list), f"Value for key {key} must be a list"
                packed_samples[key].extend(value)

            # Update sequence length
            seq_len += len(sample["input_ids"])
            sequence_lengths.extend(sample_sequence_lengths)

            while seq_len >= self.seq_len:
                emitted = {}
                carry = {}
                for key, value in packed_samples.items():
                    assert isinstance(value, list), f"Value for key {key} must be a list"
                    emitted[key] = value[: self.seq_len]
                    carry[key] = value[self.seq_len :]

                emitted_lengths, carry_lengths = self._split_sequence_lengths(sequence_lengths, self.seq_len)
                emitted["sequence_lengths"] = emitted_lengths

                self.carry = defaultdict(list, carry)
                self.carry_sequence_lengths = carry_lengths
                packed_samples = self.carry
                sequence_lengths = self.carry_sequence_lengths
                seq_len = sum(sequence_lengths)
                yield emitted


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
                sequence_lengths = []
                for bucket_item in self.buckets[bucket_idx]:
                    num_samples += 1
                    sequence_lengths.append(len(bucket_item["input_ids"]))
                    for key, value in bucket_item.items():
                        pad_tokens = [0] * (self.bucket_sizes[bucket_idx] - len(value))
                        if key == "loss_mask":
                            num_tokens += len(value)
                            num_trainable_tokens += sum(value)
                            num_pad_tokens += len(pad_tokens)
                        packed_samples[key].append(value + pad_tokens)
                packed_samples["sequence_lengths"] = sequence_lengths
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
    sample = samples[0]
    return {
        "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device="cuda"),
        "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long, device="cuda"),
        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool, device="cuda"),
        "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long, device="cuda"),
        "sequence_lengths": torch.tensor(sample["sequence_lengths"], dtype=torch.int32, device="cuda"),
    }


def cat_collate(samples: list[Sample]) -> Batch:
    sample = samples[0]
    sequence_lengths = sample.get("sequence_lengths", [len(sample["input_ids"])])
    cu_seqlens, max_seqlen = build_cu_seqlens(sequence_lengths, device=torch.device("cuda"))
    return {
        "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool, device="cuda").unsqueeze(0),
        "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.int32, device="cuda"),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
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


def load_sft_dataset(config: SFTDataConfig) -> Dataset:
    """Load and interleave the raw HF dataset. This is the expensive I/O step."""
    logger = get_logger()
    if config.subsets is None and config.splits is None:
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, "train")],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is not None and config.splits is None:
        logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(subset, "train") for subset in config.subsets],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is None and config.splits is not None:
        logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, split) for split in config.splits],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    else:
        assert config.subsets is not None and config.splits is not None
        logger.debug(f"Loading datasets for subsets {config.subsets} with splits {config.splits}")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=list(zip(config.subsets, config.splits)),
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    non_dp_size: int = 1,
    *,
    max_epochs: int | None = None,
    raw_dataset: Dataset | None = None,
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size, seq_len=config.seq_len, length=config.length, input_ids=config.input_ids
        )
    elif config.type == "sft":
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config)
        sampler = SystemPromptSampler(config.system_prompt_pool_path) if config.system_prompt_pool_path else None
        return SFTDataset(
            raw_dataset,
            tokenizer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            system_prompt_sampler=sampler,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(dataset: StatefulIterableDataset, config: DataConfig) -> StatefulDataLoader:
    if config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
