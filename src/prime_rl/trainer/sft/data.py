import base64
import io
import json
import uuid
from collections import defaultdict
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Float, Int
from PIL import Image
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import DataConfig, LossMaskConfig, SFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import build_incremental_token_mask, deserialize_tool_calls, strip_message_content
from prime_rl.utils.logger import get_logger

STACKING_DATASET_BUCKET_TIMEOUT = 10


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    pixel_values: bytes | None
    pixel_values_shape: list[int] | None
    image_grid_thw: list[list[int]] | None


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    pixel_values: Float[Tensor, "num_patches patch_dim"] | None
    image_grid_thw: Int[Tensor, "num_images 3"] | None


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


def _extract_images_from_messages(messages: list[dict]) -> list[Image.Image]:
    """Extract PIL images from OpenAI-format messages with image_url content."""
    images = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:image"):
                    header, data = url.split(",", 1)
                    img_bytes = base64.b64decode(data)
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                elif url.startswith("file://"):
                    images.append(Image.open(url.removeprefix("file://")).convert("RGB"))
            elif item.get("type") == "image":
                url = item.get("image", "")
                if url.startswith("data:image"):
                    header, data = url.split(",", 1)
                    img_bytes = base64.b64decode(data)
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                elif url.startswith("file://"):
                    images.append(Image.open(url.removeprefix("file://")).convert("RGB"))
    return images


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
        processor=None,
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
        self.processor = processor

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

    @staticmethod
    def _parse_messages(field) -> list[dict]:
        """Parse a prompt/completion field that may be a JSON string or a list of dicts."""
        if isinstance(field, str):
            return json.loads(field)
        return field

    def _process(self, example: dict) -> dict | None:
        # Skip processing if no tokenizer was provided
        if self.tokenizer is None:
            return example

        # Assert that the example has a 'prompt' and 'completion' column
        if "prompt" not in example or "completion" not in example:
            raise ValueError("All examples in the dataset must have a 'prompt' and 'completion' column for SFT")

        # Deserialize tool call arguments from message list, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#handling-function-calls
        prompt = deserialize_tool_calls(self._parse_messages(example["prompt"]))
        completion = deserialize_tool_calls(self._parse_messages(example["completion"]))

        # Strip content from all messages so that incremental tokenization works
        # NOTE: This has the side effect that we do never train on leading or trailing whitespace
        prompt = strip_message_content(prompt)
        completion = strip_message_content(completion)

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

        input_ids, loss_mask = build_incremental_token_mask(
            self.tokenizer,
            prompt + completion,
            role_to_mask=should_mask,
            tools=tools,
            chat_template_kwargs=example.get("chat_template_kwargs", {}),
            collapse_consecutive_tool_messages=True,
        )

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

    def _process_multimodal(self, example: dict) -> dict | None:
        """Process a multimodal example using the processor for image handling.

        Note: loss_mask_config is not applied here — the loss mask is a simple
        prompt/completion binary split. Per-role masking would require incremental
        tokenization with expanded image tokens, which is not yet implemented.
        """
        if "prompt" not in example or "completion" not in example:
            raise ValueError("All examples in the dataset must have a 'prompt' and 'completion' column for SFT")

        prompt = self._parse_messages(example["prompt"])
        completion = self._parse_messages(example["completion"])
        all_messages = prompt + completion

        # Extract images from all messages
        images = _extract_images_from_messages(all_messages)
        if not images:
            return self._process(example)

        # Use apply_chat_template with original messages (keeps image markers for token expansion)
        # Prompt-only: determines loss mask boundary (uses only prompt images)
        prompt_images = _extract_images_from_messages(prompt)
        prompt_text = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_inputs = self.processor(
            text=[prompt_text], images=prompt_images or None, return_tensors="pt", padding=False
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Full sequence (prompt + completion) with all images
        full_text = self.processor.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=False)
        full_inputs = self.processor(text=[full_text], images=images, return_tensors="pt", padding=False)
        input_ids = full_inputs["input_ids"][0].tolist()

        # Build loss mask: False for prompt, True for completion
        loss_mask = [False] * min(prompt_len, len(input_ids)) + [True] * max(0, len(input_ids) - prompt_len)

        # Append EOS if missing
        if self.tokenizer.eos_token_id not in input_ids:
            self.logger.warning("Appending missing EOS token to multimodal sample")
            input_ids.append(cast(int, self.tokenizer.eos_token_id))
            loss_mask.append(True)

        # Prepare inputs (shift for next-token prediction)
        target_ids = input_ids[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping multimodal example {example.get('__index', '')} because no trainable tokens within seq_len"
            )
            return None

        # Extract pixel values and image grid info
        # pixel_values from the processor has no batch dim: [num_patches, patch_dim]
        pixel_values = full_inputs["pixel_values"]
        pv_bytes = pixel_values.to(torch.float32).numpy().tobytes()
        pv_shape = list(pixel_values.shape)
        image_grid_thw = full_inputs["image_grid_thw"].tolist()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
            "pixel_values": pv_bytes,
            "pixel_values_shape": pv_shape,
            "image_grid_thw": image_grid_thw,
        }

    def _has_images(self, example: dict) -> bool:
        """Check if an example contains image content."""
        for msg in self._parse_messages(example.get("prompt", [])) + self._parse_messages(
            example.get("completion", [])
        ):
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ("image_url", "image"):
                        return True
        return False

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

            # Process example — route multimodal samples through processor path
            example_dict = cast(dict, example)
            if self.processor is not None and self._has_images(example_dict):
                processed_example = self._process_multimodal(example_dict)
            else:
                processed_example = self._process(example_dict)

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


class SingleSampleDataset(StatefulIterableDataset):
    """Wraps a dataset to pad each sample individually. Used for multimodal SFT where packing is not possible."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len

    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def __iter__(self):
        for sample in self.dataset:
            seq_len = self.seq_len
            is_multimodal = sample.get("pixel_values") is not None

            # Skip multimodal samples that exceed seq_len (truncation would break
            # the alignment between image_pad tokens and pixel_values)
            if is_multimodal and len(sample["input_ids"]) > seq_len:
                get_logger().warning(
                    f"Skipping multimodal SFT sample that exceeds seq_len ({len(sample['input_ids'])} > {seq_len})"
                )
                continue

            # Truncate token fields (text-only samples only at this point)
            for key in ["input_ids", "target_ids", "position_ids", "loss_mask"]:
                sample[key] = sample[key][:seq_len]
            # Pad token fields to seq_len
            pad_len = seq_len - len(sample["input_ids"])
            if pad_len > 0:
                sample["input_ids"] = sample["input_ids"] + [0] * pad_len
                sample["target_ids"] = sample["target_ids"] + [0] * pad_len
                sample["position_ids"] = sample["position_ids"] + [0] * pad_len
                sample["loss_mask"] = sample["loss_mask"] + [False] * pad_len
            yield sample


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


def single_collate(samples: list[Sample]) -> Batch:
    """Collate for single-sample batches (multimodal SFT)."""
    sample = samples[0]
    batch: Batch = {
        "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool, device="cuda").unsqueeze(0),
        "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "pixel_values": None,
        "image_grid_thw": None,
    }
    if sample.get("pixel_values") is not None:
        batch["pixel_values"] = (
            torch.frombuffer(bytearray(sample["pixel_values"]), dtype=torch.float32)
            .reshape(sample["pixel_values_shape"])
            .to("cuda")
        )
        batch["image_grid_thw"] = torch.tensor(sample["image_grid_thw"], dtype=torch.long, device="cuda")
    return batch


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
    processor=None,
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size, seq_len=config.seq_len, length=config.length, input_ids=config.input_ids
        )
    elif config.type == "sft":
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config)
        return SFTDataset(
            raw_dataset,
            tokenizer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            processor=processor,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(dataset: StatefulIterableDataset, config: DataConfig) -> StatefulDataLoader:
    if config.pack_function == "single":
        single_dataset = SingleSampleDataset(dataset, config.seq_len)
        return StatefulDataLoader(single_dataset, batch_size=1, collate_fn=single_collate)
    elif config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
