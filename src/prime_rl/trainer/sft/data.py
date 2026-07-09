import glob
import hashlib
import json
import subprocess
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from renderers.base import (
    MultiModalData,
    PlaceholderRange,
    Renderer,
    build_training_sample,
)
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import DataConfig, LossMaskConfig, SFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger

STACKING_DATASET_BUCKET_TIMEOUT = 10


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    seq_lens: list[int]
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: list[int] | None


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    seq_lens: Int[Tensor, "packed"] | None
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[Tensor, "batch seq"] | None


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
                "seq_lens": [seq_len],
                "mm_kwargs": None,
                "mm_token_type_ids": None,
            }
            self.num_samples["fake"] += 1
            self.num_tokens["fake"] += len(input_ids)
            yield fake_sample


def _flatten_mm_items(
    mm_items: dict[str, list[dict[str, Any]]],
) -> dict[str, Tensor]:
    """Fold per-image renderer items into a flat dict of concatenated tensors.

    Each content type's list-of-dicts (one per image / video) is reduced to one
    tensor per kwarg by concatenating along dim 0. The kwarg names come from the
    renderer's processor (e.g. ``pixel_values``, ``image_grid_thw``) and stay
    model-agnostic: the trainer ``**``-unpacks the result into ``forward()``.
    """
    out: dict[str, Tensor] = {}
    for items in mm_items.values():
        for item in items:
            for k, v in item.items():
                if not isinstance(v, (np.ndarray, Tensor)):
                    continue
                v = torch.as_tensor(v)
                if k in out:
                    out[k] = torch.cat([out[k], v], dim=0)
                else:
                    out[k] = v
    return out


def _drop_null_fields(value: Any) -> Any:
    """Recursively strip ``None``-valued keys from dict structures.

    PyArrow's JSON loader unifies schemas across rows, so heterogeneous
    OAI content blocks (text vs image_url) end up with all union keys
    filled with ``None`` where absent. That confuses permissive
    content-type predicates inside renderers (e.g. ``"image_url" in item``
    returns ``True`` even when the value is null). Strip the noise before
    handing messages off to the renderer.
    """
    if isinstance(value, dict):
        return {k: _drop_null_fields(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_null_fields(v) for v in value]
    return value


def _find_image_safe_cut(budget: int, mm: MultiModalData | None) -> int:
    """Largest position ≤ ``budget`` that doesn't fall inside an image / video
    placeholder run.
    """
    if mm is None or not mm.mm_placeholders:
        return budget
    cut = budget
    for ranges in mm.mm_placeholders.values():
        for ph in ranges:
            if ph.offset < cut < ph.offset + ph.length:
                cut = ph.offset
    return cut


def _truncate_mm_data(mm: MultiModalData, cut: int) -> MultiModalData:
    """Drop ``mm_items`` / ``mm_placeholders`` whose ranges extend past ``cut``."""
    new_placeholders: dict[str, list[PlaceholderRange]] = {}
    new_items: dict[str, list[dict[str, Any]]] = {}
    new_hashes: dict[str, list[str]] = {}
    for ctype, ranges in mm.mm_placeholders.items():
        keep = [i for i, ph in enumerate(ranges) if ph.offset + ph.length <= cut]
        if not keep:
            continue
        new_placeholders[ctype] = [ranges[i] for i in keep]
        new_items[ctype] = [mm.mm_items[ctype][i] for i in keep]
        if ctype in mm.mm_hashes:
            new_hashes[ctype] = [mm.mm_hashes[ctype][i] for i in keep]
    return MultiModalData(
        mm_hashes=new_hashes,
        mm_placeholders=new_placeholders,
        mm_items=new_items,
    )


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt/completion or raw messages format."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer | None,
        renderer: Renderer | None = None,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        multimodal: bool = False,
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
        self.renderer = renderer
        self.multimodal = multimodal
        self._warned_chat_template_kwargs = False

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
        if self.tokenizer is None:
            return example
        if self.renderer is None:
            raise ValueError("SFT processing requires a renderer.")

        def resolve_messages(example: dict) -> list[dict]:
            # `messages` takes precedence over explicit split fields and is interpreted
            # as a whole-chat training sample with an empty prompt. Null-check rather
            # than key-check: Arrow schema union adds `messages: null` to
            # prompt/completion rows whenever other rows have a `messages` column.
            if example.get("messages") is not None:
                messages = normalize_messages(example["messages"], default_role="assistant")
            elif example.get("prompt") is not None and example.get("completion") is not None:
                messages = normalize_messages(example["prompt"], default_role="user") + normalize_messages(
                    example["completion"], default_role="assistant"
                )
            else:
                raise ValueError(
                    "All examples in the dataset must have either a 'messages' column "
                    "or both 'prompt' and 'completion' columns for SFT"
                )

            # Strip nulls before deserializing so genuine nulls inside tool-call
            # argument strings survive.
            messages = [_drop_null_fields(m) for m in messages]
            return deserialize_tool_calls(messages)

        messages = resolve_messages(example)

        # Parse available tools, if present - assumes OAI format. Accepts either
        # `tools` or `tool_defs` (the verifiers rollout format), as either a
        # JSON-encoded string of a list or a list of dicts; verifiers-shaped
        # tools are converted to OAI form for the chat template.
        raw_tools = example.get("tools", example.get("tool_defs"))
        if not raw_tools:
            tools = []
        else:
            if isinstance(raw_tools, str):
                raw_tools = json.loads(raw_tools)
            tools = [
                t
                if isinstance(t, dict) and t.get("type") == "function" and "function" in t
                else {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "parameters": t.get("parameters"),
                        **({} if t.get("strict") is None else {"strict": t["strict"]}),
                    },
                }
                for t in raw_tools
            ]

        def should_mask(message: dict) -> bool:
            assert "role" in message, "Message must have a role"
            match message["role"]:
                case "user":
                    return self.loss_mask_config.user
                case "assistant":
                    return self.loss_mask_config.assistant
                case "system":
                    return self.loss_mask_config.system
                case "tool":
                    return self.loss_mask_config.tool
                case _:
                    raise ValueError(f"Invalid message role: {message['role']}")

        if example.get("chat_template_kwargs") and not self._warned_chat_template_kwargs:
            self.logger.warning(
                "Ignoring per-example chat_template_kwargs; renderers only take "
                "template kwargs run-wide via the [renderer] config."
            )
            self._warned_chat_template_kwargs = True

        # Non-assistant roles are opted into the loss via the renderer's
        # body-only path: the message content is trained, not the role
        # scaffolding (e.g. <tool_response> tags) the harness emits.
        content_sft_roles = {role for role in ("user", "system", "tool") if getattr(self.loss_mask_config, role)}
        sample = build_training_sample(
            self.renderer,
            messages,
            role_to_mask=should_mask,
            tools=tools,
            content_sft_roles=content_sft_roles or None,
        )
        input_ids = list(sample.token_ids)
        loss_mask = list(sample.loss_mask)
        mm = sample.multi_modal_data
        mm_token_type_ids = list(sample.mm_token_type_ids) if sample.mm_token_type_ids is not None else None
        if mm is not None and mm.mm_items and not self.multimodal:
            raise ValueError(
                "Renderer produced multimodal data but [model.vlm] is not set. "
                "Set [model.vlm] to train on multimodal samples."
            )

        was_mm_truncated = False
        if mm is not None:
            # The causal shift below drops one input token; keep one extra raw token
            # so truncated multimodal samples can still fill the configured length.
            budget = self.seq_len + 1
            if len(input_ids) > budget:
                was_mm_truncated = True
                cut = _find_image_safe_cut(budget, mm)
                self.logger.debug(
                    f"Truncating example {example.get('__index', '')} from "
                    f"{len(input_ids)} → {cut} tokens (budget={budget})"
                )
                input_ids = input_ids[:cut]
                loss_mask = loss_mask[:cut]
                if mm_token_type_ids is not None:
                    mm_token_type_ids = mm_token_type_ids[:cut]
                if mm.mm_items:
                    mm = _truncate_mm_data(mm, cut)

        # If EOS token is not found, manually append it (keep mm_token_type_ids aligned).
        if not self.tokenizer.eos_token_id in input_ids:
            if was_mm_truncated:
                return None
            self.logger.warning(
                f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. Is something wrong with the chat template? Manually appending EOS token..."
            )
            input_ids.append(cast(int, self.tokenizer.eos_token_id))
            loss_mask.append(True)
            if mm_token_type_ids is not None:
                mm_token_type_ids.append(0)

        # Causal shift: model predicts next token from current.
        target_ids = input_ids.copy()[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:-1]

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return None

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

        mm_kwargs: dict[str, Tensor] | None = None
        if mm is not None and mm.mm_items:
            mm_kwargs = _flatten_mm_items(mm.mm_items)
            if any("video" in key for key in mm_kwargs):
                raise ValueError("Video SFT is not supported; sample contains video inputs")
        if mm_token_type_ids is not None:
            assert len(mm_token_type_ids) == len(input_ids)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
            "seq_lens": [len(input_ids)],
            "mm_kwargs": mm_kwargs,
            "mm_token_type_ids": mm_token_type_ids,
        }

    def __iter__(self):
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
    """A dataset that concatenates samples into a single sequence with a fixed length.

    Text-only and multimodal samples share the same pack representation. When a
    pack includes ``mm_token_type_ids``, text-only spans are represented by zeros.
    """

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len

    def state_dict(self) -> dict:
        # Known limitation: the overflow sample pending at a yield boundary is
        # not persisted, so a checkpoint resume drops at most one sample per
        # dataloader worker.
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def __iter__(self):
        packed_samples = defaultdict(list)
        packed_samples["mm_kwargs"] = None
        packed_samples["mm_token_type_ids"] = None
        seq_len = 0
        for sample in self.dataset:
            sample_len = len(sample["input_ids"])
            would_overflow = seq_len + sample_len > self.seq_len
            if seq_len > 0 and would_overflow:
                yield self._finalize_pack(packed_samples, self.seq_len)
                packed_samples = defaultdict(list)
                packed_samples["mm_kwargs"] = None
                packed_samples["mm_token_type_ids"] = None
                seq_len = 0

            existing_len = len(packed_samples["input_ids"])
            for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):
                value = sample[key]
                assert isinstance(value, list)
                packed_samples[key].extend(value)
            packed_samples["seq_lens"].append(sample_len)

            sample_mm_kwargs = sample.get("mm_kwargs")
            sample_mm_type_ids = sample.get("mm_token_type_ids")
            if sample_mm_kwargs is None:
                if packed_samples["mm_token_type_ids"] is not None:
                    packed_samples["mm_token_type_ids"].extend([0] * sample_len)
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
                        packed_samples["mm_kwargs"][key] = torch.cat(
                            [packed_samples["mm_kwargs"][key], value], dim=0
                        )

                if packed_samples["mm_token_type_ids"] is None and sample_mm_type_ids is not None:
                    packed_samples["mm_token_type_ids"] = [0] * existing_len
                if packed_samples["mm_token_type_ids"] is not None:
                    packed_samples["mm_token_type_ids"].extend(sample_mm_type_ids or [0] * sample_len)

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
            k: packed[k][:seq_len] for k in ("input_ids", "position_ids", "loss_mask", "target_ids")
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
        pad_len = seq_len - len(result["input_ids"])
        if pad_len > 0:
            result["input_ids"].extend([0] * pad_len)
            result["position_ids"].extend(range(pad_len))
            result["loss_mask"].extend([False] * pad_len)
            result["target_ids"].extend([0] * pad_len)
            result["seq_lens"].append(pad_len)
        result["mm_kwargs"] = packed["mm_kwargs"]
        if packed["mm_token_type_ids"] is not None:
            result["mm_token_type_ids"] = packed["mm_token_type_ids"][:seq_len] + [0] * pad_len
        else:
            result["mm_token_type_ids"] = None
        return result


class StackDataset(StatefulIterableDataset):
    """A dataset that stacks samples into batch with a fixed area.

    Text-only path. Multimodal samples (with ``mm_kwargs``) bypass stack
    bucketing and are emitted one at a time; use ``pack_function = "cat"``
    for multimodal sequence packing.
    """

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
            if sample.get("mm_kwargs") is not None:
                # Multimodal samples bypass bucketing.
                self.step += 1
                yield sample
                continue

            # Truncate sample if it's longer than max area
            len_sample = len(sample["input_ids"])
            if len_sample > self.max_area:
                for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):
                    value = sample[key]
                    assert isinstance(value, list)
                    sample[key] = value[: self.max_area]
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
                        for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):
                            dummy_sample[key] = [0]
                        dummy_sample["seq_lens"] = [1]
                        dummy_sample["mm_kwargs"] = None
                        dummy_sample["mm_token_type_ids"] = None
                        self.buckets[bucket_idx].append(dummy_sample)

                packed_samples = defaultdict(list)
                num_samples, num_tokens, num_trainable_tokens, num_pad_tokens = 0, 0, 0, 0
                for bucket_item in self.buckets[bucket_idx]:
                    num_samples += 1
                    for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):
                        value = bucket_item[key]
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
                packed_samples["seq_lens"] = None
                packed_samples["mm_kwargs"] = None
                packed_samples["mm_token_type_ids"] = None
                yield packed_samples
                self.step += 1
                self.buckets[bucket_idx] = []
                self.bucket_timers[bucket_idx] = None
            else:
                if self.bucket_timers[bucket_idx] is None:
                    self.bucket_timers[bucket_idx] = self.step


def stack_collate(samples: list[Sample]) -> Batch:
    mm_kwargs = _move_mm_kwargs_to_cuda(samples[0].get("mm_kwargs"))
    if mm_kwargs is not None:
        sample = samples[0]
        mm_type_ids = sample.get("mm_token_type_ids")
        # Multimodal samples are emitted solo by StackDataset with 1-D fields; add
        # the batch dim the bucketed text path gets from its list-of-lists so the
        # trainer receives [batch, seq] like everywhere else.
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool, device="cuda").unsqueeze(0),
            "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "seq_lens": torch.tensor(sample["seq_lens"], dtype=torch.long, device="cuda"),
            "mm_kwargs": mm_kwargs,
            "mm_token_type_ids": (
                torch.tensor(mm_type_ids, dtype=torch.long, device="cuda").unsqueeze(0)
                if mm_type_ids is not None
                else None
            ),
        }
    return {
        "input_ids": torch.tensor(samples[0]["input_ids"], dtype=torch.long, device="cuda"),
        "position_ids": torch.tensor(samples[0]["position_ids"], dtype=torch.long, device="cuda"),
        "loss_mask": torch.tensor(samples[0]["loss_mask"], dtype=torch.bool, device="cuda"),
        "target_ids": torch.tensor(samples[0]["target_ids"], dtype=torch.long, device="cuda"),
        "seq_lens": None,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }


def cat_collate(samples: list[Sample]) -> Batch:
    # CatDataset emits exactly one packed sample per dataloader item. For
    # multimodal packs, keep the packed sequence as batch size 1 and move the
    # concatenated model-forward kwargs to CUDA.
    if len(samples) == 1 and samples[0].get("mm_kwargs") is not None:
        sample = samples[0]
        mm_type_ids = sample.get("mm_token_type_ids")
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool, device="cuda").unsqueeze(0),
            "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "seq_lens": torch.tensor(sample["seq_lens"], dtype=torch.long, device="cuda"),
            "mm_kwargs": _move_mm_kwargs_to_cuda(sample["mm_kwargs"]),
            "mm_token_type_ids": (
                torch.tensor(mm_type_ids, dtype=torch.long, device="cuda").unsqueeze(0)
                if mm_type_ids is not None
                else None
            ),
        }
    return {
        "input_ids": torch.stack([torch.tensor(sample["input_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "position_ids": torch.stack([torch.tensor(sample["position_ids"]) for sample in samples], dim=0)
        .long()
        .to("cuda"),
        "loss_mask": torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples], dim=0).bool().to("cuda"),
        "target_ids": torch.stack([torch.tensor(sample["target_ids"]) for sample in samples], dim=0).long().to("cuda"),
        # CatDataset emits exactly one packed sample per dataloader item.
        "seq_lens": torch.tensor(samples[0]["seq_lens"], dtype=torch.long, device="cuda")
        if len(samples) == 1
        else None,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }


def _move_mm_kwargs_to_cuda(mm_kwargs: dict[str, Tensor] | None) -> dict[str, Tensor] | None:
    if mm_kwargs is None:
        return None
    return {k: v.to("cuda") for k, v in mm_kwargs.items()}


def setup_and_interleave_datasets(
    dataset_name: str,
    subsets_and_splits: list[tuple[str | None, str]],
    probabilities: list[float] | None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"],
    seed: int = 0,
    data_files: str | list[str] | None = None,
) -> Dataset:
    logger = get_logger()
    datasets = []
    for subset, split in subsets_and_splits:
        logger.debug(f"Loading dataset {dataset_name} with {subset=}, {split=}, {data_files=}")
        if data_files is not None:
            dataset = cast(Dataset, load_dataset(dataset_name, data_files=data_files, split=split))
        else:
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


def _normalize_oai_record(record: dict, multimodal: bool = False) -> dict:
    """Coerce variable JSON shapes to forms PyArrow's JSON loader can handle.

    PyArrow infers schemas across rows and crashes on shape drift:
    - ``messages[].content`` may be string or list of blocks. For multimodal
      data we wrap strings in ``[{"type": "text", "text": content}]`` so the
      column unifies with image-block content. Text-only data keeps string
      content untouched, so the legacy tokenizer / chat-template path (which
      expects strings) is unaffected.
    - ``tool_calls`` (via ``function.arguments`` and heterogeneous per-tool
      metadata) and the ``tools`` / ``tool_defs`` arrays carry per-row schemas
      that Arrow can't represent, and even shape-stable tool_calls crash when
      early batches are all-null (Arrow infers ``null`` and can't cast later
      ``list<struct>`` batches to it). Stringify each to a JSON string — a
      string column always unifies — with null/empty ``tool_calls`` coerced to
      ``""``. ``deserialize_tool_calls`` reverses this downstream.

    Both the ``messages`` layout and the TRL ``prompt``/``completion`` layout
    (where each column is itself a message list) are normalized.
    """
    new_record = dict(record)

    for key in ("messages", "prompt", "completion"):
        messages = new_record.get(key)
        if isinstance(messages, list):
            new_record[key] = [_normalize_oai_message(m, multimodal=multimodal) for m in messages]

    for key in ("tools", "tool_defs"):
        tools = new_record.get(key)
        if isinstance(tools, (list, dict)):
            new_record[key] = json.dumps(tools)

    return new_record


def _normalize_oai_message(message: Any, multimodal: bool = False) -> Any:
    if not isinstance(message, dict):
        return message
    message = dict(message)
    content = message.get("content")
    if multimodal and isinstance(content, str):
        message["content"] = [{"type": "text", "text": content}]
    if "tool_calls" in message:
        tool_calls = message["tool_calls"]
        if not isinstance(tool_calls, str):
            # "" (not None) for no-tool messages: an all-null column chunk infers
            # Arrow type null, which can't unify with string chunks
            message["tool_calls"] = json.dumps(tool_calls) if tool_calls else ""
    return message


def _resolve_local_data_files(data_files: list[str] | None, multimodal: bool = False) -> list[str] | None:
    """Materialize local files HF datasets can ingest.

    Glob patterns are expanded first (each match is materialized separately),
    then two transforms are applied side-by-side under ``$TMPDIR``:
    - ``.zst`` files are decompressed (HF handles gz/bz2/xz transparently
      but not zstd).
    - JSONL files are normalized via :func:`_normalize_oai_record` so the
      Arrow JSON loader can infer a stable schema across rows.

    Both steps stream line-by-line — no full-file load into memory — so large
    full datasets go through the same path as smaller samples.

    The temp filename embeds a digest of the source's absolute path, mtime,
    size, and the ``multimodal`` flag, so same-basename files from different
    directories (or a changed/stale source) never collide on or silently reuse
    a previous run's materialized output.

    Each node materializes into its own (node-local) ``$TMPDIR``, so only the
    local-rank-0 process on each node performs the work; a global barrier then
    syncs all ranks. Gating per-node (not on global rank 0 alone) means ranks
    on other nodes find their files instead of waiting on a path that was only
    written on node 0. The gate also avoids parallel ranks racing to write the
    same path (byte-interleaved output PyArrow rejects).
    """
    if not data_files:
        return data_files
    expanded: list[str] = []
    for path in data_files:
        if any(char in path for char in "*?["):
            matches = sorted(glob.glob(path))
            if not matches:
                raise FileNotFoundError(f"No files match data_files pattern {path!r}")
            expanded.extend(matches)
        else:
            expanded.append(path)
    is_writer = not torch.distributed.is_initialized() or get_world().local_rank == 0
    resolved: list[str] = []
    for path in expanded:
        p = Path(path)
        if p.suffix in (".zst", ".jsonl"):
            stat = p.stat()
            digest = hashlib.sha1(f"{p.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{multimodal}".encode()).hexdigest()[
                :12
            ]
            tmp = Path(tempfile.gettempdir()) / f"{p.stem}.{digest}.prime_rl_normalized.jsonl"
            if is_writer and (not tmp.exists() or tmp.stat().st_size == 0):
                if p.suffix == ".zst":
                    get_logger().info(f"Decompressing + normalizing {p} → {tmp}")
                    decompressed = Path(tempfile.gettempdir()) / f"{p.stem}.{digest}.prime_rl_raw"
                    if not decompressed.exists() or decompressed.stat().st_size == 0:
                        part = decompressed.with_name(decompressed.name + ".part")
                        subprocess.run(["zstd", "-d", "-f", "-o", str(part), str(p)], check=True)
                        part.replace(decompressed)
                    _normalize_jsonl(decompressed, tmp, multimodal=multimodal)
                else:
                    get_logger().info(f"Normalizing {p} → {tmp}")
                    _normalize_jsonl(p, tmp, multimodal=multimodal)
            resolved.append(str(tmp))
        else:
            resolved.append(str(p))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return resolved


def _normalize_jsonl(src: Path, dst: Path, multimodal: bool = False) -> None:
    """Stream-rewrite ``src`` JSONL to ``dst`` with uniform OAI message shape.

    Writes to a sidecar path and atomically renames into place, so an
    interrupted run never leaves a truncated file that a later run would
    mistake for a valid cache.
    """
    part = dst.with_name(dst.name + ".part")
    with src.open("r") as f_in, part.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            record = _normalize_oai_record(json.loads(line), multimodal=multimodal)
            f_out.write(json.dumps(record))
            f_out.write("\n")
    part.replace(dst)


def load_sft_dataset(config: SFTDataConfig, multimodal: bool = False) -> Dataset:
    """Load and interleave the raw HF dataset. This is the expensive I/O step."""
    logger = get_logger()
    data_files = _resolve_local_data_files(config.data_files, multimodal=multimodal)
    if config.subsets is None and config.splits is None:
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, "train")],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
            data_files=data_files,
        )
    elif config.subsets is not None and config.splits is None:
        logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(subset, "train") for subset in config.subsets],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
            data_files=data_files,
        )
    elif config.subsets is None and config.splits is not None:
        logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, split) for split in config.splits],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
            data_files=data_files,
        )
    else:
        assert config.subsets is not None and config.splits is not None
        logger.debug(f"Loading datasets for subsets {config.subsets} with splits {config.splits}")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=list(zip(config.subsets, config.splits)),
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
            data_files=data_files,
        )


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    non_dp_size: int = 1,
    *,
    max_epochs: int | None = None,
    raw_dataset: Dataset | None = None,
    renderer: Renderer | None = None,
    multimodal: bool = False,
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size,
            seq_len=config.seq_len,
            length=config.length,
            input_ids=config.input_ids,
        )
    elif config.type == "sft":
        if renderer is None:
            raise ValueError("SFT data requires a renderer.")
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config, multimodal=multimodal)
        return SFTDataset(
            raw_dataset,
            tokenizer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            renderer=renderer,
            multimodal=multimodal,
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
