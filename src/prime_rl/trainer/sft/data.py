import json
import subprocess
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from renderers.base import (
    MultiModalData,
    Renderer,
    build_training_sample,
)
from renderers.base import (
    is_multimodal as is_multimodal_renderer,
)
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

STACKING_DATASET_BUCKET_TIMEOUT = 10


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    # Optional multimodal payload. Keys are model-specific kwarg names produced
    # by the renderer (e.g. ``pixel_values`` + ``image_grid_thw`` for Qwen-VL).
    # Tensors are concatenated across all media items of the sample along dim 0.
    mm_kwargs: dict[str, Tensor] | None
    # Per-token modality flag aligned with ``input_ids`` after the causal shift.
    mm_token_type_ids: list[int] | None


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    # See ``Sample`` — same payload, on CUDA. ``None`` for text-only batches.
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
                if not isinstance(v, Tensor):
                    continue
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
    placeholder run. Splitting a placeholder run would desync the model's
    ``image_pad`` tokens from the corresponding ``pixel_values`` block.
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
    """Drop ``mm_items`` / ``mm_placeholders`` whose ranges extend past ``cut``.

    Keeps every entry whose placeholder lies fully within ``[0, cut)``, preserving
    one-to-one alignment between surviving placeholders and surviving pixel
    buffers.
    """
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
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        renderer: Renderer | None = None,
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
        self._warned_chat_template_kwargs = False
        self.is_multimodal = renderer is not None and is_multimodal_renderer(renderer)

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
            messages = deserialize_tool_calls(messages)
            # Drop null-valued fields (browser-agent OAI records carry many)
            # before rendering / stripping.
            messages = [_drop_null_fields(m) for m in messages]
            # Strip content from all messages so that incremental tokenization works.
            # No-op on multimodal list content (only strips string content), so it
            # is safe for VLM samples.
            return strip_message_content(messages)

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

        # Multimodal payload, populated only on the renderer path for VLMs.
        mm: MultiModalData | None = None
        mm_token_type_ids: list[int] | None = None

        if self.renderer is not None:
            if example.get("chat_template_kwargs") and not self._warned_chat_template_kwargs:
                self.logger.warning(
                    "Example carries chat_template_kwargs but a renderer is configured; "
                    "renderers don't forward chat_template_kwargs (model-specific "
                    "renderers bake their template behavior in). These kwargs will "
                    "be ignored. Further warnings suppressed for this dataset."
                )
                self._warned_chat_template_kwargs = True

            sample = build_training_sample(
                self.renderer,
                messages,
                role_to_mask=should_mask,
                tools=tools,
            )
            input_ids = list(sample.token_ids)
            loss_mask = list(sample.loss_mask)
            mm = sample.multi_modal_data
            mm_token_type_ids = list(sample.mm_token_type_ids) if sample.mm_token_type_ids is not None else None

            # VLM samples carry per-image pixel buffers that can't be packed
            # across samples, so we truncate to seq_len here (text-only samples
            # are left full-length for the downstream Cat/Stack dataset). The
            # cut is pulled back so it never splits an <|image_pad|> run, and
            # mm_items / mm_token_type_ids are sliced in lockstep so surviving
            # placeholders stay 1-to-1 with surviving pixel buffers. Budget is
            # seq_len + 1 because the causal shift below drops one token.
            if mm is not None:
                budget = self.seq_len + 1
                if len(input_ids) > budget:
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
        else:
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

        # If EOS token is not found, manually append it (keep mm_token_type_ids aligned).
        if not self.tokenizer.eos_token_id in input_ids:
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
            mm_token_type_ids = mm_token_type_ids[1:]

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
        if mm_token_type_ids is not None:
            assert len(mm_token_type_ids) == len(input_ids)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
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

            # If processed example is None, skip it (e.g. if no trainable tokens fit)
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

    Text-only: packs multiple samples back-to-back into a fixed-length window.
    Multimodal samples (with ``mm_kwargs``) are passed through individually —
    packing pixel buffers across samples while keeping image placeholders
    aligned isn't worth the bookkeeping for an effective micro_batch_size of 1.
    """

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
            if sample.get("mm_kwargs") is not None:
                # Yield any in-flight text pack first to keep ordering deterministic,
                # then yield the multimodal sample standalone.
                if seq_len > 0:
                    yield self._finalize_text_pack(packed_samples, self.seq_len)
                    packed_samples, seq_len = defaultdict(list), 0
                yield sample
                continue

            for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):
                value = sample[key]
                assert isinstance(value, list)
                packed_samples[key].extend(value)
            seq_len += len(sample["input_ids"])

            if seq_len >= self.seq_len:
                yield self._finalize_text_pack(packed_samples, self.seq_len)
                packed_samples, seq_len = defaultdict(list), 0

    def _finalize_text_pack(self, packed: dict[str, list], seq_len: int) -> dict:
        result: dict[str, Any] = {k: v[:seq_len] for k, v in packed.items()}
        result["mm_kwargs"] = None
        result["mm_token_type_ids"] = None
        return result


class StackDataset(StatefulIterableDataset):
    """A dataset that stacks samples into batch with a fixed area.

    Text-only path. Multimodal samples (with ``mm_kwargs``) bypass bucketing
    entirely and are emitted one at a time so pixel buffers stay aligned with
    their image placeholders.
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
                yield {**sample, "_solo": True}
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
    sample = samples[0]
    mm_kwargs = _move_mm_kwargs_to_cuda(sample.get("mm_kwargs"))
    mm_type_ids = sample.get("mm_token_type_ids")
    return {
        "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device="cuda"),
        "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long, device="cuda"),
        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool, device="cuda"),
        "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long, device="cuda"),
        "mm_kwargs": mm_kwargs,
        "mm_token_type_ids": (
            torch.tensor(mm_type_ids, dtype=torch.long, device="cuda") if mm_type_ids is not None else None
        ),
    }


def cat_collate(samples: list[Sample]) -> Batch:
    # Multimodal samples are emitted solo by CatDataset (not packed); treat
    # single-sample batches with mm_kwargs as the multimodal path.
    if len(samples) == 1 and samples[0].get("mm_kwargs") is not None:
        sample = samples[0]
        mm_type_ids = sample.get("mm_token_type_ids")
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool, device="cuda").unsqueeze(0),
            "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
            "mm_kwargs": _move_mm_kwargs_to_cuda(sample["mm_kwargs"]),
            "mm_token_type_ids": (
                torch.tensor(mm_type_ids, dtype=torch.long, device="cuda").unsqueeze(0)
                if mm_type_ids is not None
                else None
            ),
        }
    return {
        "input_ids": torch.stack([torch.tensor(s["input_ids"]) for s in samples], dim=0).long().to("cuda"),
        "position_ids": torch.stack([torch.tensor(s["position_ids"]) for s in samples], dim=0).long().to("cuda"),
        "loss_mask": torch.stack([torch.tensor(s["loss_mask"]) for s in samples], dim=0).bool().to("cuda"),
        "target_ids": torch.stack([torch.tensor(s["target_ids"]) for s in samples], dim=0).long().to("cuda"),
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


def _normalize_oai_record(record: dict) -> dict:
    """Coerce variable JSON shapes to forms PyArrow's JSON loader can handle.

    PyArrow infers schemas across rows and crashes on shape drift:
    - ``messages[].content`` may be string or list of blocks → wrap strings in
      ``[{"type": "text", "text": content}]``.
    - ``messages[].tool_calls[].function.arguments`` and the ``tools`` array
      both carry per-row schemas (browser-agent action params, tool-specific
      JSON schemas, etc.) that Arrow can't represent → stringify to canonical
      OAI JSON form. Renderers accept either.
    """
    new_record = dict(record)

    messages = new_record.get("messages")
    if isinstance(messages, list):
        new_messages = []
        for m in messages:
            if not isinstance(m, dict):
                new_messages.append(m)
                continue
            m = dict(m)
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = [{"type": "text", "text": content}]
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                m["tool_calls"] = [_stringify_tool_call_args(tc) for tc in tool_calls]
            new_messages.append(m)
        new_record["messages"] = new_messages

    tools = new_record.get("tools")
    if isinstance(tools, (list, dict)):
        new_record["tools"] = json.dumps(tools)

    return new_record


def _stringify_tool_call_args(tool_call: Any) -> Any:
    if not isinstance(tool_call, dict):
        return tool_call
    tc = dict(tool_call)
    func = tc.get("function")
    if isinstance(func, dict):
        func = dict(func)
        args = func.get("arguments")
        if isinstance(args, (dict, list)):
            func["arguments"] = json.dumps(args)
        tc["function"] = func
    return tc


def _resolve_local_data_files(data_files: list[str] | None) -> list[str] | None:
    """Materialize local files HF datasets can ingest.

    Two transforms applied side-by-side under ``$TMPDIR``:
    - ``.zst`` files are decompressed (HF handles gz/bz2/xz transparently
      but not zstd).
    - JSONL files are normalized via :func:`_normalize_oai_record` so the
      Arrow JSON loader can infer a stable schema across rows.

    Both steps stream line-by-line — no full-file load into memory — so the
    14GB full Plex dataset goes through the same path as the 38MB sample.

    Only the global-rank-0 process performs the work; other ranks wait on a
    barrier and then read the materialized files. Without this gate, parallel
    ranks racing to write the same TMPDIR path produced byte-interleaved
    output that PyArrow rejected with "Missing closing quotation mark".
    """
    if not data_files:
        return data_files
    is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    resolved: list[str] = []
    for path in data_files:
        p = Path(path)
        if p.suffix == ".zst":
            tmp = Path(tempfile.gettempdir()) / (p.stem + ".prime_rl_normalized.jsonl")
            if is_rank_zero and (not tmp.exists() or tmp.stat().st_size == 0):
                get_logger().info(f"Decompressing + normalizing {p} → {tmp}")
                decompressed = Path(tempfile.gettempdir()) / (p.stem + ".prime_rl_raw")
                if not decompressed.exists() or decompressed.stat().st_size == 0:
                    subprocess.run(["zstd", "-d", "-f", "-o", str(decompressed), str(p)], check=True)
                _normalize_jsonl(decompressed, tmp)
            resolved.append(str(tmp))
        elif p.suffix == ".jsonl":
            tmp = Path(tempfile.gettempdir()) / (p.stem + ".prime_rl_normalized.jsonl")
            if is_rank_zero and (not tmp.exists() or tmp.stat().st_size == 0):
                get_logger().info(f"Normalizing {p} → {tmp}")
                _normalize_jsonl(p, tmp)
            resolved.append(str(tmp))
        else:
            resolved.append(str(p))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return resolved


def _normalize_jsonl(src: Path, dst: Path) -> None:
    """Stream-rewrite ``src`` JSONL to ``dst`` with uniform OAI message shape."""
    with src.open("r") as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            record = _normalize_oai_record(json.loads(line))
            f_out.write(json.dumps(record))
            f_out.write("\n")


def load_sft_dataset(config: SFTDataConfig) -> Dataset:
    """Load and interleave the raw HF dataset. This is the expensive I/O step."""
    logger = get_logger()
    data_files = _resolve_local_data_files(config.data_files)
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
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size,
            seq_len=config.seq_len,
            length=config.length,
            input_ids=config.input_ids,
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
            renderer=renderer,
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
