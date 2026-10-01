import json
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import torch
from datasets import Dataset, interleave_datasets, load_dataset
from huggingface_hub import snapshot_download
from jaxtyping import Bool, Float, Int
from renderers import AutoRendererConfig, RendererConfig, merge_chat_template_kwargs
from renderers.base import MultiModalData, PlaceholderRange, Renderer, build_training_sample, create_renderer
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import (
    DataConfig,
    LossMaskConfig,
    PackedChoiceDataConfig,
    SFTChoiceEvalConfig,
    SFTColumnsConfig,
    SFTDataConfig,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_time


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
    seq_lens: Int[Tensor, "packed"]
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[Tensor, "batch seq"] | None


class PackedChoiceSample(TypedDict):
    input_ids: Int[np.ndarray, "seq"]
    position_ids: Int[np.ndarray, "seq"]
    loss_mask: Bool[np.ndarray, "seq"]
    target_ids: Int[np.ndarray, "seq"]
    seq_lens: Int[np.ndarray, "packed"]
    choice_ids: Int[np.ndarray, "seq choices"]
    choice_targets: Float[np.ndarray, "seq choices"]
    choice_weights: Float[np.ndarray, "seq"]
    choice_mix_weights: Float[np.ndarray, "seq components"]


class PackedChoiceBatch(Batch):
    choice_ids: Int[Tensor, "batch seq choices"]
    choice_targets: Float[Tensor, "batch seq choices"]
    choice_weights: Float[Tensor, "batch seq"]
    choice_mix_weights: Float[Tensor, "batch seq components"]


def data_rank_and_world_size(non_dp_size: int, worker_id: int = 0, num_workers: int = 1) -> tuple[int, int]:
    """This process's data rank and the data world size, counting each dataloader worker as a rank.

    Ranks in one group of ``non_dp_size`` consecutive ranks (e.g. context-parallel peers) share a data rank."""
    world = get_world()
    assert world.world_size % non_dp_size == 0, "world_size must be divisible by non_dp_size"
    return (
        world.rank // non_dp_size * num_workers + worker_id,
        world.world_size // non_dp_size * num_workers,
    )


class PackedChoiceEvalBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    seq_lens: Int[Tensor, "packed"]
    choice_ids: Int[Tensor, "batch seq choices"]
    row_ids: Int[Tensor, "batch seq"]


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self, non_dp_size: int = 1):
        self.step, self.epoch = 0, 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.fast_forward = False
        self.non_dp_size = non_dp_size
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
        self.data_rank, self.data_world_size = data_rank_and_world_size(self.non_dp_size, worker_id, num_workers)


class FakeDataset(StatefulIterableDataset):
    """A dataset of fake tokens"""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        length: Literal["fixed", "variable"] = "fixed",
        input_ids: Literal["increasing", "random"] = "random",
        seed: int = 0,
        non_dp_size: int = 1,
    ):
        super().__init__(non_dp_size)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.input_ids = input_ids
        self.seed = seed

    def _draw_sample(self, generator: torch.Generator) -> tuple[int, list[int] | None]:
        # Consume this samples "randomness" - fast forwarding must replay it to restore the generator state
        seq_len = (
            int(torch.randint(1, self.seq_len, (1,), generator=generator).item())
            if self.length == "variable"
            else self.seq_len
        )
        random_input_ids = (
            torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=generator).long().tolist()
            if self.input_ids == "random"
            else None
        )
        return seq_len, random_input_ids

    def __iter__(self):
        self._setup_world_info()
        # use a rank seeded PRNG instead of torch global default PRNG because with num workers > 0
        # the data loader reseeds the global PRNG per worker process
        generator = torch.Generator().manual_seed(self.seed + self.data_rank)
        if self.fast_forward:
            # step counts globally emmited samples but this rank is only emitted every data_world_size-TH
            already_emitted = len(range(self.data_rank, self.step, self.data_world_size))
            for _ in range(already_emitted):
                self._draw_sample(generator)
            self.fast_forward = False

        while True:
            self.step += 1

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            seq_len, random_input_ids = self._draw_sample(generator)
            input_ids = [self.step - 1] * (seq_len + 1) if random_input_ids is None else random_input_ids
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


def _flatten_mm_items(mm_items: dict[str, list[dict[str, Any]]]) -> dict[str, Tensor]:
    """Fold per-item renderer outputs into model-forward tensors."""
    out: dict[str, Tensor] = {}
    for items in mm_items.values():
        for item in items:
            for key, value in item.items():
                if not isinstance(value, (np.ndarray, Tensor)):
                    continue
                tensor = torch.as_tensor(value)
                out[key] = torch.cat([out[key], tensor], dim=0) if key in out else tensor
    return out


def _drop_null_fields(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Recursively strip ``None``-valued keys from dict structures.

    PyArrow's JSON loader unifies schemas across rows, so heterogeneous
    OAI content blocks (text vs image_url) end up with all union keys
    filled with ``None`` where absent. That confuses permissive
    content-type predicates inside renderers (e.g. ``"image_url" in item``
    returns ``True`` even when the value is null). Strip the noise before
    handing messages off to the renderer. Tool-call arguments are opaque
    JSON payloads, so preserve their null values.
    """
    if path[-3:] == ("tool_calls", "function", "arguments"):
        return value
    if isinstance(value, dict):
        return {k: _drop_null_fields(v, (*path, k)) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_null_fields(v, path) for v in value]
    return value


def _find_image_safe_cut(budget: int, mm: MultiModalData | None) -> int:
    """Return the largest cut at most ``budget`` outside placeholder runs."""
    if mm is None or not mm.mm_placeholders:
        return budget
    cut = budget
    for ranges in mm.mm_placeholders.values():
        for placeholder in ranges:
            if placeholder.offset < cut < placeholder.offset + placeholder.length:
                cut = placeholder.offset
    return cut


def _truncate_mm_data(mm: MultiModalData, cut: int) -> MultiModalData:
    """Drop multimodal items whose placeholder ranges extend past ``cut``."""
    new_placeholders: dict[str, list[PlaceholderRange]] = {}
    new_items: dict[str, list[dict[str, Any]]] = {}
    new_hashes: dict[str, list[str]] = {}
    for content_type, ranges in mm.mm_placeholders.items():
        keep = [index for index, placeholder in enumerate(ranges) if placeholder.offset + placeholder.length <= cut]
        if not keep:
            continue
        new_placeholders[content_type] = [ranges[index] for index in keep]
        new_items[content_type] = [mm.mm_items[content_type][index] for index in keep]
        if content_type in mm.mm_hashes:
            new_hashes[content_type] = [mm.mm_hashes[content_type][index] for index in keep]
    return MultiModalData(mm_hashes=new_hashes, mm_placeholders=new_placeholders, mm_items=new_items)


class RendererResolver:
    """Picks the renderer for a dataset row.

    ``columns`` maps renderer fields to dataset columns; a row's non-null
    values override the configured renderer's fields, validated as
    chat-template kwargs. Renderer configs are frozen, so renderers are cached
    per config and rows that resolve to the same config share one instance.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: RendererConfig,
        processor: Any | None = None,
        columns: dict[str, str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor
        self.columns = SFTColumnsConfig().renderer if columns is None else columns
        self.renderers: dict[RendererConfig, Renderer] = {}

    def resolve_config(self, example: dict) -> RendererConfig:
        kwargs = {field: example[column] for field, column in self.columns.items() if example.get(column) is not None}
        if not kwargs:
            return self.config
        if isinstance(self.config, AutoRendererConfig):
            raise ValueError(
                f"Per-sample renderer arguments {sorted(kwargs)} require a typed renderer config "
                "(e.g. [renderer] name = 'qwen3.8'), not renderer.name = 'auto'"
            )
        return merge_chat_template_kwargs(self.config, kwargs)

    def __call__(self, example: dict) -> Renderer:
        config = self.resolve_config(example)
        renderer = self.renderers.get(config)
        if renderer is None:
            renderer = create_renderer(self.tokenizer, config)
            if self.processor is not None and hasattr(renderer, "_processor"):
                renderer._processor = self.processor
            self.renderers[config] = renderer
        return renderer


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt/completion or raw messages format."""

    def __init__(
        self,
        dataset: Dataset,
        renderers: Callable[[dict], Renderer],
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        multimodal: bool = False,
        columns: SFTColumnsConfig = SFTColumnsConfig(),
    ):
        super().__init__(non_dp_size)
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(self.dataset)
        self.renderers = renderers
        self.columns = columns
        # Default names are optional: a dataset carries either messages or
        # prompt/completion, and tools only for tool use. A name set in the
        # config must exist.
        for field in ("messages", "prompt", "completion", "tools"):
            column = getattr(columns, field)
            if column != field and column not in dataset.column_names:
                raise ValueError(f"data.columns.{field} is {column!r}, but the dataset has only {dataset.column_names}")
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.max_examples = max_examples
        self.max_epochs = max_epochs
        self.multimodal = multimodal

        # If specified, select a subset of the dataset
        if self.max_examples is not None:
            self.num_examples = min(self.num_examples, self.max_examples)
            self.dataset = self.dataset.take(self.max_examples)

    def _process(self, example: dict) -> dict | None:
        def resolve_messages(example: dict) -> list[dict]:
            # `messages` takes precedence over explicit split fields and is interpreted
            # as a whole-chat training sample with an empty prompt. Null-check rather
            # than key-check: Arrow schema union adds `messages: null` to
            # prompt/completion rows whenever other rows have a `messages` column.
            columns = self.columns
            if example.get(columns.messages) is not None:
                messages = normalize_messages(example[columns.messages], default_role="assistant")
            elif example.get(columns.prompt) is not None and example.get(columns.completion) is not None:
                messages = normalize_messages(example[columns.prompt], default_role="user") + normalize_messages(
                    example[columns.completion], default_role="assistant"
                )
            else:
                raise ValueError(
                    f"All examples in the dataset must have either a {columns.messages!r} column "
                    f"or both {columns.prompt!r} and {columns.completion!r} columns for SFT"
                )

            # Strip nulls before deserializing so genuine nulls inside tool-call
            # argument strings survive.
            messages = [_drop_null_fields(m) for m in messages]
            return deserialize_tool_calls(messages)

        messages = resolve_messages(example)

        # Tool schemas in OpenAI function-calling format, as a list of dicts or a
        # JSON-encoded string of one.
        tools = example.get(self.columns.tools) or []
        if isinstance(tools, str):
            tools = json.loads(tools)

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

        # Defer to the renderer's sampled_mask by default: a role filter would
        # drop sampled stop markers attributed to the next message (e.g. GLM's
        # turn-closing <|user|> / <|observation|>).
        role_to_mask = None if self.loss_mask_config.assistant else should_mask

        # Non-assistant roles are opted into the loss via the renderer's
        # body-only path: the message content is trained, not the role
        # scaffolding (e.g. <|im_start|>assistant) the harness emits.
        content_sft_roles = {role for role in ("user", "system", "tool") if getattr(self.loss_mask_config, role)}
        renderer = self.renderers(example)
        sample = build_training_sample(
            renderer,
            messages,
            role_to_mask=role_to_mask,
            tools=tools,
            content_sft_roles=content_sft_roles or None,
            ensure_final_stop=True,
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

        # Causal shift: model predicts next token from current.
        target_ids = input_ids[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:-1]

        was_mm_truncated = False
        if mm is not None and len(input_ids) > self.seq_len:
            was_mm_truncated = True
            cut = _find_image_safe_cut(self.seq_len, mm)
            self.logger.debug(
                f"Truncating example {example.get('__index', '')} from "
                f"{len(input_ids)} → {cut} tokens (budget={self.seq_len})"
            )
            input_ids = input_ids[:cut]
            target_ids = target_ids[:cut]
            loss_mask = loss_mask[:cut]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[:cut]
            if mm.mm_items:
                mm = _truncate_mm_data(mm, cut)

        if was_mm_truncated and not set(renderer.get_stop_token_ids()) & set(target_ids):
            return None

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return None

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert set(renderer.get_stop_token_ids()) & set(target_ids), (
            "A renderer stop token must be present in target_ids"
        )

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
        self._setup_world_info()
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
    """Concatenate text and multimodal samples into one fixed-length row."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len
        self.pending_sample: Sample | None = None

    def state_dict(self) -> dict:
        state = {
            "dataset": self.dataset.state_dict(),
            "progress": {
                "num_samples": dict(self.dataset.num_samples),
                "num_tokens": dict(self.dataset.num_tokens),
            },
        }
        if self.pending_sample is not None:
            state["pending_sample"] = self.pending_sample
        return state

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        progress = state_dict.get("progress", {})
        self.dataset.num_samples.update(progress.get("num_samples", {}))
        self.dataset.num_tokens.update(progress.get("num_tokens", {}))
        self.pending_sample = state_dict.get("pending_sample")

    def __iter__(self):
        packed_samples = defaultdict(list)
        packed_samples["mm_kwargs"] = None
        packed_samples["mm_token_type_ids"] = None
        seq_len = 0

        pending_sample = self.pending_sample
        self.pending_sample = None

        def samples():
            if pending_sample is not None:
                yield pending_sample
            yield from self.dataset

        for sample in samples():
            sample_len = len(sample["input_ids"])
            would_overflow = seq_len + sample_len > self.seq_len
            if seq_len > 0 and would_overflow:
                self.pending_sample = sample
                yield self._finalize_pack(packed_samples, self.seq_len)
                self.pending_sample = None
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
                        packed_samples["mm_kwargs"][key] = torch.cat([packed_samples["mm_kwargs"][key], value], dim=0)

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
            result["seq_lens"][-1] += pad_len
        result["mm_kwargs"] = packed["mm_kwargs"]
        if packed["mm_token_type_ids"] is not None:
            result["mm_token_type_ids"] = packed["mm_token_type_ids"][:seq_len] + [0] * pad_len
        else:
            result["mm_token_type_ids"] = None
        return result


PACKED_CHOICE_SCHEMA_VERSION = "simile-packed-choice/v2"


@dataclass(frozen=True)
class PackedChoiceManifest:
    """The ``manifest.json`` of one split of a packed-choice export."""

    split: str
    seq_len: int
    num_bins: int
    max_choices: int
    mix_components: tuple[str, ...]
    """Names of the loss components, in the column order of each row's ``mix_weights``."""
    files: tuple[tuple[str, int], ...]
    """Arrow IPC file names and their bin counts, in read order."""

    @classmethod
    def load(cls, path: Path) -> "PackedChoiceManifest":
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["schema_version"] != PACKED_CHOICE_SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema {manifest['schema_version']!r}, expected {PACKED_CHOICE_SCHEMA_VERSION!r}"
            )
        if manifest["split"] not in ("train", "val"):
            raise ValueError(f"{path} has unknown split {manifest['split']!r}")
        files = tuple((entry["name"], entry["num_bins"]) for entry in manifest["files"])
        if sum(num_bins for _, num_bins in files) != manifest["num_bins"]:
            raise ValueError(f"{path}: per-file bin counts do not add up to num_bins = {manifest['num_bins']}")
        if not manifest["mix_components"]:
            raise ValueError(f"{path} names no loss components in mix_components")
        return cls(
            split=manifest["split"],
            seq_len=manifest["seq_len"],
            num_bins=manifest["num_bins"],
            max_choices=manifest["max_choices"],
            mix_components=tuple(manifest["mix_components"]),
            files=files,
        )


def read_packed_bins(path: Path, files: tuple[tuple[str, int], ...]) -> pa.Table:
    """The bins of an export's Arrow IPC files, memory-mapped and concatenated in read order."""
    tables = []
    for name, num_bins in files:
        table = pa.ipc.open_file(pa.memory_map(str(path / name))).read_all()
        if table.num_rows != num_bins:
            raise ValueError(f"{path / name} has {table.num_rows} bins, manifest says {num_bins}")
        tables.append(table)
    return pa.concat_tables(tables)


def pad_packed_bin(
    input_ids: np.ndarray, seq_lens: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``input_ids`` right-padded to ``seq_len``, their position ids (restarting per segment and for the
    padding) and ``seq_lens`` with the padding counted into the last segment."""
    pad_len = seq_len - len(input_ids)
    position_ids = np.concatenate([np.arange(length) for length in (*seq_lens, pad_len)])
    padded_seq_lens = seq_lens.copy()
    padded_seq_lens[-1] += pad_len
    return np.pad(input_ids, (0, pad_len)), position_ids, padded_seq_lens


def scatter_choice_rows(
    positions: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    *,
    seq_len: int,
    max_choices: int,
    fill: float,
    dtype: npt.DTypeLike,
) -> np.ndarray:
    """Dense ``[seq_len, max_choices]`` array holding row ``r``'s ``counts[r]`` consecutive ``values`` at
    ``positions[r]``, right-padded with ``fill``; every other position is ``fill``."""
    rows, slots = np.nonzero(np.arange(max_choices) < counts[:, None])
    dense = np.full((seq_len, max_choices), fill, dtype=dtype)
    dense[positions[rows], slots] = values
    return dense


class PackedChoiceDataset(StatefulIterableDataset):
    """Bins of a packed-choice export, one per micro batch, padded to ``seq_len``.

    Bin ``i`` in stored order goes to data rank ``i % data_world_size``. Without ``max_epochs`` the
    stored order repeats forever.
    """

    def __init__(self, path: Path, seq_len: int, non_dp_size: int = 1, max_epochs: int | None = None):
        super().__init__(non_dp_size)
        self.path = path
        self.manifest = PackedChoiceManifest.load(path)
        if self.manifest.seq_len > seq_len:
            raise ValueError(f"{path} was packed to {self.manifest.seq_len} tokens, above data.seq_len = {seq_len}")
        self.seq_len = seq_len
        self.max_epochs = max_epochs

    def state_dict(self) -> dict:
        return {
            "dataset": super().state_dict(),
            "progress": {"num_samples": dict(self.num_samples), "num_tokens": dict(self.num_tokens)},
        }

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict["dataset"])
        self.num_samples.update(state_dict["progress"]["num_samples"])
        self.num_tokens.update(state_dict["progress"]["num_tokens"])

    def __iter__(self):
        self._setup_world_info()
        # Mapped per iterator: memory maps cannot be pickled into dataloader workers
        bins = read_packed_bins(self.path, self.manifest.files)
        while True:
            self.step += 1
            epoch = (self.step - 1) // self.manifest.num_bins
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break
            self.epoch = epoch
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            index = (self.step - 1) % self.manifest.num_bins
            row = bins.slice(index, 1)
            self.num_samples[self.manifest.split] += len(row.column("seq_lens")[0])
            self.num_tokens[self.manifest.split] += len(row.column("input_ids")[0])
            yield self._build_sample(row, index)

    def _build_sample(self, row: pa.Table, index: int) -> PackedChoiceSample:
        def column(name: str) -> pa.Array:
            return row.column(name).combine_chunks().flatten()

        input_ids = column("input_ids").to_numpy().astype(np.int64)
        seq_lens = column("seq_lens").to_numpy().astype(np.int64)
        positions = column("positions").to_numpy().astype(np.int64)
        choice_lists = column("choice_ids")
        target_lists = column("target_probs")
        weights = column("weights").to_numpy()
        mix_lists = column("mix_weights")
        counts = choice_lists.value_lengths().to_numpy()
        num_tokens, num_rows, num_choices = len(input_ids), len(positions), self.manifest.max_choices
        num_components = len(self.manifest.mix_components)

        def check(ok: bool, message: str) -> None:
            if not ok:
                raise ValueError(f"{self.path}: bin {index} {message}")

        check(0 < num_tokens <= self.manifest.seq_len, f"has {num_tokens} tokens, not in 1..{self.manifest.seq_len}")
        check(int(seq_lens.sum()) == num_tokens, "has seq_lens that do not sum to its token count")
        check(bool(np.all(np.diff(positions) > 0)), "has positions that are not strictly increasing")
        check(num_rows == 0 or (positions[0] >= 0 and positions[-1] + 1 < num_tokens), "has out-of-range positions")
        check(len(weights) == len(counts) == len(mix_lists) == num_rows, "has per-row columns of different lengths")
        check(bool(np.array_equal(counts, target_lists.value_lengths().to_numpy())), "has choices without targets")
        check(bool(np.all((counts >= 1) & (counts <= num_choices))), f"has a row outside 1..{num_choices} choices")
        check(
            bool(np.all(mix_lists.value_lengths().to_numpy() == num_components)),
            f"has a mix_weights row without {num_components} components",
        )

        choice_ids = scatter_choice_rows(
            positions,
            counts,
            choice_lists.flatten().to_numpy(),
            seq_len=self.seq_len,
            max_choices=num_choices,
            fill=-1,
            dtype=np.int64,
        )
        choice_targets = scatter_choice_rows(
            positions,
            counts,
            target_lists.flatten().to_numpy(),
            seq_len=self.seq_len,
            max_choices=num_choices,
            fill=0.0,
            dtype=np.float32,
        )
        choice_weights = np.zeros(self.seq_len, dtype=np.float32)
        choice_weights[positions] = weights
        choice_mix_weights = np.zeros((self.seq_len, num_components), dtype=np.float32)
        choice_mix_weights[positions] = mix_lists.flatten().to_numpy().reshape(num_rows, num_components)
        loss_mask = np.zeros(self.seq_len, dtype=bool)
        loss_mask[positions] = True

        padded_input_ids, position_ids, padded_seq_lens = pad_packed_bin(input_ids, seq_lens, self.seq_len)
        return {
            "input_ids": padded_input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "target_ids": np.pad(input_ids[1:], (0, self.seq_len - num_tokens + 1)),
            "seq_lens": padded_seq_lens,
            "choice_ids": choice_ids,
            "choice_targets": choice_targets,
            "choice_weights": choice_weights,
            "choice_mix_weights": choice_mix_weights,
        }


def packed_choice_collate(samples: list[PackedChoiceSample]) -> PackedChoiceBatch:
    (sample,) = samples
    return {
        "input_ids": torch.from_numpy(sample["input_ids"]).unsqueeze(0),
        "position_ids": torch.from_numpy(sample["position_ids"]).unsqueeze(0),
        "loss_mask": torch.from_numpy(sample["loss_mask"]).unsqueeze(0),
        "target_ids": torch.from_numpy(sample["target_ids"]).unsqueeze(0),
        "seq_lens": torch.from_numpy(sample["seq_lens"]),
        "choice_ids": torch.from_numpy(sample["choice_ids"]).unsqueeze(0),
        "choice_targets": torch.from_numpy(sample["choice_targets"]).unsqueeze(0),
        "choice_weights": torch.from_numpy(sample["choice_weights"]).unsqueeze(0),
        "choice_mix_weights": torch.from_numpy(sample["choice_mix_weights"]).unsqueeze(0),
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }


def packed_choice_max_steps(config: PackedChoiceDataConfig, max_steps: int | None) -> int:
    """``max_steps`` for packed-choice data: one pass over the full batches unless set.

    Trailing bins short of a full batch are dropped; a set ``max_steps`` past one pass repeats bins.
    """
    num_bins = PackedChoiceManifest.load(config.name).num_bins
    logger = get_logger()
    if max_steps is not None:
        if max_steps * config.batch_size > num_bins:
            logger.warning(
                f"max_steps = {max_steps} trains {max_steps * config.batch_size} bins, more than the "
                f"{num_bins} of {config.name}: bins repeat"
            )
        return max_steps
    one_pass, num_dropped = divmod(num_bins, config.batch_size)
    if one_pass == 0:
        raise ValueError(f"{config.name} has {num_bins} bins, fewer than one batch of {config.batch_size}")
    logger.info(f"Training one pass over {num_bins} bins of {config.name}: max_steps = {one_pass}")
    if num_dropped > 0:
        logger.warning(f"Dropping the last {num_dropped} bins of {config.name} (short of a full batch)")
    return one_pass


def packed_choice_mix_components(
    config: PackedChoiceDataConfig, val_config: SFTDataConfig | PackedChoiceDataConfig | None
) -> tuple[str, ...]:
    """The loss component names of the train export's ``mix_weights`` columns, which a packed-choice
    validation export must share (one loss function scores both)."""
    mix_components = PackedChoiceManifest.load(config.name).mix_components
    if isinstance(val_config, PackedChoiceDataConfig):
        val_components = PackedChoiceManifest.load(val_config.name).mix_components
        if val_components != mix_components:
            raise ValueError(
                f"{val_config.name} has mix_components {list(val_components)}, but {config.name} has "
                f"{list(mix_components)}"
            )
    return mix_components


PACKED_CHOICE_EVAL_SCHEMA_VERSION = "simile-packed-choice-eval/v1"


@dataclass(frozen=True)
class PackedChoiceEvalManifest:
    """The ``manifest.json`` of one choice-eval set export."""

    name: str
    seq_len: int
    num_bins: int
    num_rows: int
    max_choices: int
    files: tuple[tuple[str, int], ...]
    """Arrow IPC file names and their bin counts, in read order."""

    @classmethod
    def load(cls, path: Path) -> "PackedChoiceEvalManifest":
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["schema_version"] != PACKED_CHOICE_EVAL_SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema {manifest['schema_version']!r}, expected {PACKED_CHOICE_EVAL_SCHEMA_VERSION!r}"
            )
        if manifest["num_bins"] < 1:
            raise ValueError(f"{path} has no bins")
        files = tuple((entry["name"], entry["num_bins"]) for entry in manifest["files"])
        if sum(num_bins for _, num_bins in files) != manifest["num_bins"]:
            raise ValueError(f"{path}: per-file bin counts do not add up to num_bins = {manifest['num_bins']}")
        return cls(
            name=manifest["name"],
            seq_len=manifest["seq_len"],
            num_bins=manifest["num_bins"],
            num_rows=manifest["num_rows"],
            max_choices=manifest["max_choices"],
            files=files,
        )


@dataclass(frozen=True)
class _EvalBin:
    input_ids: np.ndarray
    seq_lens: np.ndarray
    positions: np.ndarray
    choice_ids: np.ndarray
    """Every row's choice ids, concatenated in row order."""
    choice_counts: np.ndarray
    row_ids: np.ndarray


class PackedChoiceEvalSet:
    """Every bin of a choice-eval export, split over data ranks so that each rank runs ``num_forwards`` forwards.

    Slot ``i`` goes to data rank ``i % data_world_size``. Slots past the last bin replay bin ``i % num_bins`` so
    every rank sends real tokens through the collectives (an all-padding bin could route every token to the same
    experts); callers discard the rows of replayed bins. Context-parallel peers get the same bins.
    ``choice_counts`` holds each row's number of choices, by row id.
    """

    def __init__(self, path: Path, seq_len: int, non_dp_size: int = 1):
        self.path = path
        self.manifest = PackedChoiceEvalManifest.load(path)
        if self.manifest.seq_len > seq_len:
            raise ValueError(f"{path} was packed to {self.manifest.seq_len} tokens, above data.seq_len = {seq_len}")
        self.seq_len = seq_len
        self.data_rank, self.data_world_size = data_rank_and_world_size(non_dp_size)
        self.num_forwards = -(-self.manifest.num_bins // self.data_world_size)
        self._bins = read_packed_bins(path, self.manifest.files)
        self.choice_counts = self._validate()

    def batches(self) -> Iterator[tuple[PackedChoiceEvalBatch, bool]]:
        """This data rank's padded bins, each with whether it replays a bin that another slot scores."""
        for forward in range(self.num_forwards):
            slot = self.data_rank + forward * self.data_world_size
            yield self._build_batch(slot % self.manifest.num_bins), slot >= self.manifest.num_bins

    def _read_bin(self, index: int) -> _EvalBin:
        row = self._bins.slice(index, 1)

        def column(name: str) -> pa.Array:
            return row.column(name).combine_chunks().flatten()

        choice_lists = column("choice_ids")
        return _EvalBin(
            input_ids=column("input_ids").to_numpy().astype(np.int64),
            seq_lens=column("seq_lens").to_numpy().astype(np.int64),
            positions=column("positions").to_numpy().astype(np.int64),
            choice_ids=choice_lists.flatten().to_numpy().astype(np.int64),
            choice_counts=choice_lists.value_lengths().to_numpy().astype(np.int64),
            row_ids=column("row_ids").to_numpy().astype(np.int64),
        )

    def _validate(self) -> np.ndarray:
        """Check every bin, and that the bins hold each row id in ``0..num_rows-1`` exactly once; returns
        each row's number of choices by row id."""
        row_ids, choice_counts = [], []
        for index in range(self.manifest.num_bins):
            eval_bin = self._read_bin(index)
            self._check_bin(index, eval_bin)
            row_ids.append(eval_bin.row_ids)
            choice_counts.append(eval_bin.choice_counts)
        all_row_ids = np.concatenate(row_ids)
        order = np.argsort(all_row_ids, kind="stable")
        if not np.array_equal(all_row_ids[order], np.arange(self.manifest.num_rows)):
            raise ValueError(f"{self.path}: row_ids do not cover 0..{self.manifest.num_rows - 1} exactly once")
        return np.concatenate(choice_counts)[order]

    def _check_bin(self, index: int, eval_bin: _EvalBin) -> None:
        manifest = self.manifest
        num_tokens, positions, counts = len(eval_bin.input_ids), eval_bin.positions, eval_bin.choice_counts

        def check(ok: bool, message: str) -> None:
            if not ok:
                raise ValueError(f"{self.path}: bin {index} {message}")

        check(0 < num_tokens <= manifest.seq_len, f"has {num_tokens} tokens, not in 1..{manifest.seq_len}")
        check(int(eval_bin.seq_lens.sum()) == num_tokens, "has seq_lens that do not sum to its token count")
        check(len(positions) > 0, "has no rows")
        check(len(counts) == len(eval_bin.row_ids) == len(positions), "has per-row columns of different lengths")
        check(bool(np.all(np.diff(positions) > 0)), "has positions that are not strictly increasing")
        # A prompt ends at the position its prediction is read from, so the last one may end the bin
        check(positions[0] >= 0 and positions[-1] < num_tokens, "has out-of-range positions")
        check(bool(np.all((counts >= 1) & (counts <= manifest.max_choices))), "has a row outside 1..max_choices")
        check(bool(np.all(eval_bin.choice_ids >= 0)), "has a negative choice id")

    def _build_batch(self, index: int) -> PackedChoiceEvalBatch:
        eval_bin = self._read_bin(index)
        input_ids, position_ids, seq_lens = pad_packed_bin(eval_bin.input_ids, eval_bin.seq_lens, self.seq_len)
        choice_ids = scatter_choice_rows(
            eval_bin.positions,
            eval_bin.choice_counts,
            eval_bin.choice_ids,
            seq_len=self.seq_len,
            max_choices=self.manifest.max_choices,
            fill=-1,
            dtype=np.int64,
        )
        row_ids = np.full(self.seq_len, -1, dtype=np.int64)
        row_ids[eval_bin.positions] = eval_bin.row_ids
        return {
            "input_ids": torch.from_numpy(input_ids).unsqueeze(0),
            "position_ids": torch.from_numpy(position_ids).unsqueeze(0),
            "seq_lens": torch.from_numpy(seq_lens),
            "choice_ids": torch.from_numpy(choice_ids).unsqueeze(0),
            "row_ids": torch.from_numpy(row_ids).unsqueeze(0),
        }


def setup_choice_eval_sets(
    config: SFTChoiceEvalConfig, seq_len: int, non_dp_size: int
) -> dict[str, PackedChoiceEvalSet]:
    """The configured eval sets by name, each loaded and checked against its export's name."""
    eval_sets = {}
    for name, set_config in config.sets.items():
        eval_set = PackedChoiceEvalSet(set_config.path, seq_len=seq_len, non_dp_size=non_dp_size)
        if eval_set.manifest.name != name:
            raise ValueError(
                f"choice_eval.sets.{name} points at {set_config.path}, the export of {eval_set.manifest.name!r}"
            )
        eval_sets[name] = eval_set
    return eval_sets


def cat_collate(samples: list[Sample]) -> Batch:
    # CPU tensors only: this runs in dataloader workers then the trainer moves batches to the GPU with async copies from pinned memory
    (sample,) = samples
    mm_kwargs = sample.get("mm_kwargs")
    mm_token_type_ids = sample.get("mm_token_type_ids")
    return {
        "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0),
        "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long).unsqueeze(0),
        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool).unsqueeze(0),
        "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long).unsqueeze(0),
        "seq_lens": torch.tensor(sample["seq_lens"], dtype=torch.long),
        "mm_kwargs": dict(mm_kwargs) if mm_kwargs is not None else None,
        "mm_token_type_ids": (
            torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0) if mm_token_type_ids is not None else None
        ),
    }


def pre_download_data(data: DataConfig, env_vars: dict[str, str]) -> None:
    if not isinstance(data, SFTDataConfig):
        return
    if Path(data.name).exists():
        get_logger().info(f"Data {data.name} found at local path, skipping download")
        return

    dataset_name = data.name
    t0 = time.perf_counter()
    get_logger().info(f"Pre-downloading data {dataset_name} at revision {data.revision or 'main'}")
    snapshot = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        revision=data.revision,
        cache_dir=env_vars.get("HF_HUB_CACHE"),
    )
    data.name = snapshot
    get_logger().debug(
        f"Finished pre-downloading data {dataset_name} to {snapshot} in {format_time(time.perf_counter() - t0)}"
    )


def setup_and_interleave_datasets(
    dataset_name: str,
    subsets_and_splits: list[tuple[str | None, str]],
    probabilities: list[float] | None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"],
    seed: int = 0,
    revision: str | None = None,
) -> Dataset:
    logger = get_logger()
    datasets = []
    for subset, split in subsets_and_splits:
        logger.debug(f"Loading dataset {dataset_name} with {subset=} and {split=}")
        dataset = cast(Dataset, load_dataset(dataset_name, subset, split=split, revision=revision))
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
            revision=config.revision,
        )
    elif config.subsets is not None and config.splits is None:
        logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(subset, "train") for subset in config.subsets],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
            revision=config.revision,
        )
    elif config.subsets is None and config.splits is not None:
        logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, split) for split in config.splits],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
            revision=config.revision,
        )
    else:
        assert config.subsets is not None and config.splits is not None
        logger.debug(f"Loading datasets for subsets {config.subsets} with splits {config.splits}")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=list(zip(config.subsets, config.splits)),
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
            revision=config.revision,
        )


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    non_dp_size: int = 1,
    *,
    max_epochs: int | None = None,
    raw_dataset: Dataset | None = None,
    renderer_config: RendererConfig | None = None,
    processor: Any | None = None,
    multimodal: bool = False,
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size,
            seq_len=config.seq_len,
            length=config.length,
            input_ids=config.input_ids,
            seed=config.seed,
            non_dp_size=non_dp_size,
        )
    elif config.type == "sft":
        if renderer_config is None:
            raise ValueError("SFT data requires a renderer config.")
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config)
        renderers = RendererResolver(tokenizer, renderer_config, processor=processor, columns=config.columns.renderer)
        return SFTDataset(
            raw_dataset,
            renderers,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            multimodal=multimodal,
            columns=config.columns,
        )
    elif config.type == "packed_choice":
        return PackedChoiceDataset(config.name, seq_len=config.seq_len, non_dp_size=non_dp_size, max_epochs=max_epochs)
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(dataset: StatefulIterableDataset, config: DataConfig) -> StatefulDataLoader:
    if isinstance(dataset, PackedChoiceDataset):
        return StatefulDataLoader(
            dataset,
            batch_size=1,
            collate_fn=packed_choice_collate,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size)
    return StatefulDataLoader(
        packing_dataset,
        batch_size=1,
        collate_fn=cat_collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def get_dataset_state(dataloader: StatefulDataLoader) -> dict:
    """Dataset position per worker for the startup log, parsed from ``StatefulDataLoader.state_dict()``.

    The loader is the only source that is correct in every case: after a resume's
    ``load_state_dict`` the restored position exists solely in the loader's stashed
    state (it reaches the dataset copies inside workers when the iterator forks them;
    the main-process dataset object stays at position zero). The keys are torchdata's
    private worker-snapshot layout."""
    snapshots = dataloader.state_dict()["_snapshot"]["_worker_snapshots"]
    return {wid: snap["dataset_state"]["dataset"] for wid, snap in sorted(snapshots.items())}


def get_dataset_progress(dataloader: StatefulDataLoader) -> dict:
    """Dataset position and aggregate counters from dataloader workers."""
    snapshot = dataloader.state_dict()["_snapshot"]
    worker_snapshots = snapshot["_worker_snapshots"]
    positions = [worker_snapshot["dataset_state"]["dataset"] for worker_snapshot in worker_snapshots.values()]
    furthest = max(positions, key=lambda position: position["step"])
    num_samples = defaultdict(int)
    num_tokens = defaultdict(int)
    for worker_snapshot in worker_snapshots.values():
        progress = worker_snapshot["dataset_state"].get("progress", {})
        for name, count in progress.get("num_samples", {}).items():
            num_samples[name] += count
        for name, count in progress.get("num_tokens", {}).items():
            num_tokens[name] += count
    return {
        **furthest,
        "num_samples": dict(num_samples),
        "num_tokens": dict(num_tokens),
    }
