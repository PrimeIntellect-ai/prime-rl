import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from renderers.base import MultiModalData, PlaceholderRange, Renderer, build_training_sample
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import DataConfig, LossMaskConfig, SFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger


@dataclass
class Sample:
    """One rendered training sample, carrying where in the dataset it came from."""

    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    mm_kwargs: dict[str, Tensor] | None = None
    mm_token_type_ids: list[int] | None = None
    source: str | None = None
    step: int = 0
    epoch: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)


@dataclass
class Batch:
    """One fixed-length row of packed samples, as CPU tensors ready for the forward pass."""

    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    seq_lens: Int[Tensor, "packed"]
    samples: list[Sample]
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    @property
    def step(self) -> int:
        """Furthest dataset position among the packed samples."""
        return max(sample.step for sample in self.samples)

    @property
    def epoch(self) -> int:
        return max(sample.epoch for sample in self.samples)

    @property
    def num_samples_by_source(self) -> dict[str | None, int]:
        num_samples: dict[str | None, int] = defaultdict(int)
        for sample in self.samples:
            num_samples[sample.source] += 1
        return dict(num_samples)

    @property
    def num_tokens_by_source(self) -> dict[str | None, int]:
        num_tokens: dict[str | None, int] = defaultdict(int)
        for sample in self.samples:
            num_tokens[sample.source] += sample.num_tokens
        return dict(num_tokens)

    @property
    def num_padding_tokens(self) -> int:
        return max(0, self.input_ids.numel() - sum(sample.num_tokens for sample in self.samples))

    def pin_memory(self) -> "Batch":
        # The dataloader's pin-memory thread pins custom batch types through this method
        return replace(
            self,
            input_ids=self.input_ids.pin_memory(),
            position_ids=self.position_ids.pin_memory(),
            target_ids=self.target_ids.pin_memory(),
            loss_mask=self.loss_mask.pin_memory(),
            seq_lens=self.seq_lens.pin_memory(),
            mm_kwargs={key: value.pin_memory() for key, value in self.mm_kwargs.items()}
            if self.mm_kwargs is not None
            else None,
            mm_token_type_ids=self.mm_token_type_ids.pin_memory() if self.mm_token_type_ids is not None else None,
        )


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self, non_dp_size: int = 1):
        self.step, self.epoch = 0, 0
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
        world = get_world()
        assert world.world_size % self.non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = world.rank // self.non_dp_size * num_workers + worker_id
        self.data_world_size = world.world_size // self.non_dp_size * num_workers


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
            yield Sample(
                input_ids=input_ids[:-1],
                target_ids=input_ids[1:],
                position_ids=list(range(seq_len)),
                loss_mask=[True] * seq_len,
                source="fake",
                step=self.step,
            )


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


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt/completion or raw messages format."""

    def __init__(
        self,
        dataset: Dataset,
        renderer: Renderer,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        multimodal: bool = False,
    ):
        super().__init__(non_dp_size)
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(self.dataset)
        self.renderer = renderer
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

    def _process(self, example: dict) -> Sample | None:
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

        # Defer to the renderer's sampled_mask by default: a role filter would
        # drop sampled stop markers attributed to the next message (e.g. GLM's
        # turn-closing <|user|> / <|observation|>).
        role_to_mask = None if self.loss_mask_config.assistant else should_mask

        # Non-assistant roles are opted into the loss via the renderer's
        # body-only path: the message content is trained, not the role
        # scaffolding (e.g. <|im_start|>assistant) the harness emits.
        content_sft_roles = {role for role in ("user", "system", "tool") if getattr(self.loss_mask_config, role)}
        sample = build_training_sample(
            self.renderer,
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

        if was_mm_truncated and not set(self.renderer.get_stop_token_ids()) & set(target_ids):
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
        assert set(self.renderer.get_stop_token_ids()) & set(target_ids), (
            "A renderer stop token must be present in target_ids"
        )

        mm_kwargs: dict[str, Tensor] | None = None
        if mm is not None and mm.mm_items:
            mm_kwargs = _flatten_mm_items(mm.mm_items)
            if any("video" in key for key in mm_kwargs):
                raise ValueError("Video SFT is not supported; sample contains video inputs")
        if mm_token_type_ids is not None:
            assert len(mm_token_type_ids) == len(input_ids)

        return Sample(
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
            position_ids=list(range(len(input_ids))),
            mm_kwargs=mm_kwargs,
            mm_token_type_ids=mm_token_type_ids,
            source=example.get("__subset") or example.get("__split"),
        )

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
            sample = self._process(cast(dict, example))

            # If processed example is None, skip it (e.g. if tokenized sample exceeds context window)
            if sample is None:
                continue

            # Yield the example
            sample.step, sample.epoch = self.step, self.epoch
            example = cast(dict, example)
            self.logger.debug(
                f"Yield example {example.get('__index', '')}"
                + (f" from {sample.source} " if sample.source else " ")
                + f"with {sample.num_tokens} tokens ({sum(sample.loss_mask)} trainable tokens)"
            )
            yield sample


class CatDataset(StatefulIterableDataset):
    """Group samples into packs that together fill one fixed-length row."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len
        self.pending_sample: Sample | None = None

    def state_dict(self) -> dict:
        state = {"dataset": self.dataset.state_dict()}
        if self.pending_sample is not None:
            state["pending_sample"] = self.pending_sample
        return state

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.pending_sample = state_dict.get("pending_sample")

    def __iter__(self):
        pack: list[Sample] = []
        pack_len = 0

        pending_sample = self.pending_sample
        self.pending_sample = None

        def samples():
            if pending_sample is not None:
                yield pending_sample
            yield from self.dataset

        for sample in samples():
            if pack and pack_len + sample.num_tokens > self.seq_len:
                # Stash the overflowing sample so a checkpoint taken now doesn't lose it
                self.pending_sample = sample
                yield pack
                self.pending_sample = None
                pack, pack_len = [], 0

            pack.append(sample)
            pack_len += sample.num_tokens

            if pack_len >= self.seq_len:
                yield pack
                pack, pack_len = [], 0

        if pack:
            yield pack


def cat_collate(packs: list[list[Sample]], seq_len: int) -> Batch:
    """Concatenate one pack of samples into a fixed-length row, truncating and padding to ``seq_len``.

    CPU tensors only: this runs in dataloader workers then the trainer moves batches to the GPU
    with async copies from pinned memory."""
    (pack,) = packs

    input_ids: list[int] = []
    position_ids: list[int] = []
    loss_mask: list[bool] = []
    target_ids: list[int] = []
    mm_kwargs: dict[str, Tensor] | None = None
    mm_token_type_ids: list[int] | None = None

    for sample in pack:
        existing_len = len(input_ids)
        input_ids.extend(sample.input_ids)
        position_ids.extend(sample.position_ids)
        loss_mask.extend(sample.loss_mask)
        target_ids.extend(sample.target_ids)

        if sample.mm_kwargs is None:
            if mm_token_type_ids is not None:
                mm_token_type_ids.extend([0] * sample.num_tokens)
        else:
            if mm_kwargs is not None and ((mm_token_type_ids is None) != (sample.mm_token_type_ids is None)):
                raise ValueError("Cannot pack multimodal samples with mixed mm_token_type_ids")

            if mm_kwargs is None:
                mm_kwargs = dict(sample.mm_kwargs)
            else:
                if mm_kwargs.keys() != sample.mm_kwargs.keys():
                    raise ValueError("Cannot pack multimodal samples with different mm_kwargs keys")
                for key, value in sample.mm_kwargs.items():
                    mm_kwargs[key] = torch.cat([mm_kwargs[key], value], dim=0)

            if mm_token_type_ids is None and sample.mm_token_type_ids is not None:
                mm_token_type_ids = [0] * existing_len
            if mm_token_type_ids is not None:
                mm_token_type_ids.extend(sample.mm_token_type_ids or [0] * sample.num_tokens)

    input_ids = input_ids[:seq_len]
    position_ids = position_ids[:seq_len]
    loss_mask = loss_mask[:seq_len]
    target_ids = target_ids[:seq_len]

    seq_lens: list[int] = []
    remaining = len(input_ids)
    for sample in pack:
        if remaining <= 0:
            break
        kept = min(sample.num_tokens, remaining)
        if kept > 0:
            seq_lens.append(kept)
        remaining -= kept

    pad_len = seq_len - len(input_ids)
    if pad_len > 0:
        input_ids.extend([0] * pad_len)
        position_ids.extend(range(pad_len))
        loss_mask.extend([False] * pad_len)
        target_ids.extend([0] * pad_len)
        seq_lens[-1] += pad_len
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids[:seq_len] + [0] * pad_len

    return Batch(
        input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        position_ids=torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
        loss_mask=torch.tensor(loss_mask, dtype=torch.bool).unsqueeze(0),
        target_ids=torch.tensor(target_ids, dtype=torch.long).unsqueeze(0),
        seq_lens=torch.tensor(seq_lens, dtype=torch.long),
        samples=pack,
        mm_kwargs=mm_kwargs,
        mm_token_type_ids=(
            torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0) if mm_token_type_ids is not None else None
        ),
    )


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
    renderer: Renderer | None = None,
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
        if renderer is None:
            raise ValueError("SFT data requires a renderer.")
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config)
        return SFTDataset(
            raw_dataset,
            renderer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            multimodal=multimodal,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(dataset: StatefulIterableDataset, config: DataConfig) -> StatefulDataLoader:
    seq_len = config.seq_len * config.micro_batch_size
    return StatefulDataLoader(
        CatDataset(dataset, seq_len),
        batch_size=1,
        collate_fn=partial(cat_collate, seq_len=seq_len),
        num_workers=config.num_workers,
        pin_memory=True,
    )
