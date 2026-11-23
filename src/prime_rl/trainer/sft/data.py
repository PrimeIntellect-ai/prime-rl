import itertools
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
import torch
from datasets import Dataset, load_dataset
from renderers import AutoRendererConfig, RendererConfig, merge_chat_template_kwargs
from renderers.base import MultiModalData, PlaceholderRange, Renderer, build_training_sample, create_renderer
from torch import Tensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import HFDatasetSourceConfig, SFTDataConfig
from prime_rl.configs.supervised import ChatValSourceConfig, LossMaskConfig, SFTColumnsConfig
from prime_rl.trainer.data import (
    Sample,
    StatefulIterableDataset,
    due,
    get_dataset_state,
    mixture_changed,
    with_workers,
)
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger


class FakeSource:
    """Synthetic token sequences. The ``k``-th sample comes from a generator seeded by ``(seed, k)``, so any sample
    can be drawn on its own."""

    size = None

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        name: str = "fake",
        length: Literal["fixed", "variable"] = "fixed",
        input_ids: Literal["increasing", "random"] = "increasing",
        seed: int = 0,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.input_ids = input_ids
        self.seed = seed

    def sample(self, k: int) -> Sample:
        rng = np.random.default_rng([self.seed, k])
        seq_len = int(rng.integers(1, self.seq_len)) if self.length == "variable" else self.seq_len
        if self.input_ids == "random":
            input_ids = rng.integers(0, self.vocab_size, seq_len + 1, dtype=np.int64)
        else:
            input_ids = np.full(seq_len + 1, k, dtype=np.int64)
        return {
            "input_ids": input_ids[:-1],
            "target_ids": input_ids[1:],
            "position_ids": np.arange(seq_len),
            "loss_mask": np.ones(seq_len, dtype=bool),
            "seq_lens": [seq_len],
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        }


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
            if not getattr(renderer, "is_prefix_stable", False):
                get_logger().warning(
                    f"SFT renderer {type(renderer).__name__} does not guarantee prefix stability. "
                    "SFT renders each conversation once to produce one training sample; "
                    "it does not expand N assistant turns into N samples. Depending on the template, "
                    "reasoning from earlier turns may be omitted from the training sample."
                )
            self.renderers[config] = renderer
        return renderer


class HFSource:
    """Rows of a HF dataset with prompt/completion or messages columns, rendered into samples. The ``k``-th sample is
    row ``k % size`` of pass ``k // size``, each pass in its own seeded order."""

    def __init__(
        self,
        dataset: Dataset,
        renderers: Callable[[dict], Renderer],
        name: str = "train",
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        multimodal: bool = False,
        columns: SFTColumnsConfig = SFTColumnsConfig(),
    ):
        self.logger = get_logger()
        self.name = name
        self.dataset = dataset
        self.size = len(self.dataset)
        self.renderers = renderers
        self.columns = columns
        # Default names are optional: a dataset carries either messages or
        # prompt/completion, and tools only for tool use. A name set in the
        # config must exist.
        for field in ("messages", "prompt", "completion", "tools", "message_loss_mask"):
            column = getattr(columns, field)
            if column != field and column not in dataset.column_names:
                raise ValueError(f"data.columns.{field} is {column!r}, but the dataset has only {dataset.column_names}")
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.multimodal = multimodal
        if max_examples is not None:
            self.size = min(self.size, max_examples)
            self.dataset = self.dataset.take(self.size)
        self.order, self.order_pass = self.dataset, None

    def process(self, example: dict) -> dict | None:
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

        # Let the renderer identify sampled targets when assistant loss is enabled.
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
            message_loss_mask=example.get(self.columns.message_loss_mask),
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
            "input_ids": np.array(input_ids, dtype=np.int64),
            "target_ids": np.array(target_ids, dtype=np.int64),
            "loss_mask": np.array(loss_mask, dtype=bool),
            "position_ids": np.arange(len(input_ids)),
            "seq_lens": [len(input_ids)],
            "mm_kwargs": mm_kwargs,
            "mm_token_type_ids": np.array(mm_token_type_ids, dtype=np.int64) if mm_token_type_ids is not None else None,
        }

    def sample(self, k: int) -> Sample | None:
        """The ``k``-th sample, or None if the row has no trainable tokens within ``seq_len``."""
        pass_, index = divmod(k, self.size)
        if pass_ != self.order_pass:
            self.order = self.dataset.shuffle(seed=self.seed + pass_) if self.shuffle else self.dataset
            self.order_pass = pass_
        example = cast(dict, self.order[index])
        sample = self.process(example)
        if sample is not None:
            self.logger.debug(
                f"Yield example {example.get('__index', '')} from {self.name} with {len(sample['input_ids'])} tokens "
                f"({sample['loss_mask'].sum()} trainable tokens)"
            )
        return sample


class SFTDataset(StatefulIterableDataset):
    """Samples drawn from one or more sources by weight.

    The sources take turns by due time (weighted fair queuing, `due`): the next draw goes to the source with the
    smallest ``(draws + 1 - start) / weight``, ties to the first, so every stretch of draws holds the mixture; ``start``
    is every source's draws when the mixture last changed. Without weights, a source's weight is its number of rows,
    and every ``sum(rows)`` draws read each source exactly once (an epoch). Every data rank walks the same draws and
    keeps every ``data_world_size``-th one, so the draws are the same on every rank, for any number of ranks, and
    after a resume."""

    def __init__(
        self,
        sources: list[HFSource | FakeSource],
        weights: list[float] | None = None,
        dp_rank: int = 0,
        dp_size: int = 1,
    ):
        super().__init__(dp_rank, dp_size)
        self.sources = sources
        if weights is None:
            weights = [1.0] if len(sources) == 1 else [float(source.size) for source in sources]
        assert len(weights) == len(sources) and sum(weights) > 0
        self.weights = weights
        self.num_rows = sum(source.size for source in sources if source.size is not None)
        self.mixture = {
            source.name: {"weight": weight, "rows": source.size} for source, weight in zip(sources, weights)
        }
        self.draws, self.start = [0] * len(sources), [0] * len(sources)

    def state_dict(self) -> dict:
        names = [source.name for source in self.sources]
        self._setup_world_info()
        return {
            "step": self.step,
            "epoch": self.epoch,
            "draws": dict(zip(names, self.draws)),
            "start": dict(zip(names, self.start)),
            "sources": self.mixture,
            "layout": [self.data_rank, self.data_world_size],
        }

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if "draws" not in state_dict:
            # A single-source state from before sources had names: every step was a draw of that source, and its
            # counters were keyed by split.
            (source,) = self.sources
            state_dict = {**state_dict, "draws": {source.name: self.step}}
            for counts in (self.num_samples, self.num_tokens):
                total = sum(counts.values())
                counts.clear()
                counts[source.name] = total
        # Sources match by name (`mixture_changed`). A state from before they were recorded has the configured ones.
        saved = state_dict.get("sources", self.mixture)
        draws, start = state_dict["draws"], state_dict.get("start", {})
        if draws.keys() != saved.keys():
            raise ValueError(
                f"The checkpoint's SFT data state has the sources {list(draws)}, not the configured {list(saved)}: "
                "resume once with the sources it was saved with"
            )
        self._setup_world_info()
        # A changed mixture, or one the state does not record, counts the due times from here.
        if mixture_changed(saved, self.mixture, log=self.data_rank == 0) or "sources" not in state_dict:
            start = draws
        names = [source.name for source in self.sources]
        self.draws = [draws.get(name, 0) for name in names]
        self.start = [start.get(name, 0) for name in names]
        for counts in (self.num_samples, self.num_tokens):
            for name in counts.keys() - set(names):
                del counts[name]

    def saved_view(self, dataloader: StatefulDataLoader, directory: Path, rank: int, world_size: int) -> tuple:
        """What rank ``rank`` of ``world_size`` adds to the decision of a resume (`start_together`), from the saved
        files ``directory / rank_<rank>.pt``, ``rank_<rank + world_size>.pt``, ... (the ranks read every file between
        them, also on fewer ranks than the saved run):

        - the draw every saved worker had reached and whether it held a sample drawn but not packed, by its data rank
          and number of data ranks;
        - the dataset state of the saved worker that got furthest;
        - whether something changed for this rank: the mixture, or its workers loaded states that other data ranks
          saved (another number of GPUs, model-parallel layout or workers, or a state from before the layout was
          recorded)."""
        positions, furthest = {}, None
        for saved_rank in itertools.count(rank, world_size):
            path = directory / f"rank_{saved_rank}.pt"
            if not path.exists():
                break
            for key, worker in torch.load(path, weights_only=False)["_snapshot"]["_worker_snapshots"].items():
                state = worker["dataset_state"]
                layout = tuple(state["dataset"].get("layout", [(saved_rank, key), None]))
                positions[layout] = (state["dataset"]["step"], "pending_sample" in state)
                if furthest is None or state["dataset"]["step"] > furthest["step"]:
                    furthest = state["dataset"]
        loaded = list(get_dataset_state(dataloader).values())
        workers = dataloader.num_workers
        layouts = [[self.dp_rank * workers + worker, self.dp_size * workers] for worker in range(workers)]
        moved = [state.get("layout") for state in loaded] != layouts
        changed = moved or mixture_changed(loaded[0].get("sources", self.mixture), self.mixture, log=False)
        return positions, furthest, changed

    def start_together(self, dataloader: StatefulDataLoader, views: list[tuple]) -> tuple[int, int] | None:
        """Decide a resume the same way on every rank, from what every rank saw (`saved_view`). If anything changed,
        every worker reads on from the dataset state of the saved worker that got furthest: at a draw, the state is
        the same on every data rank. The others' draws up to there and the samples drawn but not packed are not
        trained on. Returns that draw and the number of those draws (for states from before the layout was recorded,
        as if every worker's last draw was its own), else None."""
        if not any(changed for *_, changed in views):
            return None
        positions = {layout: position for saved, *_ in views for layout, position in saved.items()}
        furthest = max((state for _, state, _ in views if state is not None), key=lambda state: state["step"])
        dataloader.load_state_dict(with_workers(dataloader.next_iter_state, dataloader.num_workers, furthest))
        skipped = 0
        for (data_rank, data_world), (step, pending) in positions.items():
            if data_world is None:
                data_rank, data_world = (step - 1) % len(positions), len(positions)
            # The draws of data rank d of W before draw n: those with k % W == d.
            skipped += (furthest["step"] - data_rank - 1) // data_world - (step - data_rank - 1) // data_world + pending
        return furthest["step"], skipped

    def next_source(self) -> int:
        return int(np.argmin(due(np.array(self.draws) + 1, np.array(self.start), np.array(self.weights))))

    def progress(self, state: dict) -> dict[str, float]:
        """Every source's draws and, for a source with rows, its passes over them (``epoch``) in ``state``."""
        metrics = {}
        for source in self.sources:
            draws = state["draws"].get(source.name, 0)
            metrics[f"progress/{source.name}/num_samples"] = draws
            if source.size:
                metrics[f"progress/{source.name}/epoch"] = draws / source.size
        return metrics

    def __iter__(self):
        self._setup_world_info()
        while True:
            index = self.next_source()
            source, k = self.sources[index], self.draws[index]
            self.draws[index] += 1
            self.step += 1
            if self.num_rows:
                rows_drawn = sum(draws for source, draws in zip(self.sources, self.draws) if source.size is not None)
                self.epoch = max(rows_drawn - 1, 0) // self.num_rows

            # Skip draws that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            sample = source.sample(k)
            if sample is None:
                continue
            self.num_samples[source.name] += 1
            self.num_tokens[source.name] += len(sample["input_ids"])
            yield sample


def load_sft_dataset(config: SFTDataConfig) -> dict[str, Dataset]:
    """Load the rows of every HF source, by source name. This is the expensive I/O step."""
    datasets = {}
    for source in config.source:
        get_logger().debug(f"Loading source {source.name}: {source.dataset} {source.subset=} {source.split=}")
        dataset = cast(
            Dataset, load_dataset(source.dataset, source.subset, split=source.split, revision=source.revision)
        )
        datasets[source.name] = dataset.add_column(
            "__index", list(range(len(dataset))), new_fingerprint=str(uuid.uuid4())
        )
    return datasets


def source_renderers(
    tokenizer: PreTrainedTokenizer,
    config: RendererConfig,
    source: HFDatasetSourceConfig | ChatValSourceConfig,
    processor: Any | None = None,
) -> RendererResolver:
    """The renderers of a source's rows: ``config`` with the source's chat-template kwargs (``source.renderer``) and
    its rows' ``columns.renderer`` values."""
    if source.renderer:
        if isinstance(config, AutoRendererConfig):
            raise ValueError(
                f"Source {source.name!r} sets renderer arguments, which require a typed renderer config "
                "(e.g. [renderer] name = 'qwen3.8'), not renderer.name = 'auto'"
            )
        config = merge_chat_template_kwargs(config, source.renderer)
    return RendererResolver(tokenizer, config, processor=processor, columns=source.columns.renderer)


def setup_sft_dataset(
    tokenizer: PreTrainedTokenizer,
    config: SFTDataConfig,
    *,
    dp_rank: int,
    dp_size: int,
    raw_datasets: dict[str, Dataset] | None = None,
    renderer_config: RendererConfig | None = None,
    processor: Any | None = None,
    multimodal: bool = False,
) -> SFTDataset:
    if config.fake is not None:
        fake = FakeSource(
            tokenizer.vocab_size,
            config.seq_len,
            length=config.fake.length,
            input_ids=config.fake.input_ids,
            seed=config.seed,
        )
        return SFTDataset([fake], dp_rank=dp_rank, dp_size=dp_size)
    if renderer_config is None:
        raise ValueError("SFT data requires a renderer config.")
    if raw_datasets is None:
        raw_datasets = load_sft_dataset(config)
    sources = []
    for source in config.source:
        sources.append(
            HFSource(
                raw_datasets[source.name],
                source_renderers(tokenizer, renderer_config, source, processor),
                name=source.name,
                shuffle=config.shuffle,
                seed=config.seed,
                seq_len=config.seq_len,
                loss_mask_config=config.loss_mask,
                multimodal=multimodal,
                columns=source.columns,
            )
        )
    weights = [source.weight for source in config.source]
    return SFTDataset(
        sources,
        weights=None if None in weights else weights,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
