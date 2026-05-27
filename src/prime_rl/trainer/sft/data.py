import json
import uuid
from collections import defaultdict
from typing import Any, Callable, Literal, TypedDict, cast

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from renderers.base import (
    Message,
    Renderer,
    ToolSpec,
    build_training_sample,
    is_multimodal,
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


class Sample(TypedDict, total=False):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    # Per-token modality marker (0=text, 1=image, 2=video). Populated by
    # multimodal renderers via ``mm_token_type_id_map``; absent on the
    # text-only path.
    mm_token_type_ids: list[int]
    # Per-image processor outputs keyed by the model's forward kwarg
    # names (e.g. ``pixel_values``, ``image_grid_thw``). One tensor per
    # image, not yet batched — packing decides which to keep, the
    # collate step concatenates along dim=0.
    mm_items: dict[str, list[Tensor]]


class Batch(TypedDict, total=False):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    # Multimodal kwargs forwarded verbatim to ``model(**kwargs)``: tensors
    # are concatenated per-key along dim=0 (no batch dim), matching what
    # ``prime_rl.trainer.model.forward`` expects.
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[Tensor, "batch seq"] | None


def build_mm_training_sample(
    renderer: Renderer,
    messages: list[Message],
    *,
    role_to_mask: Callable[[Message], bool],
    tools: list[ToolSpec] | None = None,
) -> tuple[list[int], list[bool], list[int], dict[str, list[Tensor]]]:
    """Multimodal variant of ``renderers.base.build_training_sample``.

    Returns ``(token_ids, loss_mask, mm_token_type_ids, mm_items)``:

    - ``loss_mask`` follows the same rules as upstream — AND of the
      renderer's ``sampled_mask`` (when populated) and ``role_to_mask``.
    - ``mm_token_type_ids[k]`` is ``1`` for image-placeholder tokens,
      ``2`` for video, ``0`` otherwise. Derived from
      ``renderer.mm_token_type_id_map`` (single source of truth). On a
      text-only renderer the map is missing / empty and every entry is ``0``.
    - ``mm_items[key]`` is the per-image tensor list emitted by the HF
      processor inside the renderer (e.g. ``"pixel_values"``,
      ``"image_grid_thw"`` for Qwen3-VL). Packing decides which to keep;
      the collate step concatenates per key along dim=0.
    """
    rendered = renderer.render(messages, tools=tools)
    has_sampled_info = len(rendered.sampled_mask) == len(rendered.token_ids)

    loss_mask: list[bool] = []
    for k, msg_idx in enumerate(rendered.message_indices):
        if msg_idx < 0:
            loss_mask.append(False)
            continue
        if has_sampled_info and not rendered.sampled_mask[k]:
            loss_mask.append(False)
            continue
        loss_mask.append(role_to_mask(messages[msg_idx]))

    mtt_map = getattr(renderer, "mm_token_type_id_map", {}) or {}
    mm_token_type_ids = [mtt_map.get(tid, 0) for tid in rendered.token_ids]

    mm_items: dict[str, list[Tensor]] = {}
    mmd = rendered.multi_modal_data
    if mmd is not None and not mmd.is_empty():
        for items in mmd.mm_items.values():
            for item in items:
                for key, payload in item.items():
                    mm_items.setdefault(key, []).append(torch.as_tensor(payload))

    return rendered.token_ids, loss_mask, mm_token_type_ids, mm_items


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
        self.is_multimodal = renderer is not None and is_multimodal(renderer)
        self._warned_chat_template_kwargs = False

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

        # Parse available tools, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#function-tool-example
        # Accepts either `tools` or `tool_defs` (the verifiers rollout format),
        # as either a JSON-encoded string of a list or a list of dicts. Tools
        # arriving in the verifiers shape are converted to OAI form so any
        # downstream chat template can consume them.
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
                    return True if self.loss_mask_config.user else False
                case "assistant":
                    return True if self.loss_mask_config.assistant else False
                case "system":
                    return True if self.loss_mask_config.system else False
                case "tool":
                    return True if self.loss_mask_config.tool else False
                case _:
                    raise ValueError(f"Invalid message role: {message['role']}")

        mm_token_type_ids: list[int] | None = None
        mm_items: dict[str, list[Tensor]] | None = None

        if self.renderer is not None:
            if example.get("chat_template_kwargs") and not self._warned_chat_template_kwargs:
                self.logger.warning(
                    "Example carries chat_template_kwargs but use_renderer=True; "
                    "renderers don't forward chat_template_kwargs (model-specific "
                    "renderers bake their template behavior in). These kwargs will "
                    "be ignored. Further warnings suppressed for this dataset."
                )
                self._warned_chat_template_kwargs = True

            if self.is_multimodal:
                input_ids, loss_mask, mm_token_type_ids, mm_items = build_mm_training_sample(
                    self.renderer,
                    messages,
                    role_to_mask=should_mask,
                    tools=tools,
                )
            else:
                input_ids, loss_mask = build_training_sample(
                    self.renderer,
                    messages,
                    role_to_mask=should_mask,
                    tools=tools,
                )
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

        # If EOS token is not found, manually append it
        if not self.tokenizer.eos_token_id in input_ids:
            self.logger.warning(
                f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. Is something wrong with the chat template? Manually appending EOS token..."
            )
            input_ids.append(cast(int, self.tokenizer.eos_token_id))
            loss_mask.append(True)
            if mm_token_type_ids is not None:
                mm_token_type_ids.append(0)

        # Prepare inputs
        target_ids = input_ids.copy()[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]
        if mm_token_type_ids is not None:
            # Align with the shifted ``input_ids``: position k feeds token
            # input_ids[k] into the model, so mm_token_type_ids must
            # describe positions 0..N-2 of the original stream.
            mm_token_type_ids = mm_token_type_ids[:-1]

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return

        # Multimodal samples must never be sliced — a mid-image-pad-run
        # truncation would leave image-feature tensors that no longer
        # correspond to placeholder tokens. Drop on oversize and warn.
        # We gate on the sample actually carrying images (rather than the
        # renderer being multimodal-capable) so text-only data passing
        # through a multimodal renderer (e.g. Qwen3.5 text SFT) keeps the
        # existing truncate-in-packer behaviour.
        sample_has_images = bool(mm_items)
        if sample_has_images and len(input_ids) > self.seq_len:
            self.logger.warning(
                f"Skipping multimodal example {example.get('__index', '')} "
                f"because it has {len(input_ids)} tokens > seq_len ({self.seq_len}). "
                "Multimodal samples cannot be safely truncated; increase seq_len or drop the example."
            )
            return

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

        sample: dict[str, Any] = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
        }
        # When the renderer is multimodal-capable we always attach
        # ``mm_token_type_ids`` (a list — ``CatDataset`` accepts, and
        # ``cat_collate`` silently ignores when unused) so packed batches
        # can mix image-bearing and text-only samples without breaking
        # ``MMCatDataset``'s buffer alignment. ``mm_items`` is a dict and
        # only attached when actually present, so the ``CatDataset`` path
        # (which asserts every sample value is a ``list``) keeps working
        # for text-only data flowing through a multimodal renderer.
        if mm_token_type_ids is not None:
            assert len(mm_token_type_ids) == len(input_ids), (
                f"mm_token_type_ids length mismatch: {len(mm_token_type_ids)=} vs {len(input_ids)=}"
            )
            sample["mm_token_type_ids"] = mm_token_type_ids
        if sample_has_images:
            sample["mm_items"] = mm_items
        return sample

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


class MMCatDataset(StatefulIterableDataset):
    """Boundary-aware concatenation packer for (multi)modal samples.

    Unlike :class:`CatDataset`, this packer:

    1. Never splits a source sample. When the next sample wouldn't fit in
       ``seq_len``, the current buffer is padded to ``seq_len`` and
       yielded. The next sample starts a fresh buffer. This keeps
       image-pad runs intact and avoids dropping image tokens whose
       processed tensors would still describe the full image set.
    2. Carries ``mm_items`` (per-image tensors) alongside the text-side
       lists. Tensors are forwarded verbatim to the collate step which
       concatenates per-key along dim=0 — the shape the trainer's
       ``forward`` expects.
    3. Resets ``position_ids`` at every sample boundary by construction
       (each source sample contributes its own ``range(0, L)``). HF
       flash-attention varlen recovers ``cu_seqlens`` from the resets.

    For text-only data this is byte-equivalent to ``CatDataset`` except
    for the pad-rather-than-truncate trailing behaviour: every yielded
    sequence is exactly ``seq_len`` tokens, padded with ``pad_token_id``
    when the buffer can't be filled without splitting the next sample.
    """

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int, pad_token_id: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def _empty_buffers(self) -> tuple[dict[str, list], dict[str, list[Tensor]]]:
        return defaultdict(list), defaultdict(list)

    def _finalize(self, buf: dict[str, list], mm_items: dict[str, list[Tensor]], cur_len: int) -> dict:
        """Pad text-side fields to ``seq_len`` and emit as a single packed sample."""
        pad_len = self.seq_len - cur_len
        if pad_len < 0:
            # Should be impossible: we check ``cur + L > seq_len`` before
            # extending. Defensive — if a bug lets us overshoot, truncate
            # at the tail (text-only fields only; mm_items stay intact).
            self.logger.warning(
                f"MMCatDataset overshoot ({cur_len} > {self.seq_len}); truncating tail."
            )
            for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "mm_token_type_ids"):
                if key in buf:
                    buf[key] = buf[key][: self.seq_len]
            pad_len = 0

        if pad_len > 0:
            buf["input_ids"].extend([self.pad_token_id] * pad_len)
            buf["target_ids"].extend([self.pad_token_id] * pad_len)
            buf["position_ids"].extend([0] * pad_len)
            buf["loss_mask"].extend([False] * pad_len)
            if "mm_token_type_ids" in buf:
                buf["mm_token_type_ids"].extend([0] * pad_len)

        out: dict[str, Any] = dict(buf)
        if mm_items:
            out["mm_items"] = dict(mm_items)
        return out

    def __iter__(self):
        buf, mm_items = self._empty_buffers()
        cur_len = 0
        for sample in self.dataset:
            L = len(sample["input_ids"])
            if L > self.seq_len:
                # ``_process`` should have dropped these already (for the
                # multimodal path) or callers should have set seq_len
                # appropriately. Defensive skip — never split.
                self.logger.warning(
                    f"MMCatDataset dropping a sample of length {L} > seq_len {self.seq_len}."
                )
                continue

            if cur_len > 0 and cur_len + L > self.seq_len:
                yield self._finalize(buf, mm_items, cur_len)
                buf, mm_items = self._empty_buffers()
                cur_len = 0

            for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "mm_token_type_ids"):
                value = sample.get(key)
                if value is None:
                    continue
                buf[key].extend(value)
            for key, tensors in (sample.get("mm_items") or {}).items():
                mm_items[key].extend(tensors)
            cur_len += L

            if cur_len == self.seq_len:
                yield self._finalize(buf, mm_items, cur_len)
                buf, mm_items = self._empty_buffers()
                cur_len = 0


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


def mm_collate(samples: list[Sample]) -> Batch:
    """Collate a single MMCatDataset-packed sample into model-ready tensors.

    ``StatefulDataLoader`` is built with ``batch_size=1`` for this packer
    (the packing already produced a sequence of length ``seq_len * micro_batch_size``),
    so we operate on ``samples[0]`` directly. ``mm_kwargs`` are per-image
    tensors concatenated along dim=0 — same shape as the RL path produces
    (``src/prime_rl/trainer/rl/data.py``).
    """
    s = samples[0]
    batch: Batch = {
        "input_ids": torch.tensor(s["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "position_ids": torch.tensor(s["position_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "target_ids": torch.tensor(s["target_ids"], dtype=torch.long, device="cuda").unsqueeze(0),
        "loss_mask": torch.tensor(s["loss_mask"], dtype=torch.bool, device="cuda").unsqueeze(0),
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }
    if "mm_token_type_ids" in s:
        batch["mm_token_type_ids"] = torch.tensor(
            s["mm_token_type_ids"], dtype=torch.long, device="cuda"
        ).unsqueeze(0)
    mm_items = s.get("mm_items") or {}
    if mm_items:
        batch["mm_kwargs"] = {
            key: torch.cat([t.to("cuda") for t in tensors], dim=0).contiguous()
            for key, tensors in mm_items.items()
        }
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
    renderer: Renderer | None = None,
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
            renderer=renderer,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(
    dataset: StatefulIterableDataset,
    config: DataConfig,
    *,
    pad_token_id: int | None = None,
) -> StatefulDataLoader:
    if config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    elif config.pack_function == "mm_cat":
        if pad_token_id is None:
            raise ValueError(
                "mm_cat packer requires a pad_token_id; pass it from the tokenizer or model config."
            )
        mm_dataset = MMCatDataset(dataset, config.seq_len * config.micro_batch_size, pad_token_id)
        return StatefulDataLoader(mm_dataset, batch_size=1, collate_fn=mm_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
