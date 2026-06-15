from typing import Literal

import msgspec

TrainingMode = Literal["rl", "opd", "sft"]


# Legacy encoded tensor payload retained for wire compatibility.
class EncodedTensor(msgspec.Struct, array_like=True, gc=False):
    dtype: str
    shape: list[int]
    data: bytes


# Lightweight image references shipped instead of materialized pixels when
# defer_mm_materialization is on: the orchestrator emits these and the trainer
# materializes pixels in its data loader.
class MMRefs(msgspec.Struct, array_like=True, gc=False):
    # Descriptor-only mm_data ({"mm_items": {...grid/placeholder...}, "mm_hashes": {...}}),
    # transport-safe (grids arrive as msgpack wire payloads, hashes as str). + the
    # candidate file:// image URIs for this sample. Trainer materializes from these.
    descriptor: dict
    uris: list[str]


# Routed experts are large per-token arrays. tolist() is too expensive, so we
# send raw bytes through msgpack and carry the shape/dtype needed to rebuild.
class RoutedExperts(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    data: bytes
    shape: list[int]  # [seq_len, layers, topk]
    dtype: str


# Orchestrator -> Packer
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    completion_temperatures: list[float]  # Per-token temperatures used during generation
    env_name: str
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None

    # Legacy eager multimodal kwargs. VLM training now ships mm_refs instead;
    # eager mm_kwargs are rejected before training.
    mm_kwargs: dict[str, EncodedTensor] | None = None

    routed_experts: RoutedExperts | None = None

    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Loss dispatch is batch-driven: rl/opd use default_loss_fn (with mode-specific
    # taus), sft uses sft_loss_fn. Stamped by the orchestrator from training_mode.
    training_mode: TrainingMode = "rl"

    # Lightweight image references (deferred materialization). Exactly one of
    # {mm_kwargs, mm_refs} is populated per multimodal sample. APPENDED LAST:
    # array_like=True structs encode positionally, so new fields must go at the
    # end to preserve the wire positions of existing fields.
    mm_refs: MMRefs | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingSample]
    step: int
    run_idx: int | None = None


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training."""

    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperatures: list[float]  # Per-token temperatures used during generation
    env_names: list[str]
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: RoutedExperts | None = None

    # See TrainingSample.mm_kwargs.
    mm_kwargs: dict[str, EncodedTensor] | None = None
    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Loss dispatch is batch-driven (rl/opd → default loss with mode-specific taus,
    # sft → sft loss). All samples packed into a micro batch share the same mode.
    training_mode: TrainingMode = "rl"
    rewards: list[float] | None = None

    # See TrainingSample.mm_refs. APPENDED LAST — array_like=True is positional, so
    # new fields go at the end to preserve existing field wire positions.
    mm_refs: MMRefs | None = None

    # Packer-derived metadata used for run-local token exports.
    run_id: str | None = None
    run_step: int | None = None

    # Packed multimodal sample boundaries. Sum equals len(input_ids) when present.
    seq_lens: list[int] | None = None
