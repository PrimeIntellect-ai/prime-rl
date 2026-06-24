from typing import Literal

import msgspec

TrainingMode = Literal["rl", "opd", "sft"]


# Encoded tensor: {dtype: "float32", shape: [...], data: <bytes>}.
# Mirrors verifiers.utils.serve_utils.msgpack_encoder so the same wire
# shape is used end-to-end from renderer → orchestrator → trainer.
class EncodedTensor(msgspec.Struct, array_like=True, gc=False):
    dtype: str
    shape: list[int]
    data: bytes


# Routed experts are large per-token arrays. tolist() is too expensive, so we
# send raw bytes through msgpack and carry the shape/dtype needed to rebuild.
class RoutedExperts(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    data: bytes
    shape: list[int]  # [seq_len, layers, topk]
    dtype: str


class MMRefs(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """Raw multimodal sidecar references for one sample.

    ``descriptor`` carries JSON-safe renderer metadata (hashes, grids, placeholder
    layout). ``uris`` carries the raw image files that the trainer materializes
    with its own processor. Processed tensors are intentionally not part of this
    transport.
    """

    descriptor: dict
    uris: list[str]


# Orchestrator -> Packer
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example — one branch of a rollout as a flat token sequence.

    There is no prompt/completion split: an agentic, multi-turn branch interleaves context and
    model-sampled spans, so ``mask`` marks which tokens are trainable (model-sampled) and
    ``logprobs`` / ``temperatures`` are aligned per token. All four arrays share the length of
    ``token_ids``."""

    token_ids: list[int]
    mask: list[bool]
    logprobs: list[float]
    temperatures: list[float]
    env_name: str
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None

    # Legacy eager multimodal payloads are rejected by the v1 raw-image-ref
    # path. Keep the field so old batches fail at a clear boundary.
    mm_kwargs: dict[str, EncodedTensor] | None = None

    mm_refs: MMRefs | None = None

    routed_experts: RoutedExperts | None = None

    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Loss dispatch is batch-driven: rl/opd use default_loss_fn (with mode-specific
    # taus), sft uses sft_loss_fn. Stamped by the orchestrator from training_mode.
    training_mode: TrainingMode = "rl"


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
    sequence_lengths: list[int]
    temperatures: list[float]  # Per-token temperatures used during generation
    env_names: list[str]
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: RoutedExperts | None = None

    # Legacy eager multimodal payloads are rejected by the v1 raw-image-ref
    # path. Keep the field so old batches fail at a clear boundary.
    mm_kwargs: dict[str, EncodedTensor] | None = None
    mm_refs: MMRefs | None = None
    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Loss dispatch is batch-driven (rl/opd → default loss with mode-specific taus,
    # sft → sft loss). All samples packed into a micro batch share the same mode.
    training_mode: TrainingMode = "rl"
    rewards: list[float] | None = None

    # Packer-derived metadata used for run-local token exports.
    run_id: str | None = None
    run_step: int | None = None
