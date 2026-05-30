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

    # Generic multimodal kwargs: flat dict keyed by the kwarg names the
    # model's forward expects (e.g. {"pixel_values": ..., "image_grid_thw":
    # ...} for Qwen3-VL; just {"pixel_values": ...} for Gemma3). The
    # orchestrator batches per-image renderer items by torch.cat along
    # dim=0 generically — no model-specific knowledge in prime-rl. The
    # trainer ``**`` -unpacks this into the model forward, so any VLM
    # whose HF processor / forward agree on kwarg names works without
    # touching this transport.
    mm_kwargs: dict[str, EncodedTensor] | None = None

    routed_experts: RoutedExperts | None = None

    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Loss dispatch is batch-driven: rl/opd use default_loss_fn (with mode-specific
    # taus), sft uses sft_loss_fn. Stamped by the orchestrator from training_mode.
    training_mode: TrainingMode = "rl"

    # Per-token echo alpha (parallel to prompt_ids + completion_ids).
    # Three-state encoding:
    #   - None (whole field):     env opted out of echo entirely (no overlay)
    #   - None per-token:         this position is NOT echoed; the RL gradient
    #                             (or prompt-mask exclusion) applies as usual
    #   - float per-token:        this position IS echoed at this alpha; in
    #                             ``prepare_sample`` the per-token alpha
    #                             overwrites the RL advantage AND flips
    #                             loss_mask=True, making it a pure
    #                             cross-entropy contribution
    # Built by ``_step_echo_alpha`` from the renderer's prompt_attribution
    # (message_roles, message_indices, is_content, message_tool_names) and
    # the env's ``EchoConfig`` per-role sub-configs. The three-state encoding
    # is required because ``alpha=0`` is a legitimate "kill the RL gradient"
    # value (the assistant-role use case), distinct from "not echoed at all".
    # Distinct from ``training_mode='sft'`` above: that switches the whole
    # sample to the SFT loss function; this is a per-token overlay co-existing
    # with the RL loss in default_loss_fn.
    echo_alpha: list[float | None] | None = None


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

    # Per-token echo mask (parallel to input_ids): True where the token is an
    # echo position (per-role cross-entropy overlay, see ``EchoConfig``).
    # Survives packing and padding identical to ``loss_mask``. None when no
    # sample in this micro-batch carried any echo positions. Used by the loss
    # function to skip the trust-region clip and zero out the IS-ratio on echo
    # positions (the off-policy correction concept doesn't apply to tokens the
    # model didn't sample).
    echo_mask: list[bool] | None = None
