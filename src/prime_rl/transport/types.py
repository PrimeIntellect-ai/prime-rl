import msgspec


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

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None

    routed_experts: list[list[list[int]]] | None = None  # [seq_len, layers, topk]

    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    sft_loss: bool = False  # When True, trainer uses SFT loss instead of RL loss for this sample

    # Per-token SFT-on-tool-body mask (parallel to prompt_ids + completion_ids).
    # True on tool body tokens whose tool name matches the env's SFTConfig.tool_names
    # filter (and is_content per the renderer attribution). None when the env has
    # no SFTConfig or SFTConfig.on_tool_outputs is False. Distinct from sft_loss
    # above: that switches the whole sample to the SFT loss function; this is a
    # per-token overlay co-existing with the RL loss in default_loss_fn.
    sft_mask: list[bool] | None = None
    # Per-env constant weight on the SFT advantage. The trainer overlays
    # ``alpha / n_sft_tokens`` (or ``alpha`` if trainer config sets
    # ``disable_echo``) on the advantages tensor at sft_mask positions.
    # None when sft_mask is None.
    sft_alpha: float | None = None


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
    routed_experts: list[list[list[int]]] | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None
    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    sft_loss: bool = False  # When True, trainer uses SFT loss instead of RL loss for this batch

    # Per-token SFT-on-tool-body mask (parallel to input_ids). Survives packing
    # and padding identical to ``loss_mask`` — see ``trainer/batch.py`` for the
    # overlay logic. None when no sample in this micro-batch carried an SFT mask.
    sft_mask: list[bool] | None = None
