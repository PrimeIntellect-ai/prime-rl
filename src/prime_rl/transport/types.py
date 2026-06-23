import msgspec


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
    ref_logprobs: list[float] | None = None  # reference-model logprobs (ref_kl component)
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

    # Per-token component weight streams (full prompt+completion length),
    # stamped by the orchestrator from the env's algorithm. The training loss
    # is a sum of three components, each normalized by its own global token
    # count: rl (importance-weighted PG + KL), ce (masked NLL), and ref_kl
    # (reverse KL to a reference model as the PG signal). A weight scales that
    # component's per-token loss; 0.0 leaves the token out of the component
    # (mask and denominator). ``None`` means absent: no ce/ref_kl component,
    # and an rl weight of 1.0 on every trainable token — so the plain GRPO
    # wire stays as small as before.
    rl_weights: list[float] | None = None
    ce_weights: list[float] | None = None
    ref_kl_weights: list[float] | None = None

    # Per-token advantages (full prompt+completion length), the fourth stream:
    # the orchestrator broadcasts the rollout's scalar over the completion for
    # scalar algorithms. ``None`` means no rl credit assigned — legal only for
    # samples without live rl member tokens (the trainer raises otherwise).
    advantages: list[float] | None = None

    # Orchestrator-internal, cleared before transport: interleaving's
    # provenance record for env-provided observation tokens — one
    # ``[completion_start, step_idx, step_prompt_start, length]`` entry per
    # span that landed as a later-turn prompt extension, mapping sample
    # positions back to trajectory-step coordinates. Algorithms that train
    # on observations (echo) consume it at group time and write the
    # ``ce_weights`` stream directly.
    obs_spans: list[list[int]] | None = None


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
    ref_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: RoutedExperts | None = None

    # See TrainingSample.mm_kwargs.
    mm_kwargs: dict[str, EncodedTensor] | None = None
    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Per-token component weight streams (see TrainingSample). ``None`` means
    # absent: no ce/ref_kl component, rl weight 1.0 everywhere — packing
    # materializes a stream as soon as one packed sample carries it.
    rl_weights: list[float] | None = None
    ce_weights: list[float] | None = None
    ref_kl_weights: list[float] | None = None
    rewards: list[float] | None = None

    # Packer-derived metadata used for run-local token exports.
    run_id: str | None = None
    run_step: int | None = None
