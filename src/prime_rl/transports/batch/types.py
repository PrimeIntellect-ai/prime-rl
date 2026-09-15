import msgspec


class MMImageRef(msgspec.Struct, array_like=True, gc=False):
    url: str
    offset: int
    length: int


class MMRefs(msgspec.Struct, array_like=True, gc=False):
    images: list[MMImageRef]


# Routed experts are large per-token arrays. tolist() is too expensive, so we
# send raw bytes through msgpack and carry the shape/dtype needed to rebuild.
class RoutedExperts(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    data: bytes
    shape: list[int]  # [seq_len, layers, topk]
    dtype: str


# Sampling masks for top-p/top-k replay: flat int32 token-id bytes plus an
# int32 count per token position (0 = no mask); len(ids) == 4 * counts.sum().
class SamplingMask(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    ids: bytes
    counts: bytes


# Produced by the orchestrator's train sink; consumed in-process by
# ``prepare_batch``, which packs samples into per-rank ``MicroBatch``es.
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

    mm_refs: MMRefs | None = None

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

    # Appended fields only: array_like structs encode positionally, so appending
    # keeps the wire layout of earlier fields stable across versions.
    sampling_mask: SamplingMask | None = None

    # Identity of the branch this sample was built from, so the trainer can key
    # its per-token annotations back to the rollout trace. ``None`` on synthetic
    # samples (e.g. fake data).
    trace_id: str | None = None
    branch_index: int | None = None


# Orchestrator -> Trainer
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
    seq_lens: list[int]
    ref_logprobs: list[float] | None = None
    routed_experts: RoutedExperts | None = None

    mm_refs: MMRefs | None = None
    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Per-token component weight streams (see TrainingSample). ``None`` means
    # absent: no ce/ref_kl component, rl weight 1.0 everywhere — packing
    # materializes a stream as soon as one packed sample carries it.
    rl_weights: list[float] | None = None
    ce_weights: list[float] | None = None
    ref_kl_weights: list[float] | None = None

    # See TrainingSample.sampling_mask; appended for wire-layout stability.
    sampling_mask: SamplingMask | None = None

    # Per-sequence branch identity, parallel to ``sequence_lengths`` (see
    # TrainingSample.trace_id). ``""`` / ``-1`` mark an unknown sequence
    # (e.g. a dummy micro batch). ``None`` when no packed sample carried one.
    trace_ids: list[str] | None = None
    branch_indices: list[int] | None = None
