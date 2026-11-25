import msgspec
import numpy as np


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


class SamplerSupport(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """The sampler's support per token in CSR form: an int32 count per token position (0 = none),
    the flat int32 ``ids`` and a bool per token marking a replayed sampling mask (any other row is
    a head of the sampler's distribution). ``logprobs`` optionally carries the sampler's float32
    logprob of each id, NaN where a packed sample has none."""

    ids: bytes
    counts: bytes
    replayed: bytes
    logprobs: bytes | None = None


# Where an env worker that keeps a trace's samples leaves the references to them:
# ``trace.info[PAYLOAD_INFO_KEY] = {"url": ..., "branches": {index: SampleSummary as a dict}}``.
PAYLOAD_INFO_KEY = "prime_rl_payload"

UNLABELED = -1
"""``SampleBody.versions`` of a sampled token whose engine reported no policy version: it
takes its episode's dispatch version."""

UNVERSIONED = -2
"""``SampleBody.versions`` of a token no policy version applies to (context): it never goes stale."""


class SampleBody(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """One training sample as the env worker that ran it keeps it: a trainable branch of a
    finished trace, flattened to per-token columns (raw bytes). The trainer fetches it by
    reference (``SampleRef``) and builds its micro batches from it.

    ``mask`` marks the tokens that train (a sampled node shared with an earlier branch is
    context here), ``sampled`` every model-sampled token (the trace's advantage spreads over
    them). ``versions`` is the int32 policy version that sampled each token (``UNLABELED``,
    ``UNVERSIONED``). The weight streams hold the env's and the algorithm's per-token loss
    routing, before the orchestrator's credit and staleness."""

    token_ids: bytes  # int32
    mask: bytes  # bool
    sampled: bytes  # bool
    logprobs: bytes  # float32
    versions: bytes  # int32
    rl_weights: bytes | None = None  # float32
    ce_weights: bytes | None = None  # float32
    ref_kl_weights: bytes | None = None  # float32
    ref_logprobs: bytes | None = None  # float32
    mm_token_type_ids: bytes | None = None  # int32
    mm_refs: MMRefs | None = None
    routed_experts: RoutedExperts | None = None
    sampler_support: SamplerSupport | None = None


class SampleRef(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A training sample by reference: the env worker that keeps its body (``url``/``id``),
    the first ``num_tokens`` of it that train, and what the orchestrator decided about it."""

    url: str
    id: str
    num_tokens: int
    env_name: str
    trace_id: str
    branch_index: int
    temperature: float
    advantage: float | None = None
    """Spread over the sample's sampled tokens; None when the trace has no credit."""
    rl_weight: float | None = None
    """The rl weight of every trainable token; None keeps the body's."""
    policy_start: int | None = None
    """The policy version the episode was dispatched at, which its unlabeled tokens take;
    None for a frozen-model sample, which never goes stale."""
    fields: list[str] = []
    """Which optional training arrays the body holds (see ``prime_rl.transports.payload``)."""
    routed_shape: list[int] | None = None  # [layers, top_k] of its routing rows
    routed_dtype: str | None = None
    images: list[list[int]] | None = None
    """``[offset, length]`` of each image placeholder run within ``num_tokens``."""


class MicroBatchPlan(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """One micro batch as the orchestrator ships it: its samples by reference, in packing
    order. The trainer fetches their bodies and builds the ``MicroBatch``."""

    samples: list[SampleRef]
    trained_version: int
    """The policy version this step trains (per-token staleness)."""
    min_version: int
    """Tokens sampled by an older policy version leave every loss component."""
    padding: int = 0
    """Trailing padding tokens."""
    dummy: bool = False
    """A zero-loss copy that evens out the micro batch count across data ranks."""


# Built by a trainer rank from a fetched ``SampleBody`` and its ``SampleRef``; packed into
# ``MicroBatch``es (``prime_rl.transports.batch.materialize``).
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example — one branch of a rollout as a flat token sequence.

    There is no prompt/completion split: an agentic, multi-turn branch interleaves context and
    model-sampled spans, so ``mask`` marks which tokens are trainable (model-sampled) and
    ``logprobs`` / ``temperatures`` are aligned per token. All four arrays share the length of
    ``token_ids``.

    Per-token streams are numpy arrays (int32 ids, bool mask, float32 values).
    ``prepare_sample`` turns them into a one-sequence ``MicroBatch``."""

    token_ids: np.ndarray
    mask: np.ndarray
    logprobs: np.ndarray
    temperatures: np.ndarray
    env_name: str
    ref_logprobs: np.ndarray | None = None  # reference-model logprobs (ref_kl component)

    mm_refs: MMRefs | None = None

    # mm_token_type_ids: token type ids per token [batch seq] (0=text, 1=image, 2=video)
    mm_token_type_ids: np.ndarray | None = None

    # Per-token component weight streams (full prompt+completion length),
    # stamped by the orchestrator from the env's algorithm. The training loss
    # is a sum of three components, each normalized by its own global token
    # count: rl (importance-weighted PG + KL), ce (masked NLL), and ref_kl
    # (reverse KL to a reference model as the PG signal). A weight scales that
    # component's per-token loss; 0.0 leaves the token out of the component
    # (mask and denominator). ``None`` means absent: no ce/ref_kl component,
    # and an rl weight of 1.0 on every trainable token — so the plain GRPO
    # wire stays as small as before.
    rl_weights: np.ndarray | None = None
    ce_weights: np.ndarray | None = None
    ref_kl_weights: np.ndarray | None = None

    # Per-token advantages (full prompt+completion length), the fourth stream:
    # the orchestrator broadcasts the rollout's scalar over the completion for
    # scalar algorithms. ``None`` means no rl credit assigned — legal only for
    # samples without live rl member tokens (the trainer raises otherwise).
    advantages: np.ndarray | None = None

    # Identity of the branch this sample was built from, so the trainer can key
    # its per-token annotations back to the rollout trace. ``None`` on synthetic
    # samples (e.g. fake data).
    trace_id: str | None = None
    branch_index: int | None = None
    staleness: np.ndarray | None = None
    # The branch's routing rows and sampler support, cut to its tokens; None when it has none.
    routed_experts: RoutedExperts | None = None
    sampler_support: SamplerSupport | None = None


# Per-token streams of a MicroBatch and the dtype of each. A trainer rank builds them as numpy
# arrays and reads each with `np.frombuffer`.
MICRO_BATCH_DTYPES: dict[str, type[np.generic]] = {
    "input_ids": np.int32,
    "loss_mask": np.bool_,
    "advantages": np.float32,
    "inference_logprobs": np.float32,
    "position_ids": np.int32,
    "temperatures": np.float32,
    "ref_logprobs": np.float32,
    "mm_token_type_ids": np.int32,
    "rl_weights": np.float32,
    "ce_weights": np.float32,
    "ref_kl_weights": np.float32,
    "staleness": np.int16,
}


class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training, built by a trainer rank from a ``MicroBatchPlan``.
    Per-token streams are numpy arrays of the ``MICRO_BATCH_DTYPES``."""

    input_ids: memoryview
    loss_mask: memoryview
    advantages: memoryview
    inference_logprobs: memoryview
    position_ids: memoryview
    sequence_lengths: list[int]
    temperatures: memoryview  # Per-token temperatures used during generation
    env_names: list[str]  # One per sequence, parallel to ``sequence_lengths``
    seq_lens: list[int]
    ref_logprobs: memoryview | None = None
    routed_experts: RoutedExperts | None = None

    mm_refs: MMRefs | None = None
    # mm_token_type_ids: token type ids per token [batch seq] (0=text, 1=image, 2=video)
    mm_token_type_ids: memoryview | None = None

    # Per-token component weight streams (see TrainingSample). ``None`` means
    # absent: no ce/ref_kl component, rl weight 1.0 everywhere — packing
    # materializes a stream as soon as one packed sample carries it.
    rl_weights: memoryview | None = None
    ce_weights: memoryview | None = None
    ref_kl_weights: memoryview | None = None

    sampler_support: SamplerSupport | None = None

    # Per-sequence branch identity, parallel to ``sequence_lengths`` (see
    # TrainingSample.trace_id). ``""`` / ``-1`` mark an unknown sequence
    # (e.g. a dummy micro batch). ``None`` when no packed sample carried one.
    trace_ids: list[str] | None = None
    branch_indices: list[int] | None = None
    # Trailing padding tokens, folded into the last sequence's length by ``pad_micro_batch``.
    padding: int = 0
    staleness: memoryview | None = None
