import msgspec
import numpy as np


class MMImageRef(msgspec.Struct, array_like=True, gc=False):
    url: str
    offset: int
    length: int


class MMRefs(msgspec.Struct, array_like=True, gc=False):
    images: list[MMImageRef]


# Per-token arrays travel as raw buffers. The orchestrator fills them with bytes; the trainer
# decodes them as memoryviews into the received frame, so the gigabytes of routing and top
# logprobs in a step are never copied by the decoder.


# Routed experts are large per-token arrays. tolist() is too expensive, so we
# send raw bytes through msgpack and carry the shape/dtype needed to rebuild.
class RoutedExperts(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    data: memoryview
    shape: list[int]  # [seq_len, layers, topk]
    dtype: str


# Sampling masks for top-p/top-k replay: flat int32 token-id bytes plus an
# int32 count per token position (0 = no mask); len(ids) == 4 * counts.sum().
# ``logprobs`` optionally carries the sampler's float32 logprob of each id.
class SamplingMask(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    ids: memoryview
    counts: memoryview
    logprobs: memoryview | None = None


class TopLogprobs(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """Flat int32 ids, float32 logprobs and int32 row counts (zero for context)."""

    ids: memoryview
    logprobs: memoryview
    counts: memoryview


# A sample's training arrays kept by the env-server worker at ``url`` (see
# ``prime_rl.transports.payload``): the trainer fetches them by ``id`` and keeps the first
# ``num_tokens``.
class PayloadSlice(msgspec.Struct, array_like=True, gc=False):
    url: str
    id: str
    num_tokens: int
    fields: list[str]  # which of routed_experts / sampling_mask / top_logprobs it holds


# One branch's training arrays as an env-server worker serves them.
class BranchArrays(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    routed_experts: RoutedExperts | None = None
    sampling_mask: SamplingMask | None = None
    top_logprobs: TopLogprobs | None = None


# Produced by the orchestrator's train sink; consumed in-process by
# ``prepare_batch``, which packs samples into per-rank ``MicroBatch``es.
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example — one branch of a rollout as a flat token sequence.

    There is no prompt/completion split: an agentic, multi-turn branch interleaves context and
    model-sampled spans, so ``mask`` marks which tokens are trainable (model-sampled) and
    ``logprobs`` / ``temperatures`` are aligned per token. All four arrays share the length of
    ``token_ids``.

    Per-token streams are numpy arrays (int32 ids, bool mask, float32 values): a sample lives
    only in the orchestrator, and per-token Python lists would make its garbage collector walk
    every token. ``prepare_sample`` turns them into the ``MicroBatch`` wire format."""

    token_ids: np.ndarray
    mask: np.ndarray
    logprobs: np.ndarray
    temperatures: np.ndarray
    env_name: str
    ref_logprobs: np.ndarray | None = None  # reference-model logprobs (ref_kl component)

    mm_refs: MMRefs | None = None

    routed_experts: RoutedExperts | None = None

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

    # Appended fields only: array_like structs encode positionally, so appending
    # keeps the wire layout of earlier fields stable across versions.
    sampling_mask: SamplingMask | None = None

    # Identity of the branch this sample was built from, so the trainer can key
    # its per-token annotations back to the rollout trace. ``None`` on synthetic
    # samples (e.g. fake data).
    trace_id: str | None = None
    branch_index: int | None = None
    top_logprobs: TopLogprobs | None = None
    payload: PayloadSlice | None = None  # the arrays above, kept by an env server


# Per-token streams of a MicroBatch and the dtype each travels as. The orchestrator holds them as
# numpy arrays and the batch encoder ships each as its raw bytes (`encode_array`); the trainer
# decodes them as memoryviews into the received frame and reads them with `micro_batch_stream`.
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
}


def encode_array(value: object) -> memoryview:
    """msgspec ``enc_hook``: a numpy array as its raw bytes."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).data
    raise NotImplementedError(f"cannot encode {type(value).__name__}")


def micro_batch_stream(micro_batch: "MicroBatch", name: str) -> np.ndarray | None:
    """One per-token stream of a micro batch as a numpy array, whether it holds the array (the
    orchestrator) or its received bytes (the trainer). Views a received frame read-only."""
    value = getattr(micro_batch, name)
    if value is None:
        return None
    dtype = MICRO_BATCH_DTYPES[name]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=dtype)
    return np.asarray(value, dtype=dtype)


# Orchestrator -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training. Per-token streams are numpy arrays in the orchestrator
    and raw buffers on the wire (``MICRO_BATCH_DTYPES``)."""

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

    # See TrainingSample.sampling_mask; appended for wire-layout stability.
    sampling_mask: SamplingMask | None = None

    # Per-sequence branch identity, parallel to ``sequence_lengths`` (see
    # TrainingSample.trace_id). ``""`` / ``-1`` mark an unknown sequence
    # (e.g. a dummy micro batch). ``None`` when no packed sample carried one.
    trace_ids: list[str] | None = None
    branch_indices: list[int] | None = None
    top_logprobs: TopLogprobs | None = None
    # The samples' arrays kept by env servers, in packing order; the trainer fetches them
    # into the fields above.
    payloads: list[PayloadSlice] | None = None
    # Trailing padding tokens, folded into the last sequence's length by ``pad_micro_batch``.
    padding: int = 0
