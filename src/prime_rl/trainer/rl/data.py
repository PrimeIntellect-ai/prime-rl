import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.multimodal import ForwardPolicy, MultimodalAdapter
from prime_rl.trainer.multimodal import materialize_mm_refs
from prime_rl.trainer.world import get_world
from prime_rl.transports.batch import (
    BatchReceiver,
    MicroBatch,
    MMRefs,
    TransportConfig,
    setup_batch_receiver,
)
from prime_rl.transports.batch.types import MICRO_BATCH_DTYPES, SamplingMask, TopLogprobs
from prime_rl.transports.payload import delete_payloads, resolve_payloads
from prime_rl.utils.logger import format_time, get_logger


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    ref_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]
    temperatures: Float[Tensor, "batch seq"]  # Per-token temperatures
    env_names: list[str]  # One per sequence, parallel to sequence_lengths
    sequence_lengths: list[int]
    padding: int  # Trailing padding tokens, folded into the last sequence's length

    # Per-sequence branch identity, parallel to sequence_lengths; None on
    # synthetic data. "" / -1 mark an unknown sequence (e.g. a dummy batch).
    trace_ids: list[str] | None
    branch_indices: list[int] | None

    # Batch level
    seq_lens: Int[Tensor, "segments"]

    # MoE router replay
    routed_experts: Int[Tensor, "batch seq layers topk"] | None

    # Sampler support per token as CSR rows: token i owns entries support_offsets[i]:support_offsets[i + 1],
    # its sampling mask (support_replayed) and/or its top logprobs. support_logprobs is None when some head
    # carries no sampler logprobs.
    support_ids: Int[Tensor, "entries"] | None
    support_logprobs: Float[Tensor, "entries"] | None
    support_offsets: Int[Tensor, "seq_plus_one"] | None
    support_replayed: Bool[Tensor, "seq"] | None

    # Multimodal inputs. The data loader sets mm_refs (undecoded image references), and prepare_micro_batch
    # replaces them with mm_kwargs (CPU tensors for the vision encoder) and mm_forward_policy.
    mm_refs: MMRefs | None
    mm_kwargs: NotRequired[dict[str, Tensor] | None]
    mm_forward_policy: NotRequired[ForwardPolicy | None]
    # mm_token_type_ids: token type per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    # Per-token component weight streams. ``None`` means absent: no ce/ref_kl
    # component, rl weight 1.0 on every loss-masked token.
    rl_weights: Float[Tensor, "batch seq"] | None
    ce_weights: Float[Tensor, "batch seq"] | None
    ref_kl_weights: Float[Tensor, "batch seq"] | None

    # Per-token staleness, -1 on tokens the policy did not sample
    staleness: Int[Tensor, "batch seq"] | None


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig, seq_len: int, dp_world_size: int):
        self.world = get_world()
        self.dp_world_size = dp_world_size
        self.non_dp_world_size = self.world.world_size // self.dp_world_size
        self.dp_rank = self.world.rank // self.non_dp_world_size

        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // self.dp_world_size
        self.seq_len = seq_len
        self.generate_samples = config.generate_samples
        self.doc_len = config.doc_len
        self.batch_counter = 0

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[TensorMicroBatch]:
        if not self.generate_samples:
            get_micro_batch_fn = self._get_micro_batch
        else:
            get_micro_batch_fn = self._get_sample_micro_batch

        # Seeded by the global micro batch index, so every DP layout trains the same batch and
        # all CP ranks of a data parallel group receive the same micro batch.
        micro_batches = []
        for micro_batch_idx in range(self.num_micro_batches):
            seed = self.batch_counter * 1000000 + self.dp_rank * self.num_micro_batches + micro_batch_idx
            generator = torch.Generator().manual_seed(seed)
            micro_batches.append(get_micro_batch_fn(generator))

        self.batch_counter += 1
        return micro_batches

    def _get_sample_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        total_seq_len = 0
        input_ids = []
        position_ids = []
        sequence_lengths = []

        while total_seq_len < self.seq_len:
            if self.doc_len is not None:
                seq_len_to_generate = self.doc_len
            else:
                # Generate reasonably long documents
                seq_len_to_generate = torch.randint(1, self.seq_len // 8, (1,), generator=generator).item()
            if seq_len_to_generate + total_seq_len > self.seq_len:
                seq_len_to_generate = self.seq_len - total_seq_len
            total_seq_len += seq_len_to_generate
            sequence_lengths.append(seq_len_to_generate)
            tmp_input_ids = torch.randint(0, 120000, (seq_len_to_generate,), generator=generator).long()
            tmp_position_ids = torch.arange(seq_len_to_generate).long()

            input_ids.append(tmp_input_ids)
            position_ids.append(tmp_position_ids)

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
        advantages = torch.randn(input_ids.shape[0], generator=generator)
        inference_logprobs = torch.randn(input_ids.shape[0], generator=generator)

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(input_ids.shape[0]).unsqueeze(0),
            "env_names": ["fake"] * len(sequence_lengths),
            "sequence_lengths": sequence_lengths,
            "padding": 0,
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": loss_mask.unsqueeze(0),
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "support_ids": None,
            "support_logprobs": None,
            "support_offsets": None,
            "support_replayed": None,
            "mm_refs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
            "staleness": None,
        }

    def _get_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        return {
            "input_ids": torch.randint(
                0,
                100,
                (
                    1,
                    self.seq_len,
                ),
                generator=generator,
            ),
            "position_ids": torch.cat([torch.arange(self.seq_len)]).unsqueeze(0),
            "advantages": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "inference_logprobs": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(self.seq_len).unsqueeze(0),
            "env_names": ["fake"],
            "sequence_lengths": [self.seq_len],
            "padding": 0,
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "seq_lens": torch.tensor([self.seq_len], dtype=torch.long),
            "routed_experts": None,
            "support_ids": None,
            "support_logprobs": None,
            "support_offsets": None,
            "support_replayed": None,
            "mm_refs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
            "staleness": None,
        }


def prepare_micro_batch(
    micro_batch: TensorMicroBatch, processor: Any | None, mm_adapter: MultimodalAdapter | None
) -> TensorMicroBatch:
    """Prepare a micro batch for the training loop. Currently only multimodal samples need preparation."""
    micro_batch["mm_kwargs"] = None
    micro_batch["mm_forward_policy"] = None
    mm_refs = micro_batch.get("mm_refs")
    if mm_refs is not None:
        if processor is None or mm_adapter is None:
            raise ValueError("Received multimodal samples but [model.vlm] is not set")
        materialized = materialize_mm_refs(mm_refs, processor, mm_adapter)
        micro_batch["mm_kwargs"] = materialized.kwargs
        micro_batch["mm_forward_policy"] = materialized.forward_policy
        micro_batch["mm_refs"] = None
    return micro_batch


def drop_micro_batches(micro_batches: list[TensorMicroBatch], indices: list[int]) -> None:
    """Keep micro batches out of every loss component (they still run forward/backward,
    so all ranks issue the same collectives)."""
    for index in indices:
        micro_batch = micro_batches[index]
        micro_batch["loss_mask"] = torch.zeros_like(micro_batch["loss_mask"])
        for name in ("rl_weights", "ce_weights", "ref_kl_weights"):
            if micro_batch[name] is not None:
                micro_batch[name] = torch.zeros_like(micro_batch[name])


class DataLoader:
    """Receives packed micro batches from the orchestrator, one stream per DP rank."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        config: TransportConfig,
    ):
        self.world = get_world()

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.payload_pool = ThreadPoolExecutor(32, thread_name_prefix="payload")
        self.failed_micro_batches: list[int] = []
        """Micro batches of the last batch whose arrays could not be fetched on this rank."""
        self.payload_urls: list[str] = []
        """The last batch's fetched arrays, for `release_payloads`."""

        self.receiver: BatchReceiver = setup_batch_receiver(
            output_dir,
            data_rank=dp_rank,
            rank=self.world.rank,
            world_size=self.world.world_size,
            current_step=start_step,
            transport=config,
        )

    def wait_for_batch(self) -> None:
        self.receiver.wait()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        start = time.perf_counter()
        self.failed_micro_batches, self.payload_urls = resolve_payloads(micro_batches, self.payload_pool)
        get_logger().debug(
            f"Fetched payloads of {len(micro_batches)} micro batches in {format_time(time.perf_counter() - start)}"
        )
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def release_payloads(self) -> None:
        """Free the last batch's arrays on the env servers. Call once every rank that may
        read them has fetched them."""
        delete_payloads(self.payload_urls)
        self.payload_urls = []

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct of raw buffers) to a TensorMicroBatch (dict with tensors)."""
        routed_experts = None
        packed_routed_experts = micro_batch.routed_experts
        if packed_routed_experts is not None:
            routed_experts = (
                torch.frombuffer(
                    packed_routed_experts.data,
                    dtype=_torch_dtype(packed_routed_experts.dtype),
                )
                .reshape(packed_routed_experts.shape)
                .unsqueeze(0)
            )
        support = _merge_support(micro_batch.top_logprobs, micro_batch.sampling_mask)

        def stream(name: str, dtype: type[np.generic]) -> Tensor | None:
            """A per-token stream as a [1, seq] tensor: one numpy copy out of the received frame
            (which stays read-only) that also widens it to the trainer's dtype."""
            values = getattr(micro_batch, name)
            if values is None:
                return None
            return torch.from_numpy(np.frombuffer(values, MICRO_BATCH_DTYPES[name]).astype(dtype)).unsqueeze(0)

        return TensorMicroBatch(
            input_ids=stream("input_ids", np.int64),
            position_ids=stream("position_ids", np.int64),
            advantages=stream("advantages", np.float32),
            inference_logprobs=stream("inference_logprobs", np.float32),
            ref_logprobs=stream("ref_logprobs", np.float32),
            loss_mask=stream("loss_mask", np.bool_),
            temperatures=stream("temperatures", np.float32),
            env_names=micro_batch.env_names,
            sequence_lengths=micro_batch.sequence_lengths,
            padding=micro_batch.padding,
            trace_ids=micro_batch.trace_ids,
            branch_indices=micro_batch.branch_indices,
            seq_lens=torch.tensor(micro_batch.seq_lens, dtype=torch.long),
            mm_refs=micro_batch.mm_refs,
            mm_token_type_ids=stream("mm_token_type_ids", np.int64),
            routed_experts=routed_experts,
            **support,
            rl_weights=stream("rl_weights", np.float32),
            ce_weights=stream("ce_weights", np.float32),
            ref_kl_weights=stream("ref_kl_weights", np.float32),
            staleness=stream("staleness", np.int16),
        )


def _merge_support(top_logprobs: TopLogprobs | None, sampling_mask: SamplingMask | None) -> dict[str, Tensor | None]:
    """Merge the top-logprobs and sampling-mask heads into token-major CSR support fields."""
    heads = [head for head in (top_logprobs, sampling_mask) if head is not None]
    if not heads:
        return dict(support_ids=None, support_logprobs=None, support_offsets=None, support_replayed=None)
    head_counts = [np.frombuffer(head.counts, dtype=np.int32) for head in heads]
    num_tokens = len(head_counts[0])
    offsets = np.zeros(num_tokens + 1, dtype=np.int64)
    np.cumsum(sum(head_counts), out=offsets[1:])
    rows = np.concatenate([np.repeat(np.arange(num_tokens), counts) for counts in head_counts])
    order = np.argsort(rows, kind="stable")
    ids = np.concatenate([np.frombuffer(head.ids, dtype=np.int32) for head in heads])[order]
    logprobs = None
    if all(head.logprobs is not None for head in heads):
        logprobs = np.concatenate([np.frombuffer(head.logprobs, dtype=np.float32) for head in heads])[order]
    replayed = (
        np.frombuffer(sampling_mask.counts, dtype=np.int32) > 0
        if sampling_mask is not None
        else np.zeros(num_tokens, dtype=np.bool_)
    )
    return dict(
        support_ids=torch.from_numpy(ids),
        support_logprobs=torch.from_numpy(logprobs) if logprobs is not None else None,
        support_offsets=torch.from_numpy(offsets),
        support_replayed=torch.from_numpy(replayed),
    )


SCORE_CENTERING_DATA_ERRORS = (
    "Missing or out-of-vocabulary sampler top logprobs",
    "Sampler top logprobs must be finite and nonpositive",
    "Duplicate ids in the replayed support",
    "Sampler probabilities must normalize over the recorded support",
    "Sampled token is missing from the recorded support",
)


def score_centering_data_errors(
    input_ids: Int[Tensor, " seq"],
    rl_mask: Bool[Tensor, " seq"],
    ids: Int[Tensor, " entries"],
    values: Float[Tensor, " entries"],
    rows: Int[Tensor, " entries"],
    replayed: Bool[Tensor, " seq"],
    *,
    vocab_size: int,
) -> Bool[Tensor, " checks"]:
    """Flag each check in ``SCORE_CENTERING_DATA_ERRORS`` that the CSR sampler evidence on ``rl_mask`` tokens
    fails (``rows``: each entry's token). Tokens without a sampling mask carry i.i.d. sampler draws, which may
    repeat. Runs on the tensors' device without a host sync, so the trainer can validate on the GPU and check
    the flags once per step with ``raise_score_centering_data_errors``."""
    ids, rows = ids.long(), rows.long()

    def any_per_token(flags: Tensor) -> Tensor:
        return torch.zeros(rl_mask.numel(), dtype=torch.int32, device=ids.device).index_add_(0, rows, flags.int()) > 0

    keys = torch.where(replayed[rows], rows * vocab_size + ids, -1 - torch.arange(ids.numel(), device=ids.device))
    sorted_keys, order = keys.sort()
    duplicate = torch.zeros_like(keys, dtype=torch.bool)
    duplicate[order[1:]] = sorted_keys[1:] == sorted_keys[:-1]
    q = torch.zeros(rl_mask.numel(), dtype=values.dtype, device=values.device).index_add_(0, rows, values.exp())
    errors = torch.stack(
        [
            ~any_per_token(torch.ones_like(rows, dtype=torch.bool)) | any_per_token((ids < 0) | (ids >= vocab_size)),
            any_per_token(~values.isfinite() | (values > 1e-6)),
            any_per_token(duplicate),
            replayed & ~torch.isclose(q, torch.ones_like(q), atol=1e-4, rtol=0),
            replayed & ~any_per_token(ids == input_ids[rows]),
        ],
        dim=-1,
    )
    return (errors & rl_mask.unsqueeze(-1)).any(0)


def raise_score_centering_data_errors(errors: list[Bool[Tensor, " checks"]]) -> None:
    """Raise the first failed check of the first failing micro batch, given each micro batch's
    ``score_centering_data_errors`` in order. Syncs once."""
    if not errors:
        return
    failed = torch.stack(errors).flatten().cpu().nonzero()
    if failed.numel():
        raise ValueError(SCORE_CENTERING_DATA_ERRORS[int(failed[0]) % len(SCORE_CENTERING_DATA_ERRORS)])


def validate_score_centering_data(batch: TensorMicroBatch, *, vocab_size: int) -> None:
    """Validate sampler evidence on the CPU before the LM head consumes it."""
    mask = batch["loss_mask"][0]
    if batch["rl_weights"] is not None:
        mask = mask & (batch["rl_weights"][0] != 0)
    if not mask.any():
        return
    ids, values, offsets = batch["support_ids"], batch["support_logprobs"], batch["support_offsets"]
    if ids is None or values is None:
        raise ValueError("Score centering requires top logprobs on every RL token")
    counts = offsets.diff()
    rows = torch.repeat_interleave(torch.arange(len(counts)), counts)
    errors = score_centering_data_errors(
        batch["input_ids"][0], mask, ids, values, rows, batch["support_replayed"], vocab_size=vocab_size
    )
    raise_score_centering_data_errors([errors])


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a numpy/torch dtype name (e.g. ``"float32"``) to torch.dtype."""
    # Strip the ``numpy.`` prefix some dtype reprs carry.
    name = name.replace("numpy.", "")
    if hasattr(torch, name):
        return getattr(torch, name)
    # numpy ↔ torch alias mismatches (rare but possible) — fall back via numpy.
    import numpy as np

    return torch.from_numpy(np.zeros(1, dtype=np.dtype(name))).dtype
