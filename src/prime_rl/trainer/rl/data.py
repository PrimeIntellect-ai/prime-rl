import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.rl.planner import plan_digest, prepare_batch
from prime_rl.trainer.world import get_world
from prime_rl.transports.batch import (
    BatchReceiver,
    MicroBatch,
    MMRefs,
    setup_batch_receiver,
)
from prime_rl.transports.batch.types import MICRO_BATCH_DTYPES, SamplerSupport, StepBatch
from prime_rl.transports.payload import (
    delete_payloads,
    mark_trained,
    materialize_micro_batches,
    require_rollout_store,
)
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
    # its sampling mask (support_replayed) or a head of the sampler's distribution. support_logprobs is None
    # when no packed sample carries sampler logprobs.
    support_ids: Int[Tensor, "entries"] | None
    support_logprobs: Float[Tensor, "entries"] | None
    support_offsets: Int[Tensor, "seq_plus_one"] | None
    support_replayed: Bool[Tensor, "seq"] | None
    # Set by the trainer's micro batch preparation: whether the LM head may take its sparse backward
    sparse_head_backward: NotRequired[bool]

    # Multimodal inputs. The data loader sets mm_refs (undecoded image references), and prepare_micro_batch
    # replaces them with mm_kwargs (CPU tensors for the vision encoder).
    mm_refs: MMRefs | None
    mm_kwargs: NotRequired[dict[str, Tensor] | None]
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
    def commit(self, step: int, release: bool) -> None:
        pass

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
    micro_batch: TensorMicroBatch, materialize_images: Callable[[MMRefs], dict[str, Tensor]] | None
) -> TensorMicroBatch:
    """Prepare a micro batch for the training loop. Currently only multimodal samples need preparation."""
    micro_batch["mm_kwargs"] = None
    mm_refs = micro_batch.get("mm_refs")
    if mm_refs is not None:
        if materialize_images is None:
            raise ValueError("Received multimodal samples but [model.vlm] is not set")
        micro_batch["mm_kwargs"] = materialize_images(mm_refs)
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


@dataclass
class PreparedStep:
    """One step's micro batches for this rank, planned and built from the fetched samples."""

    step: int
    digest: int
    """The digest of the step's plans for every data rank (`plan_digest`)."""
    micro_batches: list[MicroBatch]
    failed: list[int]
    unbuildable: list[int]
    urls: list[str]
    ahead: bool
    """Prepared before the trainer took the step, while it trained the one before."""


class StepPrefetcher:
    """Prepares each step on a thread of its own, which owns the step receiver.

    As soon as the orchestrator offers a step, the thread plans it and builds this rank's micro
    batches, while the trainer still trains the step before. The trainer takes the step at its
    boundary (`take`): that request is still when the orchestrator learns the trainer took it.
    A step the orchestrator has not offered by then is prepared once it arrives."""

    def __init__(self, receiver: BatchReceiver, prepare: Callable[[StepBatch, int, bool], PreparedStep]):
        self.receiver = receiver
        self.prepare = prepare
        self.take_requested = threading.Event()
        self.taken = threading.Event()
        self.prepared: queue.SimpleQueue[PreparedStep | BaseException] = queue.SimpleQueue()
        self.error: BaseException | None = None
        threading.Thread(target=self._run, name="step-prefetch", daemon=True).start()

    def _run(self) -> None:
        try:
            while True:
                step, prepared = self.receiver.current_step, None
                while not self.take_requested.is_set():
                    if prepared is not None:
                        self.take_requested.wait()
                    elif self.receiver.poll():
                        prepared = self.prepare(self.receiver.batch(), step, True)
                self.take_requested.clear()
                self.receiver.wait()
                batch = self.receiver.receive()
                self.taken.set()
                self.prepared.put(prepared if prepared is not None else self.prepare(batch, step, False))
        except BaseException as error:
            self.error = error
            self.taken.set()
            self.prepared.put(error)

    def take(self) -> None:
        """Request the current step and block until the orchestrator handed it over."""
        self.taken.clear()
        # The thread sets the error before `taken` and then exits: an error from before the
        # clear shows here, and one from after it sets `taken` again, so this never waits on a
        # thread that is gone.
        if self.error is not None:
            raise self.error
        self.take_requested.set()
        self.taken.wait()
        if self.error is not None:
            raise self.error

    def next(self) -> PreparedStep:
        """The step `take` took, once prepared."""
        prepared = self.prepared.get()
        if isinstance(prepared, BaseException):
            raise prepared
        return prepared


class DataLoader:
    """Receives each step's samples from the orchestrator, plans the step's micro batches
    for every data rank and builds this rank's from the samples the env worker pools keep,
    the next step while the trainer still trains the current one (`StepPrefetcher`)."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        config: ZMQTransportConfig,
        seq_len: int,
        pad_to_multiple_of: int,
        bin_cost: Callable[[Sequence[int]], int],
    ):
        self.world = get_world()
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.bin_cost = bin_cost
        """The training FLOPs of a micro batch's sequences, which the data ranks are balanced by."""
        self.plan_digest = 0
        """The digest of the last step's plans for every data rank (`plan_digest`)."""

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.payload_pool = ThreadPoolExecutor(32, thread_name_prefix="payload")
        self.failed_micro_batches: list[int] = []
        """Micro batches of the last batch with a sample this rank could not fetch."""
        self.unbuildable_micro_batches: list[int] = []
        """Failed multimodal micro batches of the last batch that cannot run the vision encoder."""
        self.payload_urls: list[str] = []
        """The last batch's fetched samples, for `release_payloads`."""
        self.step = start_step

        self.receiver: BatchReceiver = setup_batch_receiver(
            output_dir,
            rank=self.world.rank,
            world_size=self.world.world_size,
            current_step=start_step,
            transport=config,
        )

        self.retain = config.retain
        self.stores: set[str] = set()
        """The rollout stores holding arrays of a step no checkpoint covers yet."""
        self.prefetcher = StepPrefetcher(self.receiver, self._prepare)

    def wait_for_batch(self) -> None:
        self.prefetcher.take()

    def get_batch(self) -> list[TensorMicroBatch]:
        prepared = self.prefetcher.next()
        self.step = prepared.step
        self.plan_digest = prepared.digest
        self.failed_micro_batches, self.unbuildable_micro_batches = prepared.failed, prepared.unbuildable
        self.payload_urls = prepared.urls
        get_logger().debug(
            f"Step {prepared.step}'s micro batches were {'ready' if prepared.ahead else 'built after'} "
            "when the trainer took it"
        )
        return [self._micro_batch_to_tensor(micro_batch) for micro_batch in prepared.micro_batches]

    def _prepare(self, batch: StepBatch, step: int, ahead: bool) -> PreparedStep:
        """Plan the step for every data rank and build this rank's micro batches. Runs on the
        prefetcher's thread."""
        start = time.perf_counter()
        plans_per_rank = prepare_batch(
            batch.samples,
            self.seq_len,
            self.dp_world_size,
            self.bin_cost,
            trained_version=batch.trained_version,
            min_version=batch.min_version,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        plans = plans_per_rank[self.dp_rank]
        get_logger().debug(
            f"Planned {len(plans)} micro batches per data rank from {len(batch.samples)} samples of step {step} "
            f"in {format_time(time.perf_counter() - start)}"
        )
        start = time.perf_counter()
        materialized = materialize_micro_batches(plans, self.payload_pool)
        get_logger().debug(
            f"Fetched the samples of {len(materialized.micro_batches)} micro batches of step {step} "
            f"in {format_time(time.perf_counter() - start)}"
        )
        return PreparedStep(
            step=step,
            digest=plan_digest(plans_per_rank),
            micro_batches=materialized.micro_batches,
            failed=materialized.failed,
            unbuildable=materialized.unbuildable,
            urls=materialized.urls,
            ahead=ahead,
        )

    def release_payloads(self) -> None:
        """Free the last batch's samples on the env workers. Call once every rank that may
        read them has fetched them. A retaining run's rollout stores keep them until
        `commit`: a restarted trainer fetches them again."""
        if self.retain:
            # Arrays a checkpoint holds (`file://`) stay with it: no rollout store has them.
            urls = [url for url in self.payload_urls if not url.startswith("file://")]
            stores = {url.rpartition("/")[0] for url in urls}
            for store in stores - self.stores:
                require_rollout_store(store)
            self.stores |= stores
            mark_trained(self.stores, urls, self.step, self.receiver.committed)
        else:
            delete_payloads(self.payload_urls)
        self.payload_urls = []

    def commit(self, step: int, release: bool) -> None:
        """A checkpoint covers every step up to ``step``: the orchestrator learns it with the
        next request, and with ``release`` the rollout stores this rank marked now."""
        self.receiver.committed = step
        if self.retain and release:
            mark_trained(self.stores, [], step, step)
            self.stores = set()

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
        support = _decode_support(micro_batch.sampler_support)

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


def _decode_support(support: SamplerSupport | None) -> dict[str, Tensor | None]:
    """The packed sampler support as token-major CSR tensors."""
    if support is None:
        return dict(support_ids=None, support_logprobs=None, support_offsets=None, support_replayed=None)
    counts = np.frombuffer(support.counts, dtype=np.int32)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    logprobs = None
    if support.logprobs is not None:
        logprobs = torch.from_numpy(np.frombuffer(support.logprobs, dtype=np.float32).copy())
    return dict(
        support_ids=torch.from_numpy(np.frombuffer(support.ids, dtype=np.int32).copy()),
        support_logprobs=logprobs,
        support_offsets=torch.from_numpy(offsets),
        support_replayed=torch.from_numpy(np.frombuffer(support.replayed, dtype=np.bool_).copy()),
    )


SCORE_CENTERING_DATA_ERRORS = (
    "Missing or out-of-vocabulary sampler top logprobs",
    "Sampler top logprobs must be finite and nonpositive",
    "Duplicate sampler head ids",
    "Sampler head mass exceeds one",
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
    fails (``rows``: each entry's token). Runs on the tensors' device without a host sync, so the trainer can
    validate on the GPU and check the flags once per step with ``raise_score_centering_data_errors``."""
    ids, rows = ids.long(), rows.long()

    def any_per_token(flags: Tensor) -> Tensor:
        return torch.zeros(rl_mask.numel(), dtype=torch.int32, device=ids.device).index_add_(0, rows, flags.int()) > 0

    in_vocab = (ids >= 0) & (ids < vocab_size)
    # Out-of-vocabulary ids get unique keys: they fail the first check, not this one
    keys = torch.where(in_vocab, rows * vocab_size + ids, -1 - torch.arange(ids.numel(), device=ids.device))
    sorted_keys, order = keys.sort()
    duplicate = torch.zeros_like(ids, dtype=torch.bool)
    duplicate[order[1:]] = sorted_keys[1:] == sorted_keys[:-1]
    q = torch.zeros(rl_mask.numel(), dtype=values.dtype, device=values.device).index_add_(0, rows, values.exp())
    errors = torch.stack(
        [
            ~any_per_token(torch.ones_like(rows, dtype=torch.bool)) | any_per_token(~in_vocab),
            any_per_token(~values.isfinite() | (values > 1e-6)),
            any_per_token(duplicate),
            q > 1.0001,
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
