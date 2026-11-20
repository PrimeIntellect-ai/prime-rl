from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.model import forward, is_tt_moe_model, take_moe_routing_stats
from prime_rl.trainer.models.layers.lm_head import TokenSupport, support_replay_mask
from prime_rl.trainer.optim.offload import GradientOffloadManager
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.rl.annotations import AnnotationWriter
from prime_rl.trainer.rl.data import (
    MMRefs,
    TensorMicroBatch,
    prepare_micro_batch,
    raise_score_centering_data_errors,
    score_centering_data_errors,
    validate_score_centering_data,
)
from prime_rl.trainer.rl.loss import (
    _mismatch_kl_from_log_ratio,
    compute_entropy,
    compute_loss,
    selective_log_softmax,
    selective_log_softmax_with_support,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.utils import (
    STALENESS_BIN_EDGES,
    STALENESS_BINS,
    TensorStats,
    begin_backward,
    finish_backward,
    prepare_gradient_offload,
)
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import (
    gather_for_cp,
    gather_for_cp_wo_grad,
    gather_ragged_for_cp,
    setup_cp_params,
    shard_for_cp,
    validate_cp_seq_lens,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.worker_pool import WorkerPool


class ForwardBackwardResult(NamedTuple):
    tensor_stats: TensorStats
    routing_stats: list[Tensor]
    num_tokens: int
    flops: float
    grad_scale: float


class LossScales(NamedTuple):
    """Each loss component's count relative to the largest count, and the factor that scales the accumulated
    gradients to the per-token mean over the global batch."""

    rl: float
    ce: float
    ref_kl: float
    max_scale: float
    grad_scale: float


class ContextParallel(NamedTuple):
    rank: int
    size: int
    group: dist.ProcessGroup


class LossInputs(NamedTuple):
    """The micro batch's loss-side tensors on the GPU. Field names match the micro batch's keys."""

    loss_mask: Tensor
    advantages: Tensor
    inference_logprobs: Tensor
    ref_logprobs: Tensor | None
    rl_weights: Tensor | None
    ce_weights: Tensor | None
    ref_kl_weights: Tensor | None

    @classmethod
    def from_micro_batch(cls, micro_batch: TensorMicroBatch) -> "LossInputs":
        return cls(*(micro_batch[name].to("cuda") if micro_batch[name] is not None else None for name in cls._fields))

    @property
    def rl_mask(self) -> Tensor:
        return self.loss_mask[0] if self.rl_weights is None else self.loss_mask[0] & (self.rl_weights[0] != 0)

    @property
    def mismatch_mask(self) -> Tensor:
        """Tokens with meaningful sampling logprobs: rl and ref_kl member tokens are policy-sampled, while ce
        tokens come from a frozen model."""
        if self.rl_weights is None and self.ref_kl_weights is None:
            return self.loss_mask
        sampled_mask = (self.rl_weights != 0) if self.rl_weights is not None else self.loss_mask
        if self.ref_kl_weights is not None:
            sampled_mask = sampled_mask | (self.ref_kl_weights != 0)
        return self.loss_mask & sampled_mask


class ModelInputs(NamedTuple):
    """The micro batch's forward inputs on the GPU, sharded for this CP rank."""

    input_ids: Tensor
    position_ids: Tensor
    labels: Tensor
    temperatures: Tensor
    seq_lens: Tensor
    seq_lens_are_pre_shard: bool
    routed_experts: Tensor | None
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Tensor | None
    label_support: TokenSupport | None
    # Support entries per CP rank, to gather the support logprobs of the whole sequence
    cp_shard_entries: list[int] | None


def sparse_head_backward(micro_batch: TensorMicroBatch) -> bool:
    """Whether the LM head may skip its dense softmax backward: only rl tokens carry gradient (no ce / ref_kl
    stream), and each is replayed against a support holding its label."""
    if (
        micro_batch["support_offsets"] is None
        or micro_batch["ce_weights"] is not None
        or micro_batch["ref_kl_weights"] is not None
    ):
        return False
    rl_tokens = micro_batch["loss_mask"][0]
    if micro_batch["rl_weights"] is not None:
        rl_tokens = rl_tokens & (micro_batch["rl_weights"][0] != 0)
    # Without rl tokens the check below passes vacuously. Such a micro batch trains nothing, so it takes the dense
    # backward of micro batches whose support misses a label.
    if not rl_tokens.any():
        return False
    support = TokenSupport(
        ids=micro_batch["support_ids"], offsets=micro_batch["support_offsets"], replayed=micro_batch["support_replayed"]
    )
    replay = support_replay_mask(support, micro_batch["input_ids"][0], support.row_index())
    return bool(replay[rl_tokens].all())


def loss_scales(
    micro_batches: list[TensorMicroBatch], gradient_divide_factor: float, group: dist.ProcessGroup
) -> LossScales:
    """Normalize each loss component c (rl, ce, ref_kl) by its global count N_c.

    A per-rank count would upweight ranks with fewer tokens after FSDP averaging. rl counts the sum of its weights
    (tokens for 0/1 weights, groups under prompt-mean aggregation). ce and ref_kl count member tokens, so a
    fractional ce weight scales the loss. One batched collective keeps every rank issuing the same op. To keep
    activation gradients independent of batch size, the division is deferred until after backward, once per step:
      max_scale = max_c N_c
      backward on sum_c L_c * max_scale / N_c, with L_c the component's loss sum
      grads *= grad_scale = gradient_divide_factor / max_scale (also undoes FSDP's divide)
    """
    local_rl = 0.0
    local_ce = 0
    local_ref_kl = 0
    for micro_batch in micro_batches:
        mask = micro_batch["loss_mask"]
        rl_weights = micro_batch["rl_weights"]
        local_rl += float(rl_weights[mask].sum(dtype=torch.float64)) if rl_weights is not None else int(mask.sum())
        if micro_batch["ce_weights"] is not None:
            local_ce += int((micro_batch["ce_weights"] != 0).sum())
        if micro_batch["ref_kl_weights"] is not None:
            local_ref_kl += int((micro_batch["ref_kl_weights"] != 0).sum())
    global_counts = torch.tensor([local_rl, local_ce, local_ref_kl], dtype=torch.float64, device="cuda")
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=group)
    rl, ce, ref_kl = (count if count > 0 else 1 for count in global_counts.tolist())
    max_scale = max(rl, ce, ref_kl)
    return LossScales(rl / max_scale, ce / max_scale, ref_kl / max_scale, max_scale, gradient_divide_factor / max_scale)


def token_env_ids(micro_batch: TensorMicroBatch) -> tuple[Tensor, list[str]]:
    """Each token's env index, spreading the sequences' env names over their lengths, and the env names."""
    env_names = sorted(set(micro_batch["env_names"]))
    env_ids = np.repeat(
        np.array([env_names.index(name) for name in micro_batch["env_names"]], dtype=np.int64),
        micro_batch["sequence_lengths"],
    )
    return torch.from_numpy(env_ids).to("cuda"), env_names


def replay_routed_experts(micro_batch: TensorMicroBatch, enable_router_replay: bool) -> Tensor | None:
    """The routed experts to replay, on the host in their wire dtype (uint8/uint16) until the CP shard."""
    if not enable_router_replay:
        return None
    if micro_batch["routed_experts"] is None:
        raise ValueError(
            "You must set `enable_return_routed_experts=True` in the inference config or pass `--enable-return-routed-experts` to vLLM server to use router replay."
        )
    return micro_batch["routed_experts"]


def micro_batch_support(micro_batch: TensorMicroBatch, score_centering: bool) -> TokenSupport | None:
    """The sampler support on the GPU, when score centering or replayed rows need it."""
    if micro_batch["support_offsets"] is None:
        return None
    if not score_centering and not bool(micro_batch["support_replayed"].any()):
        return None
    return TokenSupport(
        ids=micro_batch["support_ids"],
        offsets=micro_batch["support_offsets"],
        replayed=micro_batch["support_replayed"],
        logprobs=micro_batch["support_logprobs"],
        sparse_backward=micro_batch["sparse_head_backward"],
    ).to("cuda")


def score_centering_errors(
    micro_batch: TensorMicroBatch,
    input_ids: Tensor,
    rl_mask: Tensor,
    support: TokenSupport | None,
    vocab_size: int,
) -> Tensor | None:
    """Check the sampler evidence that score centering needs. Returns the failed checks as a GPU tensor, which the
    caller checks once after the last micro batch, so the host never waits on it."""
    if support is None or support.logprobs is None:
        # Raises unless the micro batch has no RL tokens
        validate_score_centering_data(micro_batch, vocab_size=vocab_size)
        return None
    return score_centering_data_errors(
        input_ids[0],
        rl_mask,
        support.ids,
        support.logprobs,
        support.row_index(),
        support.replayed,
        vocab_size=vocab_size,
    )


def model_inputs(
    micro_batch: TensorMicroBatch,
    input_ids: Tensor,
    routed_experts: Tensor | None,
    support: TokenSupport | None,
    cp: ContextParallel | None,
) -> ModelInputs:
    """Build the forward inputs and shard them for this CP rank. Takes the micro batch's image inputs."""
    position_ids = micro_batch["position_ids"].to("cuda")
    mm_kwargs = micro_batch.pop("mm_kwargs")
    if mm_kwargs is not None:
        mm_kwargs = {key: value.to("cuda") for key, value in mm_kwargs.items()}
    mm_token_type_ids = micro_batch.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to("cuda")
    labels = shift_tensor_left(input_ids)
    temperatures = micro_batch["temperatures"].to("cuda")
    label_support = support.shift_left() if support is not None else None
    cp_shard_entries = None

    if cp is not None:
        validate_cp_seq_lens(micro_batch["seq_lens"], input_ids.shape[-1])
        # The VLM shards image inputs itself, after building their positions.
        defer_vlm_cp_to_model = mm_kwargs is not None
        if not defer_vlm_cp_to_model:
            input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp.rank, cp.size)
            if routed_experts is not None:
                routed_experts = shard_for_cp(routed_experts, cp_rank=cp.rank, cp_world_size=cp.size)
        labels = shard_for_cp(labels, cp_rank=cp.rank, cp_world_size=cp.size)
        temperatures = shard_for_cp(temperatures, cp_rank=cp.rank, cp_world_size=cp.size)
        if label_support is not None:
            # The LM head consumes the support after any deferred VLM sharding, so it must follow the label shard
            # rather than the input shard.
            shard_len = label_support.num_rows // cp.size
            cp_shard_entries = label_support.offsets[::shard_len].diff().tolist()
            label_support = label_support.rows(cp.rank * shard_len, (cp.rank + 1) * shard_len)

    if label_support is not None:
        assert label_support.num_rows == labels.numel(), (
            f"support rows {label_support.num_rows} are not aligned with labels shape {tuple(labels.shape)}"
        )
    return ModelInputs(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        temperatures=temperatures,
        seq_lens=micro_batch["seq_lens"].to("cuda"),
        seq_lens_are_pre_shard=cp is not None,
        routed_experts=routed_experts.to("cuda").int() if routed_experts is not None else None,
        mm_kwargs=mm_kwargs,
        mm_token_type_ids=mm_token_type_ids,
        label_support=label_support,
        cp_shard_entries=cp_shard_entries,
    )


def forward_logprobs(
    model: nn.Module,
    inputs: ModelInputs,
    ac_offloading: bool,
    cp: ContextParallel | None,
    gather_support: bool,
    vocab_size: int,
) -> dict[str, Tensor]:
    """Run the forward with per-token temperatures. Returns the logprobs and entropies of the whole sequence,
    shifted to align with the input tokens, and the support logprobs if the support is gathered."""
    with maybe_activation_offloading(ac_offloading, model):
        out = forward(
            model,
            inputs.input_ids,
            inputs.position_ids,
            labels=inputs.labels,
            temperature=inputs.temperatures,
            mm_kwargs=inputs.mm_kwargs,
            mm_token_type_ids=inputs.mm_token_type_ids,
            seq_lens=inputs.seq_lens,
            seq_lens_are_pre_shard=inputs.seq_lens_are_pre_shard,
            routed_experts=inputs.routed_experts,
            support=inputs.label_support,
        )

    # The fused LM head returns logprobs; the vanilla one returns logits
    if out.get("logprobs") is None:
        assert out.get("logits") is not None, "Logits must be provided to compute logprobs"
        scaled_logits = out["logits"] / inputs.temperatures.unsqueeze(-1)
        if inputs.label_support is not None:
            out["logprobs"], out["support_logprobs"] = selective_log_softmax_with_support(
                scaled_logits, inputs.labels, inputs.label_support
            )
        else:
            out["logprobs"] = selective_log_softmax(scaled_logits, inputs.labels)
        out["entropy"] = compute_entropy(scaled_logits)

    if cp is not None:
        out["logprobs"] = gather_for_cp(out["logprobs"], cp.group)
        out["entropy"] = gather_for_cp_wo_grad(out["entropy"], cp.size, cp.group)
        if gather_support and inputs.label_support is not None:
            out["support_logprobs"] = gather_ragged_for_cp(out["support_logprobs"], inputs.cp_shard_entries, cp.group)

    # The first token is masked out of the loss, so its pad value only guards against a mask bug
    out["logprobs"] = shift_tensor_right(out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item())
    out["entropy"] = shift_tensor_right(out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item())
    return out


def micro_batch_loss(
    out: dict[str, Tensor],
    inputs: LossInputs,
    sequence_lengths: list[int],
    scales: LossScales,
    rl_loss_fn: Callable,
    support: TokenSupport | None,
    trainer_support_logprobs: Tensor | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    def split(tensor: Tensor | None) -> tuple[Tensor, ...] | None:
        return tensor.squeeze().split(sequence_lengths) if tensor is not None else None

    return compute_loss(
        trainer_logprobs=split(out["logprobs"]),
        inference_logprobs=split(inputs.inference_logprobs),
        ref_logprobs=split(inputs.ref_logprobs),
        advantages=split(inputs.advantages),
        loss_mask=split(inputs.loss_mask),
        rl_weights=split(inputs.rl_weights),
        ce_weights=split(inputs.ce_weights),
        ref_kl_weights=split(inputs.ref_kl_weights),
        rl_loss_fn=rl_loss_fn,
        rl_scale=scales.rl,
        ce_scale=scales.ce,
        ref_kl_scale=scales.ref_kl,
        support=support,
        trainer_support_logprobs=trainer_support_logprobs,
    )


def backward(
    loss: Tensor,
    model: nn.Module,
    gradient_manager: GradientOffloadManager | None,
    parallel_dims: ParallelDims,
    final_backward: bool,
) -> None:
    if parallel_dims.dp_replicate_enabled and gradient_manager is None:
        # HSDP keeps the reduce-scattered gradients across micro batches and all-reduces them over the replicas
        # once, after the last backward
        model.set_requires_all_reduce(final_backward)
    begin_backward(gradient_manager, final_backward=final_backward)
    loss.backward()
    finish_backward(gradient_manager)


def record_stats(
    tensor_stats: TensorStats,
    out: dict[str, Tensor],
    inputs: LossInputs,
    env_ids: Tensor,
    env_names: list[str],
    staleness: Tensor | None,
    staleness_bin_edges: Tensor,
) -> Tensor:
    """Add the micro batch's entropy and mismatch KL to the step's stats. Returns the mismatch KL."""
    tensor_stats.add("entropy", out["entropy"], inputs.loss_mask, env_ids, env_names)
    mismatch_mask = inputs.mismatch_mask
    with torch.no_grad():
        mismatch_kl = _mismatch_kl_from_log_ratio(out["logprobs"] - inputs.inference_logprobs)
    tensor_stats.add("mismatch_kl", mismatch_kl, mismatch_mask, env_ids, env_names)
    if staleness is not None:
        staleness = staleness.to("cuda")
        tensor_stats.add(
            "mismatch_kl_by_staleness",
            mismatch_kl,
            mismatch_mask & (staleness >= 0),
            torch.bucketize(staleness, staleness_bin_edges, right=True),
            STALENESS_BINS,
            fold_all=False,
        )
    return mismatch_kl


class ForwardBackward:
    """The training step's forward and backward over one step's micro batches."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        parallel_dims: ParallelDims,
        rl_loss_fn: Callable,
        gradient_manager: GradientOffloadManager | None,
        workers: WorkerPool,
        materialize_images: Callable[[MMRefs], dict[str, Tensor]] | None,
        annotation_writer: AnnotationWriter | None,
        record_function: Callable[[str], AbstractContextManager],
    ):
        self.model = model
        self.config = config
        self.parallel_dims = parallel_dims
        self.rl_loss_fn = rl_loss_fn
        self.gradient_manager = gradient_manager
        self.workers = workers
        self.materialize_images = materialize_images
        self.annotation_writer = annotation_writer
        self.record_function = record_function
        self.vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
        self.score_centering = config.loss.type == "icepop" and config.loss.score_centering
        # A custom loss may put gradient on any token, so only the built-in losses take the sparse head backward
        self.sparse_backward_allowed = config.loss.type != "custom"
        self.is_moe = is_tt_moe_model(model)
        self.dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
        self.cp = None
        if parallel_dims.cp_enabled:
            cp_mesh = parallel_dims.world_mesh["cp"]
            self.cp = ContextParallel(cp_mesh.get_local_rank(), parallel_dims.cp, cp_mesh.get_group())
        self.staleness_bin_edges = torch.tensor(STALENESS_BIN_EDGES, dtype=torch.int16, device="cuda")

    def prepare(self, micro_batch: TensorMicroBatch) -> TensorMicroBatch:
        micro_batch = prepare_micro_batch(micro_batch, self.materialize_images)
        micro_batch["sparse_head_backward"] = self.sparse_backward_allowed and sparse_head_backward(micro_batch)
        return micro_batch

    def __call__(
        self, micro_batches: list[TensorMicroBatch], *, overlap_optimizer: bool = True, annotate: bool = True
    ) -> ForwardBackwardResult:
        """Run forward and backward on the micro batches and accumulate their gradients. With
        ``overlap_optimizer``, an offloaded optimizer steps during the last backward. With ``annotate``, the
        annotation writer exports the trainer's per-token streams."""
        logger = get_logger()
        perf_counter = get_perf_counter(self.model)
        scales = loss_scales(micro_batches, self.parallel_dims.fsdp_gradient_divide_factor, self.dp_cp_group)
        prepare_gradient_offload(self.gradient_manager, scales.grad_scale, overlap_optimizer=overlap_optimizer)
        cp_size = self.parallel_dims.cp

        logger.debug(f"Starting forward and backward pass (batch_size={len(micro_batches)})")
        tensor_stats = TensorStats()  # Accumulates metrics across micro-batches and ranks for logging
        routing_stats = []
        num_tokens = 0
        flops = 0
        errors = []
        for micro_step, micro_batch in enumerate(self.workers(self.prepare, micro_batches)):
            num_tokens += micro_batch["input_ids"].shape[1]
            flops += perf_counter.flop_counter(micro_batch["seq_lens"].tolist())
            input_ids = micro_batch["input_ids"].to("cuda")
            loss_inputs = LossInputs.from_micro_batch(micro_batch)
            env_ids, env_names = token_env_ids(micro_batch)
            routed_experts = replay_routed_experts(micro_batch, self.config.enable_router_replay)
            support = micro_batch_support(micro_batch, self.score_centering)
            if self.score_centering:
                error = score_centering_errors(micro_batch, input_ids, loss_inputs.rl_mask, support, self.vocab_size)
                if error is not None:
                    errors.append(error)

            inputs = model_inputs(micro_batch, input_ids, routed_experts, support, self.cp)
            with self.record_function("forward"):
                out = forward_logprobs(
                    self.model, inputs, self.config.model.ac_offloading, self.cp, self.score_centering, self.vocab_size
                )
            trainer_support_logprobs = None
            if self.score_centering and support is not None and support.logprobs is not None:
                # The label support has no row for the first token, so pad its entries to align with `support`
                first_row = int(support.offsets[1])
                support_logprobs = out["support_logprobs"]
                trainer_support_logprobs = torch.cat([support_logprobs.new_zeros(first_row), support_logprobs])

            loss, loss_tensors = micro_batch_loss(
                out,
                loss_inputs,
                micro_batch["sequence_lengths"],
                scales,
                self.rl_loss_fn,
                support,
                trainer_support_logprobs,
            )
            with self.record_function("backward"):
                backward(
                    loss,
                    self.model,
                    self.gradient_manager,
                    self.parallel_dims,
                    final_backward=micro_step == len(micro_batches) - 1,
                )
            del inputs

            # Every CP rank computes the whole sequence's loss, and the global normalizers count each
            # sequence once per CP rank (gradients still sum correctly), so undo that for the logged value.
            # The backward loss is max_scale times the normalized loss.
            tensor_stats.add("loss", loss / scales.max_scale * cp_size)
            mismatch_kl = record_stats(
                tensor_stats,
                out,
                loss_inputs,
                env_ids,
                env_names,
                micro_batch["staleness"],
                self.staleness_bin_edges,
            )
            for key, loss_tensor in loss_tensors.items():
                tensor_stats.add(key, loss_tensor)
            if annotate and self.annotation_writer is not None:
                self.annotation_writer.export(micro_batch, out)
            if self.is_moe:
                routing_stats.append(take_moe_routing_stats(self.model))

            # Debug log with *local, micro step* stats, evaluated only when debug logging is on
            logger.opt(lazy=True).debug(
                "Micro Step {} | Loss {:.4f} | Entropy {:.4f} | Mismatch KL {:.4f}",
                lambda: f"{micro_step + 1}/{len(micro_batches)}",
                lambda: loss.item() / scales.max_scale * cp_size,
                lambda: out["entropy"][loss_inputs.loss_mask].mean().item(),
                lambda: mismatch_kl[loss_inputs.mismatch_mask].mean().item(),
            )

        if annotate and self.annotation_writer is not None:
            self.annotation_writer.flush()
        # Invalid sampler evidence fails the step before the optimizer applies its gradients
        raise_score_centering_data_errors(errors)
        return ForwardBackwardResult(tensor_stats, routing_stats, num_tokens, flops, scales.grad_scale)
