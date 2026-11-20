from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import partial
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.lm_head import TokenSupport, support_replay_mask
from prime_rl.trainer.optim.offload import GradientOffloadManager
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.data import (
    MMRefs,
    TensorMicroBatch,
    prepare_micro_batch,
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
    STALENESS_BINS,
    TensorStats,
    begin_backward,
    finish_backward,
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


class StepContext(NamedTuple):
    """What the training step needs besides its micro batches, built once per run."""

    model: nn.Module
    config: TrainerConfig
    parallel_dims: ParallelDims
    rl_loss_fn: Callable
    gradient_manager: GradientOffloadManager | None
    # Prepares a micro batch for the step on a data worker
    prepare: Callable[[TensorMicroBatch], TensorMicroBatch]
    record_function: Callable[[str], AbstractContextManager]
    cp: ContextParallel | None
    vocab_size: int
    score_centering: bool


def prepare_for_step(
    micro_batch: TensorMicroBatch,
    materialize_images: Callable[[MMRefs], dict[str, Tensor]] | None,
    sparse_backward_allowed: bool,
) -> TensorMicroBatch:
    micro_batch = prepare_micro_batch(micro_batch, materialize_images)
    micro_batch["sparse_head_backward"] = sparse_backward_allowed and sparse_head_backward(micro_batch)
    return micro_batch


def setup_step_context(
    model: nn.Module,
    config: TrainerConfig,
    parallel_dims: ParallelDims,
    rl_loss_fn: Callable,
    gradient_manager: GradientOffloadManager | None,
    materialize_images: Callable[[MMRefs], dict[str, Tensor]] | None,
    record_function: Callable[[str], AbstractContextManager],
) -> StepContext:
    cp = None
    if parallel_dims.cp_enabled:
        cp_mesh = parallel_dims.world_mesh["cp"]
        cp = ContextParallel(cp_mesh.get_local_rank(), parallel_dims.cp, cp_mesh.get_group())
    return StepContext(
        model=model,
        config=config,
        parallel_dims=parallel_dims,
        rl_loss_fn=rl_loss_fn,
        gradient_manager=gradient_manager,
        # A custom loss may put gradient on any token, so only the built-in losses take the sparse head backward
        prepare=partial(
            prepare_for_step,
            materialize_images=materialize_images,
            sparse_backward_allowed=config.loss.type != "custom",
        ),
        record_function=record_function,
        cp=cp,
        vocab_size=getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size,
        score_centering=config.loss.type == "icepop" and config.loss.score_centering,
    )


class MicroBatchOutput(NamedTuple):
    loss: Tensor
    loss_tensors: dict[str, Tensor]
    out: dict[str, Tensor]
    loss_inputs: LossInputs
    env_ids: Tensor
    env_names: list[str]
    # Failed sampler evidence checks, to raise once after the last micro batch
    score_centering_errors: Tensor | None


def forward_backward_micro_batch(
    ctx: StepContext, micro_batch: TensorMicroBatch, scales: LossScales, final_backward: bool
) -> MicroBatchOutput:
    """Run forward and backward on one prepared micro batch and accumulate its gradients. The caller sets up the
    step's gradient offload before the first micro batch."""
    input_ids = micro_batch["input_ids"].to("cuda")
    loss_inputs = LossInputs.from_micro_batch(micro_batch)
    env_ids, env_names = token_env_ids(micro_batch)
    routed_experts = replay_routed_experts(micro_batch, ctx.config.enable_router_replay)
    support = micro_batch_support(micro_batch, ctx.score_centering)
    errors = None
    if ctx.score_centering:
        errors = score_centering_errors(micro_batch, input_ids, loss_inputs.rl_mask, support, ctx.vocab_size)

    inputs = model_inputs(micro_batch, input_ids, routed_experts, support, ctx.cp)
    with ctx.record_function("forward"):
        out = forward_logprobs(
            ctx.model, inputs, ctx.config.model.ac_offloading, ctx.cp, ctx.score_centering, ctx.vocab_size
        )
    del inputs
    trainer_support_logprobs = None
    if ctx.score_centering and support is not None and support.logprobs is not None:
        # The label support has no row for the first token, so pad its entries to align with `support`
        first_row = int(support.offsets[1])
        support_logprobs = out["support_logprobs"]
        trainer_support_logprobs = torch.cat([support_logprobs.new_zeros(first_row), support_logprobs])

    loss, loss_tensors = micro_batch_loss(
        out,
        loss_inputs,
        micro_batch["sequence_lengths"],
        scales,
        ctx.rl_loss_fn,
        support,
        trainer_support_logprobs,
    )
    with ctx.record_function("backward"):
        backward(loss, ctx.model, ctx.gradient_manager, ctx.parallel_dims, final_backward)
    return MicroBatchOutput(loss, loss_tensors, out, loss_inputs, env_ids, env_names, errors)


def record_stats(
    tensor_stats: TensorStats,
    output: MicroBatchOutput,
    loss_scale: float,
    staleness: Tensor | None,
    staleness_bin_edges: Tensor,
) -> Tensor:
    """Add the micro batch's loss, entropy, mismatch KL and loss tensors to the step's stats. ``loss_scale`` turns
    the backward loss into the logged one. Returns the mismatch KL."""
    inputs = output.loss_inputs
    tensor_stats.add("loss", output.loss * loss_scale)
    tensor_stats.add("entropy", output.out["entropy"], inputs.loss_mask, output.env_ids, output.env_names)
    mismatch_mask = inputs.mismatch_mask
    with torch.no_grad():
        mismatch_kl = _mismatch_kl_from_log_ratio(output.out["logprobs"] - inputs.inference_logprobs)
    tensor_stats.add("mismatch_kl", mismatch_kl, mismatch_mask, output.env_ids, output.env_names)
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
    for key, loss_tensor in output.loss_tensors.items():
        tensor_stats.add(key, loss_tensor)
    return mismatch_kl
