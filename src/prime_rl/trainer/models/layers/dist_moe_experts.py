from __future__ import annotations

import dataclasses
import importlib.util
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from prime_rl.trainer.models.layers.activations import Silu
from prime_rl.trainer.models.layers.expert_compute import reserve_sms_for_comm, to_local
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    import dist_moe

    from prime_rl.trainer.models.layers.moe import GroupedExperts


_CONTEXTS: dict[int, dist_moe.Context] = {}


def _bf16_forward_args(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    context_key: int,
) -> tuple:
    from dist_moe._postprocess import _registered_rmsnorm_args

    context = _CONTEXTS[context_key]
    activation_buffer = context.activation_buffer
    return (
        x,
        topk_idx,
        topk_weights,
        gate_up_proj,
        down_proj,
        *_registered_rmsnorm_args(None),
        activation_buffer.activation_slot_id_1,
        activation_buffer._num_moe_layers_in_selected_slot,
        False,
        context.context_id,
    )


@torch.library.custom_op("prime_rl::dist_moe_forward", mutates_args=())
def dist_moe_forward_op(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    context_key: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    from dist_moe.api import _validate_inputs

    _validate_inputs(x, topk_idx, topk_weights, gate_up_proj, down_proj, _CONTEXTS[context_key])
    return torch.ops.dist_moe.bf16_forward.default(
        *_bf16_forward_args(x, topk_idx, topk_weights, gate_up_proj, down_proj, context_key)
    )


@dist_moe_forward_op.register_fake
def _dist_moe_forward_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    context_key: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    from dist_moe.api import _fake_registered_bf16_forward

    return _fake_registered_bf16_forward(
        *_bf16_forward_args(x, topk_idx, topk_weights, gate_up_proj, down_proj, context_key)
    )


@torch.library.custom_op("prime_rl::dist_moe_backward", mutates_args=())
def dist_moe_backward_op(
    grad_out: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    state: list[torch.Tensor],
    context_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dist_moe._postprocess import _registered_rmsnorm_args

    (
        _,
        rmsnorm_enabled,
        rmsnorm_eps,
        rmsnorm_norm_output_dtype,
        rmsnorm_output_dtype,
        rmsnorm_require_bitwise,
        _,
        rmsnorm_use_kahan,
        rmsnorm_recompute_rstd,
    ) = _registered_rmsnorm_args(None)
    return torch.ops.dist_moe.bf16_backward.default(
        grad_out.contiguous(),
        gate_up_proj,
        down_proj,
        topk_idx,
        topk_weights,
        state,
        rmsnorm_enabled,
        rmsnorm_eps,
        rmsnorm_norm_output_dtype,
        rmsnorm_output_dtype,
        rmsnorm_require_bitwise,
        rmsnorm_use_kahan,
        rmsnorm_recompute_rstd,
        gate_up_proj.dtype,
        _CONTEXTS[context_key].context_id,
    )


@dist_moe_backward_op.register_fake
def _dist_moe_backward_fake(
    grad_out: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    state: list[torch.Tensor],
    context_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(grad_out),
        torch.empty_like(topk_weights),
        torch.empty_like(gate_up_proj),
        torch.empty_like(down_proj),
    )


def _dist_moe_setup_context(ctx, inputs, output) -> None:
    _, topk_idx, topk_weights, gate_up_proj, down_proj, context_key = inputs
    _, state = output
    ctx.context_key = context_key
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*state)
    ctx.save_for_backward(gate_up_proj, down_proj, topk_idx, topk_weights, *state)


def _dist_moe_backward(ctx, grad_out: torch.Tensor, _state_grads: list[torch.Tensor | None]):
    gate_up_proj, down_proj, topk_idx, topk_weights, *state = ctx.saved_tensors
    dx, dtopk, dw1, dw2 = torch.ops.prime_rl.dist_moe_backward(
        grad_out, gate_up_proj, down_proj, topk_idx, topk_weights, state, ctx.context_key
    )
    return dx, None, dtopk, dw1, dw2, None


dist_moe_forward_op.register_autograd(_dist_moe_backward, setup_context=_dist_moe_setup_context)


@torch.library.custom_op("prime_rl::dist_moe_eval_forward", mutates_args=())
def dist_moe_eval_forward_op(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    context_key: int,
) -> torch.Tensor:
    import dist_moe

    with torch.no_grad():
        return dist_moe.routed_experts(x, topk_idx, topk_weights, gate_up_proj, down_proj, _CONTEXTS[context_key])


@dist_moe_eval_forward_op.register_fake
def _dist_moe_eval_forward_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    context_key: int,
) -> torch.Tensor:
    return torch.empty_like(x)


def validate_dist_moe_experts(experts: GroupedExperts) -> None:
    if experts.gate_proj is None and experts.gate_up_proj is None:
        raise ValueError("dist_moe requires gated experts (SwiGLU gate+up), got non-gated experts.")
    if experts.gate_up_proj is None:
        raise ValueError("dist_moe requires fused gate/up projections. Enable the gate_up fusion.")
    if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
        raise ValueError("dist_moe does not support expert biases.")
    if experts.activation is not Silu:
        raise ValueError("dist_moe requires the standard SwiGLU (`silu`) expert activation.")
    hidden = experts.down_proj.shape[1]
    if hidden % 64:
        raise ValueError(f"dist_moe requires the hidden size ({hidden}) to be a multiple of 64.")


class DistMoEExpertCompute:
    """One instance and one dist_moe context serve every MoE layer on the rank, and ``validate`` checks each layer
    against the context's shapes."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden: int,
        intermediate_hidden: int,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_moe_layers: int,
        scratch_capacity_factor: float,
        vmm_capacity_factor: float | None,
        activation_slot_capacity_factor: float | None,
        num_reserved_sms: int | None,
    ) -> None:
        capability = torch.cuda.get_device_capability()
        if capability[0] < 10:
            raise RuntimeError(
                f"dist_moe dispatch requires an SM100+ GPU, but this device is SM{capability[0]}{capability[1]}."
            )
        if importlib.util.find_spec("dist_moe") is None:
            raise RuntimeError("dist_moe dispatch requires the dist-moe package (install the `dist-moe` extra).")
        local_world_size = get_world().local_world_size
        nodes = {rank // local_world_size for rank in dist.get_process_group_ranks(group)}
        if len(nodes) > 1:
            raise ValueError(
                f"dist_moe dispatch requires the expert-parallel group to fit in one node (one NVLink domain), but "
                f"model.ep={group.size()} spans {len(nodes)} nodes of {local_world_size} GPUs."
            )
        import dist_moe

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.max_tokens_per_rank = max_tokens_per_rank
        self.group = group
        config = dist_moe.Config(
            max_num_local_input_tokens=max_tokens_per_rank,
            hidden_dim=hidden,
            intermediate_dim=intermediate_hidden,
            top_k=top_k,
            num_experts=num_experts,
            max_moe_layers_per_activation_slot=num_moe_layers,
            device_scratch_capacity_factor=scratch_capacity_factor,
            activation_slot_capacity_factor=activation_slot_capacity_factor,
            vmm=None
            if vmm_capacity_factor is None
            else dist_moe.VmmConfig(total_scratch_capacity_factor=vmm_capacity_factor),
            num_sms=None if num_reserved_sms is None else reserve_sms_for_comm(num_reserved_sms),
        )
        memory_plan = dist_moe.plan_memory(config, group.size(), device=torch.cuda.current_device())
        routing_state_bytes = 24 * num_experts * max_tokens_per_rank * num_moe_layers
        get_logger().info(
            f"{memory_plan.explain()}\n  routing state saved for backward, outside the plan: "
            f"{routing_state_bytes / 1024**3:.3f} GiB (24 B x {num_experts} experts x {max_tokens_per_rank} tokens "
            f"x {num_moe_layers} MoE layers)"
        )
        self.context = dist_moe.create_context(group=group, config=config)
        self.context_key = id(self.context)
        _CONTEXTS[self.context_key] = self.context
        self._pending_overflow_check = None

    def validate(self, experts: GroupedExperts) -> None:
        validate_dist_moe_experts(experts)
        shape = (experts.num_experts, experts.down_proj.shape[1], experts.hidden_dim)
        expected = (self.num_experts, self.hidden, self.intermediate_hidden)
        if shape != expected:
            raise ValueError(
                "The dist_moe context is sized for (num_experts, hidden, intermediate_hidden)="
                f"{expected}; got experts with {shape}."
            )

    def validate_top_k(self, top_k: int) -> None:
        if top_k != self.top_k:
            raise ValueError(f"The dist_moe context is sized for top_k={self.top_k}; got a router with top_k={top_k}.")

    def destroy(self) -> None:
        _CONTEXTS.pop(self.context_key).close()

    def begin_overflow_check(self) -> None:
        self._pending_overflow_check = self.context.activation_buffer.begin_check_overflow()

    def finish_overflow_check(self) -> dict[str, float]:
        """Raises if the activation buffer overflowed (corrupted numerics), else returns the run's peak scratch
        usage in bytes, maxed over the EP group."""
        scratch_info = self.context.activation_buffer.finish_check_overflow(self._pending_overflow_check)
        self._pending_overflow_check = None
        peaks = {
            name: value
            for name, value in dataclasses.asdict(scratch_info).items()
            if value is not None and name != "scratch_min_free"
        }
        values = torch.tensor(list(peaks.values()), dtype=torch.float64, device="cuda")
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=self.group)
        return {f"dist_moe/{name}": value for name, value in zip(peaks, values.tolist(), strict=True)}

    def __call__(
        self,
        experts: GroupedExperts,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        if num_tokens != self.max_tokens_per_rank:
            # TODO: pad variable-length micro batches to `max_tokens_per_rank` with zero rows, score 0 and expert id
            # -1 (never dispatched, and repeated -1 is allowed), then slice the output back to `num_tokens` rows.
            raise NotImplementedError(
                f"dist_moe requires pad_to_seq_len = True: every MoE call must have exactly "
                f"max_tokens_per_rank={self.max_tokens_per_rank} tokens per rank, got {num_tokens}."
            )
        args = (
            x.to(torch.bfloat16).contiguous(),
            selected_experts_indices.to(torch.int64).contiguous(),
            top_scores.to(torch.float32).contiguous(),
            to_local(experts.gate_up_proj),
            to_local(experts.down_proj),
            self.context_key,
        )
        if torch.is_grad_enabled():
            out, _ = torch.ops.prime_rl.dist_moe_forward(*args)
        else:
            out = torch.ops.prime_rl.dist_moe_eval_forward(*args)
        return out.to(x.dtype)
