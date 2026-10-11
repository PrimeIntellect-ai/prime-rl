# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import torch
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


class ExpertCompute(Protocol):
    token_group_alignment: int

    def validate(self, experts: "GroupedExperts") -> None: ...

    def __call__(
        self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor: ...


def broadcast_expert_bias(
    bias: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    target_rows: int,
) -> torch.Tensor:
    repeats = num_tokens_per_expert.to(torch.int64)
    padding_rows = repeats.new_tensor(target_rows) - repeats.sum()
    return torch.repeat_interleave(
        torch.cat((bias, bias.new_zeros((1, bias.shape[1])))),
        torch.cat((repeats, padding_rows.unsqueeze(0))),
        dim=0,
        output_size=target_rows,
    )


class GroupedGemmExpertCompute:
    def __init__(
        self,
        gemm: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        token_group_alignment: int = 8,
    ) -> None:
        self.gemm = gemm
        self.token_group_alignment = token_group_alignment

    def validate(self, experts: "GroupedExperts") -> None:
        """The shared forward handles the experts' activation, biases, and weight layout."""

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2

        def to_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        x_bf16 = x.bfloat16()

        if experts.gate_up_proj is None:
            up_proj = to_local(experts.up_proj).transpose(-2, -1)
            up = self.gemm(x_bf16, up_proj.bfloat16(), offsets)

            gate = None
            if experts.gate_proj is not None:
                gate_proj = to_local(experts.gate_proj).transpose(-2, -1)
                gate = self.gemm(x_bf16, gate_proj.bfloat16(), offsets)
        else:
            gate_up_proj = to_local(experts.gate_up_proj).transpose(-2, -1)
            gate_up = self.gemm(x_bf16, gate_up_proj.bfloat16(), offsets)
            gate, up = gate_up.chunk(2, dim=-1)

        if experts.up_proj_bias is not None:
            up_proj_bias = to_local(experts.up_proj_bias)
            up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        if gate is not None and experts.gate_proj_bias is not None:
            gate_proj_bias = to_local(experts.gate_proj_bias)
            gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()

        hidden = experts.activation.apply(gate, up)
        down_proj = to_local(experts.down_proj).transpose(-2, -1)
        output = self.gemm(hidden, down_proj.bfloat16(), offsets)
        if experts.down_proj_bias is not None:
            down_proj_bias = to_local(experts.down_proj_bias)
            output = output + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, output.shape[0]).bfloat16()
        return output.type_as(x)


class BF16ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        super().__init__(torch._grouped_mm)


class DeepGemmFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        super().__init__(grouped_fp8_gemm)


class MXFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self, kernel: ModuleType, high_precision_wgrad: bool) -> None:
        super().__init__(
            partial(kernel.grouped_gemm, high_precision_wgrad=high_precision_wgrad),
            token_group_alignment=kernel.TOKEN_GROUP_ALIGNMENT,
        )


class FusedSwigluExpertCompute:
    """prime-kernels' `moe_experts`: Hopper grouped GEMMs with the clamped SwiGLU fused into them.

    With `fp8`, the GEMMs are DeepGEMM's blockwise FP8 (forward, dgrad and wgrad) with the
    quantization fused into the surrounding passes.
    """

    def __init__(self, kernel: ModuleType, fp8: bool = False) -> None:
        self.kernel = kernel
        self.fp8 = fp8
        self.token_group_alignment = kernel.TOKEN_GROUP_ALIGNMENT

    def validate(self, experts: "GroupedExperts") -> None:
        from prime_rl.trainer.models.deepseek_v4.moe import ClampedSwiglu

        if not isinstance(experts.activation, ClampedSwiglu):
            raise ValueError("The prime_kernels expert backend requires gated experts with a clamped SwiGLU.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("The prime_kernels expert backend requires bias-free experts.")
        num_experts, hidden_size, intermediate_size = experts.down_proj.shape
        reason = self.kernel.unsupported_shape_reason(hidden_size, intermediate_size, fp8=self.fp8)
        if reason is not None:
            raise ValueError(f"The prime_kernels expert backend cannot run these experts: {reason}")

    @staticmethod
    def _weights(experts: "GroupedExperts") -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        def to_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        if experts.gate_up_proj is None:
            gate_proj, up_proj = to_local(experts.gate_proj).bfloat16(), to_local(experts.up_proj).bfloat16()
        else:
            gate_proj, up_proj = to_local(experts.gate_up_proj).bfloat16(), None
        return gate_proj, up_proj, to_local(experts.down_proj).bfloat16().contiguous()

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_proj, up_proj, down_proj = self._weights(experts)
        output = self.kernel.moe_experts(
            x.bfloat16().contiguous(),
            gate_proj,
            up_proj,
            down_proj,
            num_tokens_per_expert,
            experts.activation.limit,
            fp8=self.fp8,
        )
        return output.type_as(x)

    @property
    def can_keep_activations(self) -> bool:
        return self.fp8

    @torch.no_grad()
    def forward_keeping(
        self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> list[torch.Tensor]:
        """The FP8 forward outside autograd: its output, then what its backward reads. The FP8
        weights are quantized once per optimizer step and shared by every micro-batch."""
        w13_q, w13_sf, w2_q, w2_sf, *transposed = _quantized_expert_weights(self, experts)
        kept = torch.ops.prime_kernels.moe_experts_fp8_forward_quantized(
            x.bfloat16().contiguous(), num_tokens_per_expert, experts.activation.limit, w13_q, w13_sf, w2_q, w2_sf
        )
        return [*kept, *transposed]

    def from_kept(
        self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor, kept: list[torch.Tensor]
    ) -> torch.Tensor:
        """The kept output again, differentiable in `x` and the weights through the kept tensors."""
        gate_proj, up_proj, down_proj = self._weights(experts)
        output, *saved = kept
        return _KeptFP8Experts.apply(
            x, gate_proj, up_proj, down_proj, output, num_tokens_per_expert, experts.activation.limit, experts, *saved
        ).type_as(x)


# Bumped after every optimizer step: FP8 expert weights quantized under an older generation are stale.
_weight_generation = 0
_quantized_weights: dict[int, tuple[int, list[torch.Tensor]]] = {}


def invalidate_quantized_expert_weights() -> None:
    """Call after the expert weights change (each optimizer step)."""
    global _weight_generation
    _weight_generation += 1
    _quantized_weights.clear()


def _quantized_expert_weights(compute: FusedSwigluExpertCompute, experts: "GroupedExperts") -> list[torch.Tensor]:
    cached = _quantized_weights.get(id(experts))
    if cached is None or cached[0] != _weight_generation:
        gate_proj, up_proj, down_proj = compute._weights(experts)
        assert all(w is None or w.untyped_storage().size() > 0 for w in (gate_proj, up_proj, down_proj)), (
            "the bf16 expert weights were freed after an earlier quantization and not all-gathered again"
        )
        quantized = torch.ops.prime_kernels.moe_experts_fp8_quantize_weights(gate_proj, up_proj, down_proj)
        cached = (_weight_generation, quantized)
        _quantized_weights[id(experts)] = cached
    return cached[1]


# Weight-gradient GEMMs held back by `defer_weight_grads`, while it is active.
_deferred_weight_grads: list[Callable[[], None]] | None = None


@contextmanager
def defer_weight_grads() -> Iterator[list[Callable[[], None]]]:
    """Within the block, kept FP8 experts whose weight gradients go straight into FSDP's fp32
    accumulators compute only their data gradient in backward and leave their weight-gradient
    GEMMs in the yielded list; the caller runs them (in order) before the block ends, e.g. after
    it has sent the data gradient on."""
    global _deferred_weight_grads
    assert _deferred_weight_grads is None, "defer_weight_grads does not nest"
    _deferred_weight_grads = []
    try:
        yield _deferred_weight_grads
        assert not _deferred_weight_grads, "deferred weight gradients were left unrun"
    finally:
        _deferred_weight_grads = None


def _fp32_grad_accumulator(fsdp_param) -> torch.Tensor:
    """The local fp32 buffer FSDP reduce-scatters as `fsdp_param`'s gradient, created at the first
    micro-batch of a step."""
    if fsdp_param.unsharded_accumulated_grad is None:
        fsdp_param.unsharded_accumulated_grad = torch.zeros_like(fsdp_param.unsharded_param, dtype=torch.float32)
    grad = fsdp_param.unsharded_accumulated_grad
    return grad.to_local() if isinstance(grad, DTensor) else grad


class _KeptFP8Experts(torch.autograd.Function):
    """The experts' kept forward output, differentiated by prime-kernels' FP8 backward. When FSDP
    manages the packed weights, their gradients go straight into FSDP's fp32 accumulators (the
    weight-gradient GEMMs accumulate there), so no bf16 gradient is materialized or upcast."""

    @staticmethod
    def forward(ctx, x, gate_proj, up_proj, down_proj, output, num_tokens_per_expert, limit, experts, *saved):
        ctx.save_for_backward(num_tokens_per_expert, *saved)
        ctx.limit, ctx.packed = limit, up_proj is None
        ctx.fsdp_params = getattr(experts, "fsdp_params", None)
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        num_tokens_per_expert, *saved = ctx.saved_tensors
        nones = [None] * (len(saved) + 4)
        if ctx.packed and ctx.fsdp_params is not None and _deferred_weight_grads is not None:
            x_t, x_t_sf = saved[1:3]
            dx, *operands = torch.ops.prime_kernels.moe_experts_fp8_backward_data(
                grad_output, num_tokens_per_expert, saved[0], *saved[3:], ctx.limit
            )
            dw13 = _fp32_grad_accumulator(ctx.fsdp_params["gate_up_proj"])
            dw2 = _fp32_grad_accumulator(ctx.fsdp_params["down_proj"])
            _deferred_weight_grads.append(
                partial(
                    torch.ops.prime_kernels.moe_experts_fp8_weight_grad_accumulate,
                    num_tokens_per_expert,
                    *operands,
                    x_t,
                    x_t_sf,
                    grad_output.shape[0],
                    dw13,
                    dw2,
                )
            )
            return dx, None, None, None, *nones
        if ctx.packed and ctx.fsdp_params is not None:
            dx = torch.ops.prime_kernels.moe_experts_fp8_backward_accumulate(
                grad_output,
                num_tokens_per_expert,
                *saved,
                ctx.limit,
                _fp32_grad_accumulator(ctx.fsdp_params["gate_up_proj"]),
                _fp32_grad_accumulator(ctx.fsdp_params["down_proj"]),
            )
            return dx, None, None, None, *nones
        dx, dw1, dw3, dw2 = torch.ops.prime_kernels.moe_experts_fp8_backward(
            grad_output, num_tokens_per_expert, *saved, ctx.limit, ctx.packed
        )
        return dx, dw1, None if ctx.packed else dw3, dw2, *nones
