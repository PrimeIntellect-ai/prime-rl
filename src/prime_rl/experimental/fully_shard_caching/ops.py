"""Grouped-expert ops that read cached prepared weights when their parameters are wrapped."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import torch

from prime_rl.experimental.fully_shard_caching.fp8_cast import (
    grouped_per_block_cast_to_fp8,
    grouped_per_block_cast_to_fp8_both_layouts,
    transposed_blockwise_fp8,
)
from prime_rl.experimental.fully_shard_caching.prepared_tensor import (
    ShardBlocking,
    UnshardedPreparedTensor,
    unsharded_prepared_or_none,
)
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_token_cast_to_fp8_triton,
    ue8m0_for_device,
    unpack_rows_triton,
)
from prime_rl.trainer.models.layers.activations import Activation
from prime_rl.trainer.models.layers.fp8_grouped_gemm import _compute_grad_weight, grouped_fp8_gemm
from prime_rl.trainer.models.layers.moe import broadcast_expert_bias


def blockwise_fp8_prepare(
    weight: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    """The forward GEMM consumes ``(experts, out_features, in_features)``, the dx GEMM its transpose."""
    use_ue8m0 = ue8m0_for_device(weight.device)
    if out is None:
        qdata, scales, qdata_t, scales_t = grouped_per_block_cast_to_fp8_both_layouts(weight, use_ue8m0)
        return {"qdata": qdata, "scales": scales, "qdata_t": qdata_t, "scales_t": scales_t}
    grouped_per_block_cast_to_fp8_both_layouts(
        weight, use_ue8m0, out=out["qdata"], sf=out["scales"], out_t=out["qdata_t"], sf_t=out["scales_t"]
    )
    return out


@torch.library.custom_op("fully_shard_caching::prepared_grouped_fp8_gemm", mutates_args=())
def _fp8_grouped_gemm_prepared_forward(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    offs: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    import deep_gemm

    (
        total_m,
        padded_total_m,
        grouped_layout,
        block_to_group,
        _,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = build_grouped_layout(offs, total_m=x.size(0))

    use_ue8m0 = ue8m0_for_device(x.device)
    x_fp8 = grouped_per_token_cast_to_fp8_triton(
        x,
        padded_total_m,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        use_ue8m0,
        GROUP_ALIGNMENT,
    )
    out_padded = torch.empty((padded_total_m, out_features), device=x.device, dtype=x.dtype)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        x_fp8,
        (qdata, scales),
        out_padded,
        grouped_layout,
        use_psum_layout=False,
    )
    return unpack_rows_triton(
        out_padded,
        total_m,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    )


@torch.library.custom_op("fully_shard_caching::prepared_grouped_fp8_gemm_backward", mutates_args=())
def _fp8_grouped_gemm_prepared_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    qdata_t: torch.Tensor,
    scales_t: torch.Tensor,
    offs: torch.Tensor,
    in_features: int,
    out_features: int,
    needs_grad_x: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm

    experts = qdata_t.size(0)
    weight_t_shape = torch.Size((experts, in_features, out_features))

    (
        _,
        padded_total_m,
        grouped_layout,
        block_to_group,
        ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = build_grouped_layout(offs, total_m=x.size(0))
    grad_output = grad_output.contiguous()

    grad_weight = grad_output.new_empty(weight_t_shape)
    if needs_grad_weight:
        grad_weight = _compute_grad_weight(
            x,
            grad_output,
            grad_weight,
            padded_total_m,
            block_to_group,
            ks_tensor,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
            ks_tensor.tolist(),
        )

    grad_x = torch.empty_like(x)
    if needs_grad_x:
        use_ue8m0 = ue8m0_for_device(grad_output.device)
        dy_fp8 = grouped_per_token_cast_to_fp8_triton(
            grad_output,
            padded_total_m,
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
            use_ue8m0,
            GROUP_ALIGNMENT,
        )
        grad_x_padded = torch.empty(
            (padded_total_m, weight_t_shape[1]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            dy_fp8,
            (qdata_t, scales_t),
            grad_x_padded,
            grouped_layout,
            use_psum_layout=False,
        )
        grad_x = unpack_rows_triton(
            grad_x_padded,
            x.size(0),
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
        )
    return grad_x, grad_weight


@_fp8_grouped_gemm_prepared_forward.register_fake
def _fp8_grouped_gemm_prepared_forward_fake(x, qdata, scales, offs, out_features):
    return x.new_empty((x.shape[0], out_features))


@_fp8_grouped_gemm_prepared_backward.register_fake
def _fp8_grouped_gemm_prepared_backward_fake(
    grad_output, x, qdata_t, scales_t, offs, in_features, out_features, needs_grad_x, needs_grad_weight
):
    return torch.empty_like(x), grad_output.new_empty((qdata_t.size(0), in_features, out_features))


class PreparedFp8GroupedGemm(torch.autograd.Function):
    """Backward keeps the weight wrapper, not its prepared tensors, so a refill between forward and
    backward stays invisible here."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: UnshardedPreparedTensor, offs: torch.Tensor):
        qdata = weight.prepared_qdata
        out = _fp8_grouped_gemm_prepared_forward(x, qdata, weight.prepared_scales, offs, qdata.size(1))
        ctx.save_for_backward(x, offs)
        ctx.weight = weight
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, offs = ctx.saved_tensors
        weight = ctx.weight
        needs_grad_x, needs_grad_weight, _ = ctx.needs_input_grad
        _, out_features, in_features = weight.prepared_qdata.shape
        grad_x, grad_weight_t = _fp8_grouped_gemm_prepared_backward(
            grad_output,
            x.detach(),
            weight.prepared_qdata_t,
            weight.prepared_scales_t,
            offs,
            in_features,
            out_features,
            needs_grad_x,
            needs_grad_weight,
        )
        return (
            grad_x if needs_grad_x else None,
            grad_weight_t.transpose(1, 2) if needs_grad_weight else None,
            None,
        )


@dataclass(frozen=True)
class Fp8GroupedExpertCompute:
    activation: type[Activation]

    def prepare(self, weight: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        return blockwise_fp8_prepare(weight, out=out)

    def _gemm(self, x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        prepared = unsharded_prepared_or_none(weight)
        if prepared is None:
            return grouped_fp8_gemm(x, weight.transpose(1, 2).bfloat16(), offs)
        return PreparedFp8GroupedGemm.apply(x, prepared, offs)

    def __call__(
        self,
        x: torch.Tensor,
        gate_proj: torch.Tensor | None,
        up_proj: torch.Tensor | None,
        gate_up_proj: torch.Tensor | None,
        down_proj: torch.Tensor,
        offs: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if gate_up_proj is None:
            gate = self._gemm(x, gate_proj, offs)
            up = self._gemm(x, up_proj, offs)
        else:
            gate, up = self._gemm(x, gate_up_proj, offs).chunk(2, dim=-1)
        return self._gemm(self.activation.apply(gate, up), down_proj, offs)


@dataclass(frozen=True)
class PreGatherFp8GroupedExpertCompute(Fp8GroupedExpertCompute):
    """Quantizes each shard before the all-gather, which is exact because the 128x128 tiles span the
    feature axes while FSDP shards the expert axis."""

    shard_blocking: ClassVar[ShardBlocking] = (1, GROUP_ALIGNMENT, GROUP_ALIGNMENT)


@dataclass(frozen=True)
class BothLayoutWireFp8GroupedExpertCompute(PreGatherFp8GroupedExpertCompute):
    """Sends both fp8 layouts over the wire, leaving nothing to derive once the shards meet."""

    def prepare_shard(
        self, shard: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        return blockwise_fp8_prepare(shard, out=out)

    def complete_gathered(
        self, wire: Mapping[str, torch.Tensor], *, out: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        return dict(wire) if out is None else out


@dataclass(frozen=True)
class OneLayoutWireFp8GroupedExpertCompute(PreGatherFp8GroupedExpertCompute):
    """Sends the forward layout over the wire, permuting the gathered bytes into the dx layout."""

    def prepare_shard(
        self, shard: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        use_ue8m0 = ue8m0_for_device(shard.device)
        if out is None:
            qdata, scales = grouped_per_block_cast_to_fp8(shard, use_ue8m0)
            return {"qdata": qdata, "scales": scales}
        grouped_per_block_cast_to_fp8(shard, use_ue8m0, out=out["qdata"], sf=out["scales"])
        return out

    def complete_gathered(
        self, wire: Mapping[str, torch.Tensor], *, out: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        qdata, scales = wire["qdata"], wire["scales"]
        if out is None:
            qdata_t, scales_t = transposed_blockwise_fp8(qdata, scales)
            return {"qdata": qdata, "scales": scales, "qdata_t": qdata_t, "scales_t": scales_t}
        transposed_blockwise_fp8(qdata, scales, out=out["qdata_t"], sf=out["scales_t"])
        return out


def row_scaled_prepare(weight: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Split a weight into a unit-row-norm transpose and the per-row absmax that restores it."""
    if out is None:
        row_absmax = weight.detach().abs().amax(dim=-1).clamp_min(1e-6)
        w_t = (weight / row_absmax.unsqueeze(-1)).transpose(1, 2).contiguous()
        return {"w_t": w_t, "row_absmax": row_absmax}
    experts, out_features, in_features = weight.shape
    assert out["row_absmax"].shape == (experts, out_features)
    assert out["w_t"].shape == (experts, in_features, out_features)
    torch.amax(weight.detach().abs(), dim=-1, out=out["row_absmax"])
    out["row_absmax"].clamp_min_(1e-6)
    torch.div(weight, out["row_absmax"].unsqueeze(-1), out=out["w_t"].transpose(1, 2))
    return out


class PreparedRowScaledWeight(torch.autograd.Function):
    """Expose a prepared ``w_t`` to autograd, routing its gradient back to the weight's layout."""

    @staticmethod
    def forward(ctx, weight: UnshardedPreparedTensor):
        ctx.weight = weight
        w_t = weight.prepared_w_t
        # An alias, so autograd never attaches a grad_fn to FSDP-owned storage.
        return w_t.view_as(w_t)

    @staticmethod
    def backward(ctx, grad_w_t: torch.Tensor):
        return grad_w_t.transpose(1, 2) / ctx.weight.prepared_row_absmax.unsqueeze(-1)


@dataclass(frozen=True)
class RowScaledGroupedExpertCompute:
    """A bf16 grouped-experts op whose prepare returns two entries, both consumed by the kernel.

    The absmax is treated as a constant, as a quantization scale would be, so the wrapped and
    unwrapped branches are the same function of the weight and agree bitwise.
    """

    activation: type[Activation]

    def prepare(self, weight: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        return row_scaled_prepare(weight, out=out)

    def _gemm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        offs: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        prepared = unsharded_prepared_or_none(weight)
        if prepared is None:
            row_absmax = weight.detach().abs().amax(dim=-1).clamp_min(1e-6)
            w_t = (weight / row_absmax.unsqueeze(-1)).transpose(1, 2).contiguous()
        else:
            row_absmax = prepared.prepared_row_absmax
            w_t = PreparedRowScaledWeight.apply(prepared)
        out = torch._grouped_mm(x, w_t, offs=offs)
        scale = broadcast_expert_bias(row_absmax, num_tokens_per_expert, out.shape[0])
        return out * scale

    def __call__(
        self,
        x: torch.Tensor,
        gate_proj: torch.Tensor | None,
        up_proj: torch.Tensor | None,
        gate_up_proj: torch.Tensor | None,
        down_proj: torch.Tensor,
        offs: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if gate_up_proj is None:
            gate = self._gemm(x, gate_proj, offs, num_tokens_per_expert)
            up = self._gemm(x, up_proj, offs, num_tokens_per_expert)
        else:
            gate, up = self._gemm(x, gate_up_proj, offs, num_tokens_per_expert).chunk(2, dim=-1)
        return self._gemm(self.activation.apply(gate, up), down_proj, offs, num_tokens_per_expert)
