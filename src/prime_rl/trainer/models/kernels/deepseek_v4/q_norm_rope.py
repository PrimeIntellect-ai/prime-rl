"""DeepSeek-V4's query RMSNorm and RoPE in fp32 with a single rounding, as vLLM's fused prefill kernel does."""

import torch

from prime_rl.trainer.models.kernels.deepseek_v4.interleaved_rope import mla_rope_apply_raw_, mla_rope_unapply_raw
from prime_rl.trainer.models.layers.norms import get_quack_rmsnorm


def _rmsnorm_fwd(rows: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Unweighted RMSNorm of `(n, d)` rows to fp32, with the per-row fp32 `rstd` its backward needs."""
    if get_quack_rmsnorm() is not None:
        from quack.rmsnorm import rmsnorm_fwd

        normed, _, rstd = rmsnorm_fwd(rows, out_dtype=torch.float32, eps=eps, store_rstd=True)
        return normed, rstd
    rows = rows.float()
    rstd = torch.rsqrt(rows.square().mean(-1) + eps)
    return rows * rstd[:, None], rstd


def _rmsnorm_bwd(rows: torch.Tensor, grad: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
    if get_quack_rmsnorm() is not None:
        from quack.rmsnorm import rmsnorm_bwd

        return rmsnorm_bwd(rows, None, grad, rstd)[0]
    normed = rows.float() * rstd[:, None]
    grad = grad.float()
    return ((grad - normed * (grad * normed).mean(-1, keepdim=True)) * rstd[:, None]).to(rows.dtype)


class _QNormRoPE(torch.autograd.Function):
    """The fp32 normalized query lives only inside this node, so autograd never builds an fp32 gradient."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float) -> torch.Tensor:
        rows = q.reshape(-1, q.shape[-1])
        normed, rstd = _rmsnorm_fwd(rows, eps)
        normed = normed.view(q.shape)
        mla_rope_apply_raw_(normed, cos, sin, False)
        ctx.save_for_backward(rows, cos, sin, rstd)
        return normed.to(q.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        rows, cos, sin, rstd = ctx.saved_tensors
        # Rotates `grad` in place: unsafe if autograd hands this same tensor to another consumer.
        grad = grad.contiguous()
        mla_rope_unapply_raw(grad, cos, sin, False)
        return _rmsnorm_bwd(rows, grad.view(rows.shape), rstd).view(grad.shape), None, None, None


def q_norm_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float) -> torch.Tensor:
    """Unweighted RMSNorm over `head_dim`, then interleaved RoPE, both in fp32, rounded once to `q`'s dtype.

    `q` is `(t, h, d)` or `(1, t, h, d)` and contiguous; `cos` and `sin` are as in `apply_interleaved_rope_`.
    """
    return _QNormRoPE.apply(q, cos, sin, eps)
