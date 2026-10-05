import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4UnweightedRMSNorm
from prime_rl.trainer.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
from prime_rl.trainer.models.kernels.deepseek_v4 import dsv4_mhc


class _MixesMatmul(torch.autograd.Function):
    """`x @ w.T` for bf16 `x` and `w`, accumulated and returned in fp32.

    Every operand value is bf16, so the fp32 result is the fp32 product of the upcast operands
    without ever upcasting the (tokens, hc_mult * hidden) streams.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        return torch.mm(x, w.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, w = ctx.saved_tensors
        grad = grad.to(x.dtype)
        return torch.mm(grad, w), torch.mm(grad.t(), x, out_dtype=torch.float32).to(w.dtype)


def _fused_hyper_connection():
    """prime-kernels' fused mHC projection, gates and collapse when it is built for this GPU, else None."""
    import prime_kernels

    if "mhc_projection" in prime_kernels.KERNELS and prime_kernels.is_available("mhc_projection"):
        return prime_kernels.load("mhc_projection").hyper_connection
    return None


class DeepseekV41HyperConnection(nn.Module):
    """V4.1's mHC gates for one sublayer, computed from the streams entering it.

    The gates are V4's: `pre` collapses the streams, `post` in `[0, 2]` broadcasts the sublayer
    output back over them, and the Sinkhorn-projected `comb` remixes them. What changes is who uses
    `pre`: the gates computed at a sublayer's input collapse the streams for the *next* sublayer,
    so `forward` returns all three and leaves the collapse to the caller.
    """

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps, out_dtype=torch.float32)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # One scale per gate: `pre`, `post`, `comb`.
        self.scale = nn.Parameter(torch.empty(3))
        self.fused = _fused_hyper_connection()

    def forward(self, mhc_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = mhc_states.flatten(start_dim=2)
        # RMSNorm then projection == projection scaled by the row's rstd. Both operands hold bf16
        # values, so a bf16 GEMM accumulating into fp32 matches the fp32 product without
        # materializing the streams in fp32.
        rstd = torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + self.input_norm.eps)
        mixes = _MixesMatmul.apply(flat.reshape(-1, flat.shape[-1]), self.fn.to(flat.dtype))
        mixes = mixes.view(*flat.shape[:-1], -1) * rstd
        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = dsv4_mhc.fused_sinkhorn(comb_logits, self.hc_sinkhorn_iters, self.hc_eps)
        return pre, post, comb

    def gates_and_collapse(
        self, mhc_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`forward`'s gates plus the streams collapsed by `pre_mix`, the previous sublayer's `pre`."""
        if self.fused is not None:
            return self.fused(
                mhc_states,
                self.fn,
                self.scale,
                self.base,
                pre_mix,
                rms_eps=self.input_norm.eps,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
            )
        pre, post, comb = self(mhc_states)
        return pre, post, comb, collapse_streams(mhc_states, pre_mix)

    @staticmethod
    def update_states(
        post: torch.Tensor, comb: torch.Tensor, sublayer_out: torch.Tensor, mhc_states: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the sublayer output over the streams via `post` and remix them via `comb`."""
        dtype = mhc_states.dtype
        return dsv4_mhc.fused_post_bda(comb.to(dtype), mhc_states, post.to(dtype), sublayer_out)

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)


def collapse_streams(mhc_states: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """`(b, t, hc, d)` streams weighted by `(b, t, hc)` fp32 `pre` into one `(b, t, d)` sequence."""
    return (pre.unsqueeze(-1) * mhc_states.float()).sum(dim=2).to(mhc_states.dtype)


def identity_pre_mix(mhc_states: torch.Tensor) -> torch.Tensor:
    """The first attention's collapse: stream 0 alone (all streams are equal copies there)."""
    pre = mhc_states.new_zeros(*mhc_states.shape[:3], dtype=torch.float32)
    pre[..., 0] = 1.0
    return pre


__all__ = ["DeepseekV41HyperConnection", "collapse_streams", "identity_pre_mix"]
