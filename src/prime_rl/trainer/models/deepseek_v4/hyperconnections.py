import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.kernels.deepseek_v4 import dsv4_mhc
from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_mhc_pre import fused_mhc_pre
from prime_rl.trainer.models.layers import norms


class DeepseekV4UnweightedRMSNorm(norms.Norm):
    """RMS normalization without a learnable gain, computed in fp32."""

    def __init__(self, eps: float = 1e-6, out_dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = self.out_dtype if self.out_dtype is not None else x.dtype
        quack_rms = norms.get_quack_rmsnorm()
        if quack_rms is not None:
            return quack_rms(x, eps=self.eps, out_dtype=out_dtype)
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(out_dtype)


class DeepseekV4HyperConnection(nn.Module):
    """Manifold-constrained hyper-connection (mHC) around one sublayer.

    The residual, `mhc_states`, is `hc_mult` parallel streams shaped `(B, S, hc_mult, hidden_size)`.
    A single projection of the normalized, flattened streams produces three gates:

    - `pre`: weights that collapse the streams into the single sequence fed to the
      sublayer. Returned already applied, as `collapsed`.
    - `post`: weights in `[0, 2]` that broadcast the sublayer output back over the
      streams. Returned for the caller to apply.
    - `comb`: an `hc_mult x hc_mult` matrix that remixes the streams. It is projected
      onto the doubly-stochastic manifold by Sinkhorn-Knopp (alternating row and column
      normalization), which is what makes signal propagation non-expansive across depth.

    The norm, projection, gates and collapse are one fused op (`fused_mhc_pre`). The projection
    and the Sinkhorn iterations run in fp32; only `collapsed` is cast back to the input dtype.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # One scale per gate: `pre`, `post`, `comb`.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, mhc_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        scale = torch.cat([self.scale[0].expand(hc), self.scale[1].expand(hc), self.scale[2].expand(hc * hc)])
        collapsed, logits = fused_mhc_pre(mhc_states, self.fn, scale, self.base, self.norm_eps, self.hc_eps)
        post = 2 * torch.sigmoid(logits[..., hc : 2 * hc])
        comb_logits = logits[..., 2 * hc :].unflatten(-1, (hc, hc))
        comb = dsv4_mhc.fused_sinkhorn(comb_logits, self.hc_sinkhorn_iters, self.hc_eps)
        return post, comb, collapsed

    def update_states(
        self, post: torch.Tensor, comb: torch.Tensor, sublayer_out: torch.Tensor, mhc_states: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the sublayer output over the streams via `post` and remix them via `comb`.

        The write-back runs in fp32 and rounds once to the residual's dtype.
        """
        return dsv4_mhc.fused_post_bda(comb.float(), mhc_states, post.float(), sublayer_out)

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)


class DeepseekV4HyperHead(nn.Module):
    """Final collapse of the `hc_mult` residual streams, before the model's last norm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, mhc_states: torch.Tensor) -> torch.Tensor:
        scale = self.hc_scale.expand(self.hc_mult)
        collapsed, _ = fused_mhc_pre(mhc_states, self.hc_fn, scale, self.hc_base, self.norm_eps, self.eps)
        return collapsed

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.hc_fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)


__all__ = [
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
    "DeepseekV4UnweightedRMSNorm",
]
