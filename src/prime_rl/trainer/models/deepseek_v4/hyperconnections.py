import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


class DeepseekV4UnweightedRMSNorm(nn.Module):
    """RMS normalization without a learnable gain, computed in fp32."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV4HyperConnection(nn.Module):
    """Manifold-constrained hyper-connection (mHC) around one sublayer.

    The residual is `hc_mult` parallel streams shaped `(B, S, hc_mult, hidden_size)`.
    A single projection of the normalized, flattened streams produces three gates:

    - `pre`: weights that collapse the streams into the single sequence fed to the
      sublayer. Returned already applied, as `collapsed`.
    - `post`: weights in `[0, 2]` that broadcast the sublayer output back over the
      streams. Returned for the caller to apply.
    - `comb`: an `hc_mult x hc_mult` matrix that remixes the streams. It is projected
      onto the doubly-stochastic manifold by Sinkhorn-Knopp (alternating row and column
      normalization), which is what makes signal propagation non-expansive across depth.

    The projection and the Sinkhorn iterations run in fp32; only `collapsed` is cast
    back to the input dtype.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # One scale per gate: `pre`, `post`, `comb`.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)


class DeepseekV4HyperHead(nn.Module):
    """Final collapse of the `hc_mult` residual streams, before the model's last norm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.hc_fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)


__all__ = [
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
    "DeepseekV4UnweightedRMSNorm",
]
