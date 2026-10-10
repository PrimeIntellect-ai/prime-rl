"""Single-stream gated residuals, a cheap alternative to V4.1's mHC (`residual_type` in the config).

Both keep the residual as `(batch, seq, 1, hidden)`, i.e. one mHC stream, so the decoder layer, the engram
and the pipeline stage I/O are unchanged, and expose the same `gates_and_collapse` / `update_states` pair as
`DeepseekV41HyperConnection`. The `pre_mix` handed between sublayers stays the identity collapse (stream 0),
so the model's final `collapse_streams` returns the stream as is.

- `gated`: `x + g * f(x)` with one gate per token, `g = 2 * sigmoid(scale * (w . rmsnorm(x)) + base)`. This is
  mHC's `post` gate at `hc_mult = 1`, where `pre` only rescales the next RMSNorm's input and `comb` is 1.
- `layerscale`: `x + lambda * f(x)` with one learned scale per channel (Touvron et al., 2021).

Both start as a plain pre-norm residual (`w = 0`, `base = 0`: `g = 1`; `lambda = 1`). The update, gate included,
is one fused Triton kernel per sublayer (`kernels/deepseek_v4/dsv41_gated_residual.py`).
"""

import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
from prime_rl.trainer.models.deepseek_v41.hyperconnections import DeepseekV41HyperConnection
from prime_rl.trainer.models.kernels.deepseek_v4 import dsv41_gated_residual


class DeepseekV41GatedResidual(nn.Module):
    """Per-token sigmoid gate on the sublayer output, computed from the stream entering the sublayer."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.fn = nn.Parameter(torch.empty(config.hidden_size))
        self.base = nn.Parameter(torch.empty(1))
        self.scale = nn.Parameter(torch.empty(1))

    def gates_and_collapse(
        self, mhc_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor]:
        """`(pre_mix, None, None, sublayer input, streams)`, in `DeepseekV41HyperConnection`'s order; the gate
        is computed by `update_states`' kernel from the same streams."""
        return pre_mix, None, None, mhc_states.squeeze(2), mhc_states

    def update_states(
        self,
        post: None,
        comb: None,
        sublayer_out: torch.Tensor,
        mhc_states: torch.Tensor,
        sublayer_out2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return dsv41_gated_residual.token_gated_residual(
            mhc_states, sublayer_out, sublayer_out2, self.fn, self.base, self.scale, self.eps
        )

    def init_weights(self, init_std: float) -> None:
        nn.init.zeros_(self.fn)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)


class DeepseekV41LayerScaleResidual(nn.Module):
    """Per-channel learned scale on the sublayer output."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(config.hidden_size))

    def gates_and_collapse(
        self, mhc_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor]:
        """`(pre_mix, None, None, sublayer input, streams)`, in `DeepseekV41HyperConnection`'s order."""
        return pre_mix, None, None, mhc_states.squeeze(2), mhc_states

    def update_states(
        self,
        post: None,
        comb: None,
        sublayer_out: torch.Tensor,
        mhc_states: torch.Tensor,
        sublayer_out2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return dsv41_gated_residual.channel_scaled_residual(mhc_states, sublayer_out, sublayer_out2, self.scale)

    def init_weights(self, init_std: float) -> None:
        nn.init.ones_(self.scale)


def residual_connection(config: DeepseekV41TextConfig) -> nn.Module:
    """The residual connection `config.residual_type` selects for one sublayer."""
    return {
        "mhc": DeepseekV41HyperConnection,
        "gated": DeepseekV41GatedResidual,
        "layerscale": DeepseekV41LayerScaleResidual,
    }[config.residual_type](config)


__all__ = ["DeepseekV41GatedResidual", "DeepseekV41LayerScaleResidual", "residual_connection"]
