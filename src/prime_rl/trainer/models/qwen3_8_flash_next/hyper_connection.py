import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.kernels.qwen3_8_hyper_connection.ops import fused_hc_combine_mix, fused_hc_mix


class ExpandedRMSNorm(nn.Module):
    """RMS-normalize each residual stream with an independent affine weight."""

    def __init__(self, hidden_size: int, stream_count: int, eps: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stream_count = stream_count
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(stream_count * hidden_size))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        streams = hidden_states.float().unflatten(-1, (self.stream_count, self.hidden_size))
        variance = streams.square().mean(dim=-1, keepdim=True)
        normalized = streams * torch.rsqrt(variance + self.eps)
        return (normalized.flatten(-2) * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnection(nn.Module):
    """Learned mixing and residual injection across expanded residual streams."""

    def __init__(
        self,
        *,
        hidden_size: int,
        stream_count: int,
        low_rank: int,
        norm_eps: float,
        with_residual_injection: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stream_count = stream_count
        expanded_size = stream_count * hidden_size

        self.hc_norm = ExpandedRMSNorm(hidden_size, stream_count, norm_eps)
        self.input_mix_weight_down = nn.Linear(expanded_size, low_rank, bias=False)
        self.input_mix_weight_up = nn.Linear(low_rank, expanded_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(expanded_size, stream_count, bias=False) if with_residual_injection else None
        )

    def mix(
        self,
        expanded_states: torch.Tensor,
        pending_combine: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]:
        """The block input, and what ``combine`` needs: the streams and the injection weights. ``pending_combine`` is
        the previous sublayer's block output and injection weights, not yet combined: the mix applies it first and
        returns the combined streams."""
        weights = (
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            None if self.block_inject_weight is None else self.block_inject_weight.weight,
            self.stream_count,
            self.hc_norm.eps,
        )
        if expanded_states.is_cuda:
            if pending_combine is None:
                block_input, injection_weights = fused_hc_mix(expanded_states, *weights)
            else:
                expanded_states, block_input, injection_weights = fused_hc_combine_mix(
                    expanded_states, *pending_combine, *weights
                )
            return block_input, (expanded_states, injection_weights)

        if pending_combine is not None:
            block_output, pending_injection_weights = pending_combine
            expanded_states = self.combine(block_output, (expanded_states, pending_injection_weights))
        normalized_states = self.hc_norm(expanded_states)
        mixing_weights = F.silu(self.input_mix_weight_down(normalized_states) / self.stream_count)
        mixing_weights = torch.sigmoid(self.input_mix_weight_up(mixing_weights))
        mixing_weights = mixing_weights.unflatten(-1, (self.stream_count, self.hidden_size))
        normalized_streams = normalized_states.unflatten(-1, (self.stream_count, self.hidden_size))
        block_input = (mixing_weights * normalized_streams).mean(dim=-2)
        injection_weights = None
        if self.block_inject_weight is not None:
            injection_weights = 2.0 * torch.sigmoid(self.block_inject_weight(normalized_states) / self.stream_count)
        return block_input.to(expanded_states.dtype), (expanded_states, injection_weights)

    def forward(
        self,
        expanded_states: torch.Tensor,
        pending_combine: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]:
        return self.mix(expanded_states, pending_combine)

    def combine(
        self,
        block_output: torch.Tensor,
        residual_state: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        expanded_states, injection_weights = residual_state
        residual_streams = expanded_states.unflatten(-1, (self.stream_count, self.hidden_size))
        return (residual_streams + block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)


__all__ = ["ExpandedRMSNorm", "HyperConnection"]
