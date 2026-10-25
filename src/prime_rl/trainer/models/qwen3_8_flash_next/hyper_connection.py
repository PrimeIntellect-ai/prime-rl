import torch
import torch.nn.functional as F
from torch import nn


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

    def mix(self, expanded_states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        normalized_states = self.hc_norm(expanded_states)
        mixing_logits = self.input_mix_weight_down(normalized_states)
        mixing_weights = F.silu(mixing_logits.float() / self.stream_count).to(mixing_logits.dtype)
        mixing_weights = torch.sigmoid(self.input_mix_weight_up(mixing_weights).float())
        mixing_weights = mixing_weights.unflatten(-1, (self.stream_count, self.hidden_size))
        normalized_streams = normalized_states.float().unflatten(-1, (self.stream_count, self.hidden_size))
        block_input = torch.zeros_like(normalized_streams[..., 0, :], dtype=torch.float32)
        for stream in range(self.stream_count):
            block_input = block_input + mixing_weights[..., stream, :] * normalized_streams[..., stream, :]
        block_input = block_input / self.stream_count
        return block_input.to(expanded_states.dtype), (expanded_states, normalized_states)

    def forward(self, expanded_states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.mix(expanded_states)

    def combine(
        self,
        block_output: torch.Tensor,
        residual_state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        expanded_states, normalized_states = residual_state
        injection_logits = self.block_inject_weight(normalized_states)
        injection_weights = 2.0 * torch.sigmoid(injection_logits.float() / self.stream_count)
        residual_streams = expanded_states.float().unflatten(-1, (self.stream_count, self.hidden_size))
        output = residual_streams + block_output.float().unsqueeze(-2) * injection_weights.unsqueeze(-1)
        return output.flatten(-2).to(expanded_states.dtype)


__all__ = ["ExpandedRMSNorm", "HyperConnection"]
