import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        theta: float,
        partial_rotary_factor: float,
        mrope_section: tuple[int, int, int],
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim % 2:
            raise ValueError(f"Rotary dimension must be even, got {rotary_dim}")
        if sum(mrope_section) * 2 != rotary_dim:
            raise ValueError(f"mrope_section must contain {rotary_dim // 2} rotary pairs, got {mrope_section}")

        self.rotary_dim = rotary_dim
        self.theta = theta
        self.mrope_section = mrope_section
        self.register_buffer("inv_freq", torch.empty(rotary_dim // 2, device=device), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dimensions = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=self.inv_freq.device)
        self.inv_freq.copy_(1.0 / (self.theta ** (dimensions / self.rotary_dim)))

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                f"position_ids must have shape [batch, tokens] or [3, batch, tokens], got {position_ids.shape}"
            )

        positions = position_ids.to(device=hidden_states.device, dtype=torch.float32)
        frequencies = (self.inv_freq[None, None, :, None].float() * positions[:, :, None, :]).transpose(2, 3)
        interleaved = frequencies[0].clone()
        for dimension in (1, 2):
            indices = slice(dimension, self.mrope_section[dimension] * 3, 3)
            interleaved[..., indices] = frequencies[dimension, ..., indices]

        embeddings = torch.cat((interleaved, interleaved), dim=-1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(hidden_states.dtype)


__all__ = ["RotaryEmbedding"]
