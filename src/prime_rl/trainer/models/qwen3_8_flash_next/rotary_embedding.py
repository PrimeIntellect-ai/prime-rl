import torch
from torch import nn

from prime_rl.trainer.models.layers.rotary_emb import rotate_half


def apply_rotary_embedding(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]
    rotated = hidden_states[..., :rotary_dim]
    rotated = (rotated * cos) + (rotate_half(rotated) * sin)
    return torch.cat((rotated, hidden_states[..., rotary_dim:]), dim=-1)


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

        positions = position_ids.to(device=hidden_states.device, dtype=torch.float32)
        frequencies = (self.inv_freq[None, None, :, None].float() * positions[:, :, None, :]).transpose(2, 3)
        interleaved = frequencies[0].clone()
        for dimension in (1, 2):
            indices = slice(dimension, self.mrope_section[dimension] * 3, 3)
            interleaved[..., indices] = frequencies[dimension, ..., indices]

        embeddings = torch.cat((interleaved, interleaved), dim=-1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(hidden_states.dtype)


__all__ = ["RotaryEmbedding", "apply_rotary_embedding"]
