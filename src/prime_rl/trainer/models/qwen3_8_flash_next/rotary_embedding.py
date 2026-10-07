import itertools

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


def get_vision_position_ids(
    *,
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device | None = None,
) -> torch.LongTensor:
    grid_t = int(grid_thw[0].item())
    grid_h = int(grid_thw[1].item()) // spatial_merge_size
    grid_w = int(grid_thw[2].item()) // spatial_merge_size

    temporal = torch.arange(grid_t, device=device) + start_position
    height = torch.arange(grid_h, device=device) + start_position
    width = torch.arange(grid_w, device=device) + start_position
    return torch.stack(
        [
            temporal.repeat_interleave(grid_h * grid_w),
            height.repeat_interleave(grid_w).repeat(grid_t),
            width.repeat(grid_h * grid_t),
        ],
        dim=0,
    )


def build_mrope_position_ids(
    *,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor | None,
    spatial_merge_size: int,
    seq_lens: torch.Tensor,
) -> torch.LongTensor:
    """``[3, 1, seq]`` MRoPE positions for a packed row: text advances all three axes together, each image spans
    its (temporal, height, width) grid from the current position, and positions restart at every sequence."""
    seq_lens = seq_lens.to(device=input_ids.device, dtype=torch.long)

    image_grids = iter(image_grid_thw) if image_grid_thw is not None else None
    position_ids = torch.empty(3, 1, input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
    offset = 0
    for sequence_length_tensor in seq_lens:
        sequence_length = int(sequence_length_tensor)
        token_types = mm_token_type_ids[0, offset : offset + sequence_length]
        current_position = 0
        sequence_positions = []

        for modality, indexed_group in itertools.groupby(enumerate(token_types.tolist()), lambda item: item[1]):
            group = list(indexed_group)
            group_length = group[-1][0] - group[0][0] + 1
            if modality == 0:
                text_positions = torch.arange(group_length, device=input_ids.device) + current_position
                sequence_positions.append(text_positions.unsqueeze(0).expand(3, -1))
                current_position += group_length
                continue
            if modality == 2:
                raise ValueError("Qwen3.8 Flash Next video MRoPE is not supported")
            if modality != 1:
                raise ValueError(f"Unsupported Qwen3.8 Flash Next multimodal token type: {modality}")
            if image_grids is None:
                raise ValueError("image_grid_thw is required for image tokens")

            remaining = group_length
            while remaining:
                grid = next(image_grids, None)
                if grid is None:
                    raise ValueError("Not enough image_grid_thw rows for the image tokens")
                image_positions = get_vision_position_ids(
                    start_position=current_position,
                    grid_thw=grid,
                    spatial_merge_size=spatial_merge_size,
                    device=input_ids.device,
                )
                if image_positions.shape[1] > remaining:
                    raise ValueError("Image token group length does not match image_grid_thw")
                sequence_positions.append(image_positions)
                current_position += max(int(grid[1]), int(grid[2])) // spatial_merge_size
                remaining -= image_positions.shape[1]

        positions = torch.cat(sequence_positions, dim=1)
        if positions.shape[1] != sequence_length:
            raise ValueError("Built MRoPE positions do not match the packed sequence length")
        position_ids[:, 0, offset : offset + sequence_length] = positions
        offset += sequence_length

    if image_grids is not None and next(image_grids, None) is not None:
        raise ValueError("image_grid_thw contains unused rows")
    return position_ids


__all__ = ["RotaryEmbedding", "apply_rotary_embedding", "build_mrope_position_ids", "get_vision_position_ids"]
