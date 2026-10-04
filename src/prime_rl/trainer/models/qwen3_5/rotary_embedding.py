import itertools

import torch

from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5RopeParameters


class Qwen3_5RotaryEmbedding(RotaryEmbedding):
    """Interleaved multimodal RoPE: rotary pairs cycle through the (temporal, height, width) position axes."""

    def __init__(self, rope: Qwen3_5RopeParameters, head_dim: int, max_position_embeddings: int) -> None:
        super().__init__(rope, head_dim, max_position_embeddings)
        self.mrope_section = rope.mrope_section
        if self.mrope_section is None:
            self.mrope_section = self.scaled_mrope_section(self.inv_freq.numel())

    @staticmethod
    def scaled_mrope_section(num_rotary_pairs: int) -> list[int]:
        default_section = [11, 11, 10]
        total = sum(default_section)
        section = [num_rotary_pairs * size // total for size in default_section]
        remainder_order = sorted(
            range(len(section)),
            key=lambda index: num_rotary_pairs * default_section[index] % total,
            reverse=True,
        )
        for index in remainder_order[: num_rotary_pairs - sum(section)]:
            section[index] += 1
        return section

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        position_ids = position_ids.to(hidden_states.device)
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        positions = position_ids[:, :, None, :].float()
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (inv_freq @ positions).transpose(2, 3)
            interleaved = frequencies[0].clone()
            for dimension, offset in enumerate((1, 2), start=1):
                index = slice(offset, self.mrope_section[dimension] * 3, 3)
                interleaved[..., index] = frequencies[dimension, ..., index]
            embeddings = torch.cat((interleaved, interleaved), dim=-1)
            cos = embeddings.cos() * self.attention_scaling
            sin = embeddings.sin() * self.attention_scaling
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


def get_qwen3_5_vision_position_ids(
    *,
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    temporal_merge_size: int = 1,
    time_interval: int = 1,
    device: torch.device | None = None,
) -> torch.LongTensor:
    grid_t = int(grid_thw[0].item()) // temporal_merge_size
    grid_h = int(grid_thw[1].item()) // spatial_merge_size
    grid_w = int(grid_thw[2].item()) // spatial_merge_size

    temporal = torch.arange(grid_t, device=device) * time_interval + start_position
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


def build_qwen3_5_mrope_position_ids(
    *,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor | None,
    spatial_merge_size: int,
    seq_lens: torch.Tensor,
) -> torch.LongTensor:
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
                raise ValueError("Qwen3.5 video MRoPE is not supported")
            if modality != 1:
                raise ValueError(f"Unsupported Qwen3.5 multimodal token type: {modality}")
            if image_grids is None:
                raise ValueError("image_grid_thw is required for image tokens")

            remaining = group_length
            while remaining:
                grid = next(image_grids, None)
                if grid is None:
                    raise ValueError("Not enough image_grid_thw rows for the image tokens")
                image_positions = get_qwen3_5_vision_position_ids(
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


__all__ = ["Qwen3_5RotaryEmbedding", "build_qwen3_5_mrope_position_ids", "get_qwen3_5_vision_position_ids"]
