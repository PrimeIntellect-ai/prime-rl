import itertools

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class Qwen3_5RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5TextConfig, device=None) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = config.rope_parameters["rope_type"]
        self.rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        self.mrope_section = config.rope_parameters.get("mrope_section")
        if self.mrope_section is None:
            self.mrope_section = self.scaled_mrope_section(inv_freq.numel())
        if not config.rope_parameters["mrope_interleaved"]:
            raise ValueError("Qwen3.5 requires interleaved MRoPE")
        if sum(self.mrope_section) != inv_freq.numel():
            raise ValueError(f"mrope_section must contain {inv_freq.numel()} rotary pairs, got {self.mrope_section}")

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

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3_5TextConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        rope_parameters = config.rope_parameters
        rotary_dim = int(config.head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
        positions = torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float()
        return 1.0 / (rope_parameters["rope_theta"] ** (positions / rotary_dim)), 1.0

    def reset_parameters(self) -> None:
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.inv_freq.device)
        self.inv_freq.copy_(inv_freq)
        self.original_inv_freq.copy_(inv_freq)

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim != 3:
            raise ValueError(f"Qwen3.5 position_ids must be 2D or 3D, got {tuple(position_ids.shape)}")

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
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Packed Qwen3.5 input_ids must have shape (1, tokens), got {tuple(input_ids.shape)}")
    if mm_token_type_ids.shape != input_ids.shape:
        raise ValueError("mm_token_type_ids must have the same shape as input_ids")

    seq_lens = seq_lens.to(device=input_ids.device, dtype=torch.long)
    if seq_lens.ndim != 1 or bool((seq_lens <= 0).any()) or int(seq_lens.sum()) != input_ids.shape[1]:
        raise ValueError("seq_lens must contain positive lengths summing to the packed sequence length")

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
