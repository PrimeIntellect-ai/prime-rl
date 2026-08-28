import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from prime_rl.trainer.models.layers.attn import (
    flash_attn_3_varlen_func,
    flash_attn_4_varlen_func,
    flash_attn_varlen_func,
)
from prime_rl.trainer.models.layers.rotary_emb import rotate_half
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", torch.empty(dim // 2), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        positions = torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.inv_freq.device)
        self.inv_freq.copy_(1.0 / (self.theta ** (positions / self.dim)))

    def forward(self, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(positions, self.inv_freq)


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(pixel_values.to(self.proj.weight.dtype)).view(pixel_values.shape[0], -1)


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states), approximate="tanh"))


class Qwen3_5VisionAttention(nn.Module):
    FLASH_ATTENTION_FUNCTIONS = {
        "flash_attention_2": flash_attn_varlen_func,
        "flash_attention_3": flash_attn_3_varlen_func,
        "flash_attention_4": flash_attn_4_varlen_func,
    }

    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        attention_implementation = getattr(config, "_attn_implementation", None) or "flash_attention_3"
        self.attention_implementation = attention_implementation
        self.flash_attention = (
            None if attention_implementation == "sdpa" else self.FLASH_ATTENTION_FUNCTIONS[attention_implementation]
        )
        if attention_implementation == "flash_attention_4":
            self.flash_attention = torch.compiler.disable(self.flash_attention)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states).view(sequence_length, 3, self.num_heads, self.head_dim).unbind(dim=1)
        )
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1).float()
        sin = sin.unsqueeze(1).float()
        query_dtype = query.dtype
        query = ((query.float() * cos) + (rotate_half(query.float()) * sin)).to(query_dtype)
        key = ((key.float() * cos) + (rotate_half(key.float()) * sin)).to(query_dtype)

        if self.attention_implementation == "sdpa":
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outputs = []
            for query_chunk, key_chunk, value_chunk in zip(
                query.split(lengths),
                key.split(lengths),
                value.split(lengths),
            ):
                output = F.scaled_dot_product_attention(
                    query_chunk.transpose(0, 1).unsqueeze(0),
                    key_chunk.transpose(0, 1).unsqueeze(0),
                    value_chunk.transpose(0, 1).unsqueeze(0),
                    is_causal=False,
                )
                outputs.append(output.squeeze(0).transpose(0, 1))
            attention_output = torch.cat(outputs, dim=0)
        elif self.attention_implementation == "flash_attention_4":
            attention_output = self.flash_attention(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                causal=False,
            )
        else:
            max_sequence_length = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            attention_output = self.flash_attention(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                max_sequence_length,
                max_sequence_length,
                causal=False,
            )
        if isinstance(attention_output, tuple):
            attention_output = attention_output[0]
        return self.proj(attention_output.reshape(sequence_length, -1))


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        merged_hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.merged_hidden_size = merged_hidden_size
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_hidden_size, merged_hidden_size)
        self.linear_fc2 = nn.Linear(merged_hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states).view(-1, self.merged_hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states)))


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = math.isqrt(config.num_position_embeddings)
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
        self.blocks = nn.ModuleList(Qwen3_5VisionBlock(config) for _ in range(config.depth))
        self.merger = Qwen3_5VisionPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def rotary_embeddings(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grids = grid_thw.tolist()
        frequency_table = self.rotary_pos_emb(max(max(height, width) for _, height, width in grids))
        coordinates = []
        merge_size = self.spatial_merge_size
        for frames, height, width in grids:
            block_rows = torch.arange(height // merge_size, device=frequency_table.device)
            block_columns = torch.arange(width // merge_size, device=frequency_table.device)
            intra_rows = torch.arange(merge_size, device=frequency_table.device)
            intra_columns = torch.arange(merge_size, device=frequency_table.device)
            rows = block_rows[:, None, None, None] * merge_size + intra_rows[None, None, :, None]
            columns = block_columns[None, :, None, None] * merge_size + intra_columns[None, None, None, :]
            rows = rows.expand(height // merge_size, width // merge_size, merge_size, merge_size).reshape(-1)
            columns = columns.expand(height // merge_size, width // merge_size, merge_size, merge_size).reshape(-1)
            coordinates.append(torch.stack((rows, columns), dim=-1).repeat(frames, 1))
        return frequency_table[torch.cat(coordinates)].flatten(1)

    def interpolated_position_embeddings(self, grid_thw: torch.Tensor) -> torch.Tensor:
        embeddings = []
        merge_size = self.spatial_merge_size
        for frames, height, width in grid_thw.tolist():
            row_positions = torch.linspace(
                0,
                self.num_grid_per_side - 1,
                height,
                device=self.pos_embed.weight.device,
            )
            column_positions = torch.linspace(
                0,
                self.num_grid_per_side - 1,
                width,
                device=self.pos_embed.weight.device,
            )
            row_floor = row_positions.floor().long()
            column_floor = column_positions.floor().long()
            row_ceil = (row_floor + 1).clamp_max(self.num_grid_per_side - 1)
            column_ceil = (column_floor + 1).clamp_max(self.num_grid_per_side - 1)
            row_fraction = row_positions - row_floor
            column_fraction = column_positions - column_floor

            indices = torch.stack(
                [
                    (row_floor[:, None] * self.num_grid_per_side + column_floor[None, :]).flatten(),
                    (row_floor[:, None] * self.num_grid_per_side + column_ceil[None, :]).flatten(),
                    (row_ceil[:, None] * self.num_grid_per_side + column_floor[None, :]).flatten(),
                    (row_ceil[:, None] * self.num_grid_per_side + column_ceil[None, :]).flatten(),
                ]
            )
            weights = torch.stack(
                [
                    ((1 - row_fraction)[:, None] * (1 - column_fraction)[None, :]).flatten(),
                    ((1 - row_fraction)[:, None] * column_fraction[None, :]).flatten(),
                    (row_fraction[:, None] * (1 - column_fraction)[None, :]).flatten(),
                    (row_fraction[:, None] * column_fraction[None, :]).flatten(),
                ]
            ).to(self.pos_embed.weight.dtype)
            position_embedding = (self.pos_embed(indices) * weights[..., None]).sum(dim=0).repeat(frames, 1)
            position_embedding = (
                position_embedding.view(
                    frames,
                    height // merge_size,
                    merge_size,
                    width // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            embeddings.append(position_embedding)
        return torch.cat(embeddings)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> BaseModelOutputWithPooling:
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self.interpolated_position_embeddings(grid_thw)
        rotary = self.rotary_embeddings(grid_thw)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        sequence_lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = F.pad(sequence_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
        )


__all__ = ["Qwen3_5VisionModel"]
