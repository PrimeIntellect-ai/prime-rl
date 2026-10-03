import torch
from torch import nn
from torch.nn import functional as F

from prime_rl.trainer.models.nemotron_h_omni.configuration_nemotron_h_omni import RadioConfig


class RadioAttention(nn.Module):
    def __init__(self, config: RadioConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, hidden_size // self.num_heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        return self.proj(hidden_states.transpose(1, 2).reshape(batch_size, seq_len, hidden_size))


class RadioMLP(nn.Module):
    def __init__(self, config: RadioConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class RadioBlock(nn.Module):
    def __init__(self, config: RadioConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = RadioAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = RadioMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class RadioVisionModel(nn.Module):
    def __init__(self, config: RadioConfig, norm_mean: list[float], norm_std: list[float]) -> None:
        super().__init__()
        self.config = config
        patch_dim = config.num_channels * config.patch_size**2
        self.patch_embed = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.pos_embed = nn.Parameter(
            torch.empty(1, (config.max_resolution // config.patch_size) ** 2, config.hidden_size)
        )
        self.prefix_tokens = nn.Parameter(torch.empty(config.num_prefix_tokens, config.hidden_size))
        self.blocks = nn.ModuleList(RadioBlock(config) for _ in range(config.num_hidden_layers))
        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer("norm_std", torch.tensor(norm_std, dtype=torch.float32).view(-1, 1, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != self.config.num_channels:
            raise ValueError("pixel_values must have shape (num_images, num_channels, height, width)")
        height, width = pixel_values.shape[-2:]
        patch_size = self.config.patch_size
        if height % patch_size or width % patch_size or min(height, width) == 0:
            raise ValueError("Image dimensions must be positive multiples of the RADIO patch size")
        rows, cols = height // patch_size, width // patch_size
        patches = F.unfold(pixel_values.to(self.patch_embed.weight.dtype), patch_size, stride=patch_size).transpose(
            1, 2
        )
        hidden_states = self.patch_embed(patches)
        pos_size = self.config.max_resolution // patch_size
        position_embeddings = self.pos_embed.reshape(1, pos_size, pos_size, -1).permute(0, 3, 1, 2)
        position_embeddings = F.interpolate(
            position_embeddings.float(), size=(max(rows, cols), max(rows, cols)), mode="bilinear", align_corners=False
        ).to(hidden_states.dtype)
        position_embeddings = position_embeddings[:, :, :rows, :cols].flatten(2).transpose(1, 2)
        hidden_states = hidden_states + position_embeddings
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat((prefix_tokens.to(hidden_states.dtype), hidden_states), dim=1)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states[:, self.config.num_prefix_tokens :]
