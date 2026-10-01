import torch
from torch import nn

from prime_rl.trainer.models.nemotron_h_omni.configuration_nemotron_h_omni import RadioConfig


class RadioAttention(nn.Module):
    def __init__(self, config: RadioConfig) -> None:
        super().__init__()
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)


class RadioMLP(nn.Module):
    def __init__(self, config: RadioConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)


class RadioBlock(nn.Module):
    def __init__(self, config: RadioConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = RadioAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = RadioMLP(config)


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
        raise NotImplementedError("Nemotron Omni image execution is not supported yet")
