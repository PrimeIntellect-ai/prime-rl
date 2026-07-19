"""CRADIO v4-H vision tower for Nemotron-VL.

Self-contained reimplementation of the frozen RADIO encoder as configured in
Nemotron-3-Nano-Omni (`vit_huge_patch16_224`, `model_norm=False`,
`register_multiple=10`, CPE patch generator). Numerics follow NVIDIA's RADIO
hub code (nvidia/C-RADIOv4-H) and timm's VisionTransformer exactly, because the
tower is frozen and its outputs must match what the projector was aligned to:

- Patch embed: flatten 16x16 patches to (3*16*16) and project with a bias-free
  linear (`ViTPatchLinear`), NOT a conv.
- Position embeddings are stored at 128x128 grid (2048px / patch 16) and
  bilinearly interpolated to the input grid. RADIO's training-mode branch
  applies *random* crop/scale augmentation to the pos-embed grid; this module
  always uses the deterministic eval branch since the tower is frozen.
- 10 prefix tokens (cls + registers) are prepended before the blocks and
  dropped from the output features.
- Blocks are standard timm pre-LN ViT blocks (LayerNorm eps 1e-6, GELU, qkv
  bias, no layer scale). The final norm is Identity (`model_norm=False`).
- Input conditioning (mean/std normalization) happens in the image processor
  (Omni calls `radio_model.make_preprocessor_external()`), so
  `input_conditioner` here only carries the checkpoint buffers and does not
  transform the input.

Module tree mirrors the HF checkpoint keys under the `vision_model.` prefix,
e.g. `radio_model.model.blocks.0.attn.qkv.weight`, so state dict conversion is
a pure prefix swap.
"""

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.nemotron_vl.configuration_nemotron_vl import RadioViTConfig


class RadioInputConditioner(nn.Module):
    """Holds the checkpoint's normalization buffers; forward is a no-op (external preprocessing)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(3, 1, 1))
        self.register_buffer("norm_std", torch.ones(3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RadioClsToken(nn.Module):
    def __init__(self, embed_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.zeros(num_tokens, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)


class RadioPatchGenerator(nn.Module):
    """RADIO's ViTPatchGenerator: linear patch embed + interpolated abs pos + prefix tokens."""

    def __init__(self, config: RadioViTConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        self.num_rows = config.max_img_size // config.patch_size
        self.num_cols = config.max_img_size // config.patch_size

        self.embedder = nn.Linear(3 * config.patch_size**2, config.hidden_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_rows * self.num_cols, config.hidden_size))
        self.cls_token = RadioClsToken(config.hidden_size, config.num_cls_tokens)

    def _im_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        # 'b c (py yy) (px xx) -> b (py px) (c yy xx)', same as RADIO's Im2Patches
        b, c, h, w = x.shape
        p = self.patch_size
        py, px = h // p, w // p
        x = x.reshape(b, c, py, p, px, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, py * px, c * p * p)

    def _get_pos_embed(self, rows: int, cols: int) -> torch.Tensor:
        """Deterministic (eval-branch) CPE position embedding interpolation."""
        if (self.num_rows, self.num_cols) == (rows, cols):
            return self.pos_embed
        pos = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)
        max_dim = max(rows, cols)
        pos = F.interpolate(pos.float(), size=(max_dim, max_dim), mode="bilinear", align_corners=False)
        pos = pos[..., :rows, :cols]
        if pos.shape[-2:] != (rows, cols):
            pos = F.interpolate(pos, size=(rows, cols), mode="bilinear", align_corners=False)
        return pos.to(self.pos_embed.dtype).flatten(2).permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        patches = self.embedder(self._im_to_patches(x))
        patches = patches + self._get_pos_embed(rows, cols)
        return self.cls_token(patches)


class RadioAttention(nn.Module):
    def __init__(self, config: RadioViTConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class RadioMlp(nn.Module):
    def __init__(self, config: RadioViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RadioBlock(nn.Module):
    def __init__(self, config: RadioViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = RadioAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = RadioMlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RadioViT(nn.Module):
    """timm VisionTransformer forward_features path with CPE patch generator and Identity norm."""

    def __init__(self, config: RadioViTConfig):
        super().__init__()
        self.patch_generator = RadioPatchGenerator(config)
        self.blocks = nn.ModuleList([RadioBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_generator(x)
        for block in self.blocks:
            x = block(x)
        return x


class RadioModelWrapper(nn.Module):
    """Mirrors RADIOModel from the hub code: input_conditioner + inner ViT (`model`)."""

    def __init__(self, config: RadioViTConfig):
        super().__init__()
        self.input_conditioner = RadioInputConditioner()
        self.model = RadioViT(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.input_conditioner(x))


class RadioVisionModel(nn.Module):
    """The `visual` component of Nemotron-VL: returns patch features with prefix tokens dropped."""

    def __init__(self, config: RadioViTConfig):
        super().__init__()
        self.config = config
        self.radio_model = RadioModelWrapper(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.radio_model.model.patch_generator.embedder.weight.dtype

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(num_tiles, 3, H, W) -> (num_tiles, H/patch * W/patch, hidden_size)."""
        tokens = self.radio_model(pixel_values)
        return tokens[:, self.config.num_cls_tokens :]


__all__ = ["RadioVisionModel"]
