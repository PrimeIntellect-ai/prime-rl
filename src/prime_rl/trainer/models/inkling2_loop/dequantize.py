"""Dequantize the packed routed experts of an Inkling 2 release checkpoint.

The release stores ``model.llm.layers.N.mlp.experts.{w13,w2}_weight`` in one of two packed formats and every other
tensor as plain BF16 / FP32:

- NVFP4 (``uint8`` weight): two E2M1 codes per byte along the input dim, the even element in the low nibble. Bit 3
  of a code is the sign and bits 0-2 index ``E2M1_VALUES``. Every 16 values share an E4M3 scale
  (``*_weight_scale``), and every expert a factor ``*_weight_amax / *_weight_outer_scale_denominator``.
- MXFP8 (``float8_e4m3fn`` weight): every 32 values share a power-of-two scale ``2 ** (scale - 127)``
  (``*_weight_scale``, ``uint8``).

The operation order below, ``(code * block_scale) * expert_factor`` in FP32 then one rounding, is the one the
transformers and SkyRL loaders use. The BF16 weights are bit-identical to SkyRL's. transformers'
``from_pretrained(dtype=bfloat16)`` casts the stored scales to BF16 before the same operations, which moves its
NVFP4 experts by up to 8e-3 relative.

The decoding of the two formats (its constants, the names of its functions and the first lines of this docstring),
``dequantize_expert_weight`` and the re-quantization check are adapted from the Inkling 2 reference written for
SkyRL (Apache-2.0):
https://github.com/erictang000/SkyRL/blob/faae9039fb0f4b6be27c02dc0045c1de63e3b888/skyrl/backends/skyrl_train/patches/megatron/inkling2/quant.py
The loop over groups of experts and the table of the two codes in a byte follow ``Inkling2LoopDequantizeExperts``
of transformers (Apache-2.0, Copyright 2025 The HuggingFace Inc. team; huggingface/transformers#51050, read at
commit 6173a2f).
"""

import re

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import StateDict

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
EXPERT_WEIGHT_RE = re.compile(r"\.mlp\.experts\.w(13|2)_weight$")
NVFP4_COMPANIONS = ("_scale", "_amax", "_outer_scale_denominator")
MXFP8_COMPANIONS = ("_scale",)
# Experts dequantized at once. Bounds the FP32 intermediates of one packed tensor.
EXPERTS_PER_CHUNK = 8


def _byte_to_e2m1_pair(device: torch.device) -> Tensor:
    """FP32 values ``[256, 2]`` of the two codes in every byte: low nibble first."""
    magnitude = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    code_values = torch.cat([magnitude, -magnitude])
    byte = torch.arange(256, device=device)
    return torch.stack([code_values[byte & 0xF], code_values[byte >> 4]], dim=-1)


def dequantize_nvfp4(
    packed: Tensor, scale: Tensor, amax: Tensor, outer_scale_denominator: Tensor, dtype: torch.dtype
) -> Tensor:
    """``packed [E, N, K / 2]`` (uint8), ``scale [E, N, K / 16]`` (E4M3), ``amax [E]`` -> ``[E, N, K]``."""
    num_experts, rows, packed_columns = packed.shape
    if scale.shape != (num_experts, rows, 2 * packed_columns // NVFP4_BLOCK):
        raise ValueError(f"NVFP4 scale shape {tuple(scale.shape)} does not match weight shape {tuple(packed.shape)}")
    byte_to_pair = _byte_to_e2m1_pair(packed.device)
    expert_factor = amax.float() / outer_scale_denominator.float()
    output = torch.empty(num_experts, rows, 2 * packed_columns, dtype=dtype, device=packed.device)
    for start in range(0, num_experts, EXPERTS_PER_CHUNK):
        experts = slice(start, start + EXPERTS_PER_CHUNK)
        block_scale = scale[experts].float()
        values = byte_to_pair[packed[experts].long()].view(*block_scale.shape, NVFP4_BLOCK) * block_scale[..., None]
        values = values * expert_factor[experts, None, None, None]
        output[experts] = values.flatten(-2).to(dtype)
    return output


def dequantize_mxfp8(weight: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """``weight [E, N, K]`` (E4M3), ``scale [E, N, K / 32]`` (UE8M0 exponents, uint8) -> ``[E, N, K]``."""
    num_experts, rows, columns = weight.shape
    if scale.shape != (num_experts, rows, columns // MXFP8_BLOCK):
        raise ValueError(f"MXFP8 scale shape {tuple(scale.shape)} does not match weight shape {tuple(weight.shape)}")
    output = torch.empty(num_experts, rows, columns, dtype=dtype, device=weight.device)
    for start in range(0, num_experts, EXPERTS_PER_CHUNK):
        experts = slice(start, start + EXPERTS_PER_CHUNK)
        block_scale = torch.exp2(scale[experts].float() - 127.0)
        values = weight[experts].float().view(*block_scale.shape, MXFP8_BLOCK) * block_scale[..., None]
        output[experts] = values.flatten(-2).to(dtype)
    return output


def nvfp4_requantization_mismatches(
    dequantized: Tensor, packed: Tensor, scale: Tensor, amax: Tensor, outer_scale_denominator: Tensor
) -> int:
    """Stored codes that re-quantizing ``dequantized`` with the stored scales does not give back.

    Zero means that rounding to the weight's dtype moved no value across a midpoint of the E2M1 grid, so the
    dequantized weight still determines every stored code. The stored codes and scales are decoded here as
    ``dequantize_nvfp4`` decodes them, so this does not check that decoding (nibble order, sign bit, block layout,
    scale formula). Blocks with a zero scale are skipped: all their codes decode to zero.
    """
    factor = (amax.float() / outer_scale_denominator.float()).view(-1, 1, 1)
    divisor = scale.float().repeat_interleave(NVFP4_BLOCK, dim=-1) * factor
    ratio = dequantized.float() / torch.where(divisor == 0, 1.0, divisor)
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    nearest = grid[torch.bucketize(ratio.abs(), (grid[:-1] + grid[1:]) / 2)].copysign(ratio)
    stored = _byte_to_e2m1_pair(packed.device)[packed.long()].flatten(-2)
    return int(((nearest != stored) & (divisor != 0)).sum())


def dequantize_expert_weight(name: str, tensors: StateDict, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """The expert weight ``name`` of ``tensors`` in ``dtype``, dequantized with its ``name + suffix`` companions."""
    weight = tensors[name]
    if weight.dtype == torch.uint8:
        return dequantize_nvfp4(
            weight,
            tensors[name + "_scale"],
            tensors[name + "_amax"],
            tensors[name + "_outer_scale_denominator"],
            dtype,
        )
    if weight.dtype == torch.float8_e4m3fn:
        return dequantize_mxfp8(weight, tensors[name + "_scale"], dtype)
    return weight.to(dtype)


def dequantize_state_dict_(state_dict: StateDict) -> None:
    """Replace every packed expert weight by its BF16 values in place, and drop the scale tensors.

    Must run on the checkpoint's own key names, before any renaming. A checkpoint whose experts are already BF16
    is left untouched.
    """
    for name in [key for key in state_dict if EXPERT_WEIGHT_RE.search(key)]:
        weight = state_dict[name]
        if weight.dtype == torch.uint8:
            companions = NVFP4_COMPANIONS
        elif weight.dtype == torch.float8_e4m3fn:
            companions = MXFP8_COMPANIONS
        else:
            continue
        missing = [name + suffix for suffix in companions if name + suffix not in state_dict]
        if missing:
            raise KeyError(f"Packed expert weight {name} is missing its scale tensors {missing}")
        state_dict[name] = dequantize_expert_weight(name, state_dict)
        for suffix in companions:
            del state_dict[name + suffix]


__all__ = [
    "EXPERT_WEIGHT_RE",
    "dequantize_expert_weight",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dequantize_state_dict_",
    "nvfp4_requantization_mismatches",
]
