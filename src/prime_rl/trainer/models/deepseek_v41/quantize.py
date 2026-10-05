"""Re-quantize bf16 training weights into the published DeepSeek-V4.1 checkpoint format.

vLLM serves V4.1 from its fp8 / fp4 checkpoint and reloads broadcast weights through the same
loader, so the trainer sends each weight in the format the checkpoint stores it in:

- dense projections: `float8_e4m3fn` with one power-of-two (`float8_e8m0fnu`) scale per 32x32 block;
- routed experts: e2m1 fp4, two values per `int8` byte (low nibble first), one power-of-two scale
  per 32 consecutive input channels.

Both are the exact inverses of `deepseek_v4.dequantize`; weights that were never re-trained
round-trip bit for bit.
"""

import re

import torch
from torch import Tensor

BLOCK = 32
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_MAX = 6.0
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# Published names (after the PrimeRL -> checkpoint rename) stored as fp8 / fp4.
FP8_WEIGHTS = re.compile(
    r"\.(attn\.(wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)|ffn\.shared_experts\.w[123]|engram\.wkv)\.weight$"
)
FP4_WEIGHTS = re.compile(r"\.ffn\.experts\.\d+\.w[123]\.weight$")


def _pow2_scale(amax: Tensor, qmax: float) -> Tensor:
    """The smallest power of two `s` with `amax / s <= qmax` (UE8M0 scales are pure exponents)."""
    return torch.exp2(torch.ceil(torch.log2((amax / qmax).clamp(min=2.0**-126))))


def quantize_fp8_block(weight: Tensor) -> tuple[Tensor, Tensor]:
    rows, cols = weight.shape
    padded = torch.nn.functional.pad(weight.float(), (0, -cols % BLOCK, 0, -rows % BLOCK))
    blocks = padded.unflatten(0, (-1, BLOCK)).unflatten(-1, (-1, BLOCK))  # (R, 32, C, 32)
    scale = _pow2_scale(blocks.abs().amax(dim=(1, 3)), FP8_MAX)  # (R, C)
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.flatten(2).flatten(0, 1)[:rows, :cols].contiguous()
    return q, scale.to(torch.float8_e8m0fnu)


def quantize_fp4(weight: Tensor) -> tuple[Tensor, Tensor]:
    rows, cols = weight.shape
    groups = weight.float().view(rows, cols // BLOCK, BLOCK)
    scale = _pow2_scale(groups.abs().amax(dim=-1), FP4_MAX)
    scaled = (groups / scale[..., None]).clamp(-FP4_MAX, FP4_MAX)
    grid = FP4_GRID.to(weight.device)
    codes = (scaled.abs()[..., None] - grid).abs().argmin(dim=-1)  # nearest magnitude
    codes = (codes | ((scaled < 0) & (codes > 0)).to(codes.dtype) << 3).to(torch.uint8).view(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.view(torch.int8), scale.to(torch.float8_e8m0fnu)


def quantize_state_dict_(state_dict: dict[str, Tensor]) -> None:
    """Quantize every published-fp8/fp4 weight in place, adding its `.scale` sibling."""
    for key in list(state_dict):
        if FP4_WEIGHTS.search(key):
            quantize = quantize_fp4
        elif FP8_WEIGHTS.search(key):
            quantize = quantize_fp8_block
        else:
            continue
        state_dict[key], state_dict[key.removesuffix(".weight") + ".scale"] = quantize(state_dict[key])


__all__ = ["quantize_fp4", "quantize_fp8_block", "quantize_state_dict_"]
