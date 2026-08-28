"""Quantize a plain mini DeepSeek V4 checkpoint into the on-disk fp8/MXFP4 layout the real
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint uses, so a cheap local checkpoint can exercise
`dequantize_state_dict_` (`src/prime_rl/trainer/models/deepseek_v4/dequantize.py`) the same way
the real checkpoint does.

Quantizer math is lifted from
`tests/unit/train/models/test_deepseek_v4_dequantize_e2e.py`, which validates this same
block-quantize-then-dequantize round trip against HF's own `Fp8Dequantize`. That test's model is
tiny, so it defaults to a 16-wide block; `scripts/mini_moe.py --arch deepseek_v4` uses the real
checkpoint's actual dimensions, so this script uses the real block sizes instead: 128x128 for
dense fp8, 1x32 for packed MXFP4 (see `dequantize.py`'s module docstring).

See `dev/README.md` in this directory for the full generate -> quantize -> train loop.

Usage:
    uv run python scripts/mini_moe.py --arch deepseek_v4 --output-dir /tmp/deepseek-v4-mini
    uv run python src/prime_rl/trainer/models/deepseek_v4/dev/quantize_mini_checkpoint.py \
        --input-dir /tmp/deepseek-v4-mini --output-dir /tmp/deepseek-v4-mini-quantized
"""

import argparse
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_DENSE_FP8_SUFFIXES = {"wq_a", "wq_b", "wo_a", "wo_b", "wkv"}
_FP8_E4M3_MAX = 448.0
_MXFP4_MAX = 6.0
_MXFP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_DENSE_BLOCK = 128
_EXPERT_BLOCK = 32
_COPIED_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")


def _is_dense_fp8_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    parts = key.split(".")
    if len(parts) >= 3 and parts[-3] == "attn" and parts[-2] in _DENSE_FP8_SUFFIXES:
        return True
    return key.endswith("indexer.wq_b.weight") or ".shared_experts." in key


def _is_routed_expert_key(key: str) -> bool:
    return bool(re.search(r"\.experts\.\d+\.w[123]\.weight$", key))


def _block_scale_bytes(absmax: torch.Tensor, max_representable: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest power-of-two block scale, as `(float32 value, raw e8m0 byte)`."""
    exponent = torch.ceil(torch.log2(absmax.clamp(min=1e-12) / max_representable))
    byte = (exponent + 127.0).clamp(0, 254).to(torch.uint8)
    return torch.exp2(exponent), byte


def _quantize_dense_fp8(weight: torch.Tensor, block: int = _DENSE_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    blocked = weight.float().reshape(rows // block, block, cols // block, block)
    absmax = blocked.abs().amax(dim=(1, 3))
    scale, byte = _block_scale_bytes(absmax, _FP8_E4M3_MAX)
    scale_expanded = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    quantized = (weight.float() / scale_expanded).to(torch.float8_e4m3fn)
    return quantized, byte.view(torch.float8_e8m0fnu)


def _quantize_expert_mxfp4(weight: torch.Tensor, block: int = _EXPERT_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    blocked = weight.float().reshape(rows, cols // block, block)
    absmax = blocked.abs().amax(dim=-1)
    scale, byte = _block_scale_bytes(absmax, _MXFP4_MAX)
    scale_expanded = scale.repeat_interleave(block, dim=-1)
    normalized = weight.float() / scale_expanded

    magnitudes = _MXFP4_MAGNITUDES.to(normalized.device)
    magnitude_idx = (normalized.abs().unsqueeze(-1) - magnitudes).abs().argmin(dim=-1)
    sign_bit = (normalized < 0).to(torch.uint8) * 8
    nibble = magnitude_idx.to(torch.uint8) + sign_bit

    low, high = nibble[:, 0::2], nibble[:, 1::2]
    packed = (low | (high << 4)).contiguous().view(torch.int8)
    return packed, byte.view(torch.float8_e8m0fnu)


def _quantize_selected(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Quantize dense fp8 keys, plus routed-expert keys to packed MXFP4."""
    quantized = {}
    for key, tensor in state_dict.items():
        if _is_routed_expert_key(key):
            weight, scale = _quantize_expert_mxfp4(tensor)
        elif _is_dense_fp8_key(key):
            weight, scale = _quantize_dense_fp8(tensor)
        else:
            quantized[key] = tensor
            continue
        quantized[key] = weight
        quantized[key.removesuffix(".weight") + ".scale"] = scale
    return quantized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Plain mini checkpoint from scripts/mini_moe.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    state_dict = load_file(args.input_dir / "model.safetensors")
    quantized = _quantize_selected(state_dict)

    n_dense = sum(1 for k in state_dict if _is_dense_fp8_key(k))
    n_experts = sum(1 for k in state_dict if _is_routed_expert_key(k))
    print(f"Quantized {n_dense} dense fp8 keys and {n_experts} routed-expert MXFP4 keys of {len(state_dict)} total.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(quantized, args.output_dir / "model.safetensors", metadata={"format": "pt"})
    for name in _COPIED_FILES:
        src = args.input_dir / name
        if src.exists():
            shutil.copy(src, args.output_dir / name)
    print(f"Saved quantized checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
