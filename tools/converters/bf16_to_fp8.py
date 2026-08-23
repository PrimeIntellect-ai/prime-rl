"""Quantize a bf16 HF safetensors checkpoint to blockwise FP8 (DeepSeek/GLM format).

2D linear weights are quantized to e4m3 with per-128x128-block fp32 scales
stored under ``<name>.weight_scale_inv``; norms, embeddings, lm_head, router
gates and other sensitive modules stay in the source dtype. The output dir gets
the quantized shards, a rewritten index, all non-weight assets, and a
``quantization_config`` block in ``config.json`` that vLLM loads natively.

Usage (from the prime-rl repo):
    uv run python tools/converters/bf16_to_fp8.py <model_dir> [output_dir] [--block-size 128]

Writes to ``<model_dir>-FP8`` by default.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise

# Module-name substrings that stay unquantized: norms, embeddings, output head,
# MoE router gates, GatedDeltaNet low-rank projections, MTP/indexer projections,
# vision tower. Mirrors the trainer's _DEFAULT_FP8_IGNORE_PATTERNS
# (configs/trainer.py), adapted to HF checkpoint names.
SKIP_SUBSTRINGS = (
    "norm",
    "embed",
    "lm_head",
    "shared_expert_gate",
    "in_proj_a",
    "in_proj_b",
    "eh_proj",
    "weights_proj",
    "visual.",
    "router",
)


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    return (
        name.endswith(".weight")
        and tensor.ndim == 2
        and tensor.is_floating_point()
        and tensor.element_size() > 1
        and not name.endswith(".gate.weight")
        and not any(substring in name for substring in SKIP_SUBSTRINGS)
    )


def quantize_state_dict(
    tensors: dict[str, torch.Tensor], block_size: int, device: str
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Quantize the quantizable tensors of one state-dict slice; returns (out, modules kept in bf16)."""
    out: dict[str, torch.Tensor] = {}
    modules_to_not_convert: list[str] = []
    for name, tensor in tensors.items():
        if should_quantize(name, tensor):
            quantized, scales = quantize_to_fp8_blockwise(tensor.to(device), block_size)
            out[name] = quantized.cpu()
            out[name + "_scale_inv"] = scales.cpu()
        else:
            if name.endswith(".weight") and tensor.ndim == 2 and tensor.is_floating_point():
                modules_to_not_convert.append(name.removesuffix(".weight"))
            if name.endswith(".weight") and tensor.ndim > 2 and tensor.numel() > 1_000_000:
                print(f"Warning: leaving large {tensor.ndim}D tensor unquantized: {name} {tuple(tensor.shape)}")
            out[name] = tensor.cpu()
    return out, modules_to_not_convert


def quantization_config(modules_to_not_convert: list[str], block_size: int) -> dict:
    return {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_size, block_size],
        "modules_to_not_convert": sorted(set(modules_to_not_convert)),
    }


def list_shards(model_dir: Path) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return sorted(set(weight_map.values()))
    if (model_dir / "model.safetensors").exists():
        return ["model.safetensors"]
    raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")


def convert(input_dir: Path, output_dir: Path | None = None, block_size: int = 128) -> Path:
    input_dir = input_dir.resolve()
    output_dir = output_dir or input_dir.with_name(input_dir.name + "-FP8")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_map: dict[str, str] = {}
    total_size = 0
    modules_to_not_convert: list[str] = []
    num_quantized = 0

    for shard_name in list_shards(input_dir):
        with safe_open(input_dir / shard_name, framework="pt", device=device) as f:
            tensors = {name: f.get_tensor(name) for name in f.keys()}
        out_shard, shard_modules = quantize_state_dict(tensors, block_size, device)
        modules_to_not_convert.extend(shard_modules)
        num_quantized += sum(1 for name in out_shard if name.endswith("_scale_inv"))
        for name, tensor in out_shard.items():
            weight_map[name] = shard_name
            total_size += tensor.nbytes
        save_file(out_shard, output_dir / shard_name, metadata={"format": "pt"})
        print(f"Quantized {shard_name} ({num_quantized} tensors so far)")

    if num_quantized == 0:
        raise ValueError(f"No quantizable weights found in {input_dir} - is this a bf16 checkpoint?")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    for path in input_dir.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != "model.safetensors.index.json":
            shutil.copyfile(path, output_dir / path.name)

    config_path = input_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = quantization_config(modules_to_not_convert, block_size)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Done: {num_quantized} weights quantized -> {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="HF safetensors model dir (bf16/fp16/fp32)")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-FP8")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.block_size)


if __name__ == "__main__":
    main()
