"""Quantize a bf16 HF safetensors checkpoint to blockwise FP8 (DeepSeek/GLM format).

2D linear weights are quantized to e4m3 with per-128x128-block fp32 scales
stored under ``<name>.weight_scale_inv``; norms, embeddings, lm_head, router
gates and other sensitive modules stay in the source dtype. The output dir gets
the quantized shards, a rewritten index, all non-weight assets, and a
``quantization_config`` block in ``config.json`` that vLLM loads natively.

Usage (from the prime-rl repo):
    uv run python scripts/bf16_to_fp8.py <model_dir> [output_dir] [--block-size 128]

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
# MoE router gates, GatedDeltaNet low-rank projections, MTP projection, vision tower.
SKIP_SUBSTRINGS = (
    "norm",
    "embed",
    "lm_head",
    "shared_expert_gate",
    "in_proj_a",
    "in_proj_b",
    "eh_proj",
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


def list_shards(model_dir: Path) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return sorted(set(weight_map.values()))
    if (model_dir / "model.safetensors").exists():
        return ["model.safetensors"]
    raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="HF safetensors model dir (bf16/fp16/fp32)")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-FP8")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir or input_dir.with_name(input_dir.name + "-FP8")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_map: dict[str, str] = {}
    total_size = 0
    modules_to_not_convert: list[str] = []
    num_quantized = 0

    for shard_name in list_shards(input_dir):
        out_shard: dict[str, torch.Tensor] = {}
        with safe_open(input_dir / shard_name, framework="pt", device=device) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if should_quantize(name, tensor):
                    quantized, scales = quantize_to_fp8_blockwise(tensor, args.block_size)
                    out_shard[name] = quantized.cpu()
                    out_shard[name + "_scale_inv"] = scales.cpu()
                    num_quantized += 1
                else:
                    if name.endswith(".weight") and tensor.ndim == 2 and tensor.is_floating_point():
                        modules_to_not_convert.append(name.removesuffix(".weight"))
                    if name.endswith(".weight") and tensor.ndim > 2 and tensor.numel() > 1_000_000:
                        print(f"Warning: leaving large {tensor.ndim}D tensor unquantized: {name} {tuple(tensor.shape)}")
                    out_shard[name] = tensor.cpu()
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
    config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [args.block_size, args.block_size],
        "modules_to_not_convert": sorted(set(modules_to_not_convert)),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Done: {num_quantized} weights quantized -> {output_dir}")


if __name__ == "__main__":
    main()
