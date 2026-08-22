"""Dequantize a blockwise-FP8 HF checkpoint (DeepSeek/GLM format) to bf16.

The inverse of ``scripts/bf16_to_fp8.py``, and works on official fp8-only
releases (e.g. GLM-5-FP8): every e4m3 tensor is multiplied by its
``<name>_scale_inv`` per-block scales and stored as bf16, scale tensors are
dropped, and ``quantization_config`` is stripped from ``config.json``.

Usage (from the prime-rl repo):
    uv run python scripts/fp8_to_bf16.py <model_dir> [output_dir]

Writes to ``<model_dir>-BF16`` by default.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequantize_blockwise(weight: torch.Tensor, scales: torch.Tensor, block_size: tuple[int, int]) -> torch.Tensor:
    rows, cols = weight.shape
    block_rows, block_cols = block_size
    expanded = scales.repeat_interleave(block_rows, dim=0)[:rows].repeat_interleave(block_cols, dim=1)[:, :cols]
    return (weight.to(torch.float32) * expanded).to(torch.bfloat16)


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
    parser.add_argument("input_dir", type=Path, help="HF safetensors model dir with blockwise-fp8 weights")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-BF16")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir or input_dir.with_name(input_dir.name + "-BF16")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = json.loads((input_dir / "config.json").read_text())
    quantization_config = config.pop("quantization_config", {})
    if quantization_config.get("scale_fmt") == "ue8m0":
        raise ValueError("ue8m0 scales are not supported - only plain float weight_scale_inv checkpoints")
    block_size = tuple(quantization_config.get("weight_block_size", [128, 128]))

    index_path = input_dir / "model.safetensors.index.json"
    scale_to_shard = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else {}

    # Scales can live in a different shard than their weight; cache the most
    # recently opened shards to avoid re-reading them per tensor.
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def get_scales(name: str, current_shard: str) -> torch.Tensor:
        shard_name = scale_to_shard.get(name, current_shard)
        if shard_name not in shard_cache:
            with safe_open(input_dir / shard_name, framework="pt", device=device) as f:
                shard_cache[shard_name] = {key: f.get_tensor(key) for key in f.keys()}
            while len(shard_cache) > 2:
                shard_cache.pop(next(iter(shard_cache)))
        if name not in shard_cache[shard_name]:
            raise KeyError(f"Missing {name} for fp8 tensor {name.removesuffix('_scale_inv')}")
        return shard_cache[shard_name][name]

    weight_map: dict[str, str] = {}
    total_size = 0
    num_dequantized = 0

    for shard_name in list_shards(input_dir):
        out_shard: dict[str, torch.Tensor] = {}
        with safe_open(input_dir / shard_name, framework="pt", device=device) as f:
            for name in f.keys():
                if name.endswith("_scale_inv"):
                    continue
                tensor = f.get_tensor(name)
                if tensor.element_size() == 1:
                    scales = get_scales(name + "_scale_inv", shard_name)
                    out_shard[name] = dequantize_blockwise(tensor, scales, block_size).cpu()
                    num_dequantized += 1
                else:
                    out_shard[name] = tensor.cpu()
        for name, tensor in out_shard.items():
            weight_map[name] = shard_name
            total_size += tensor.nbytes
        save_file(out_shard, output_dir / shard_name, metadata={"format": "pt"})
        print(f"Dequantized {shard_name} ({num_dequantized} tensors so far)")

    if num_dequantized == 0:
        raise ValueError(f"No fp8 weights found in {input_dir} - is this an fp8 checkpoint?")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    for path in input_dir.iterdir():
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name not in ("model.safetensors.index.json", "config.json")
        ):
            shutil.copyfile(path, output_dir / path.name)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Done: {num_dequantized} weights dequantized -> {output_dir}")


if __name__ == "__main__":
    main()
