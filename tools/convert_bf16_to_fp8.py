"""Quantize a bf16 HF safetensors checkpoint to blockwise FP8 (DeepSeek/GLM format).

2D linear weights are quantized to e4m3 with per-128x128-block fp32 scales
stored under ``<name>.weight_scale_inv``; norms, embeddings, lm_head, router
gates and other sensitive modules stay in the source dtype. Fused MoE experts
(``experts.gate_up_proj`` and ``experts.down_proj``) are split into per-expert
``experts.<i>.{gate,up,down}_proj.weight`` first. The output dir gets
the quantized shards, a rewritten index, all non-weight assets, and a
``quantization_config`` block in ``config.json`` that vLLM loads natively.

Usage (from the prime-rl repo):
    uv run python tools/convert_bf16_to_fp8.py <model_dir> [output_dir] [--block-size 128]

Writes to ``<model_dir>-FP8`` by default.
"""

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn import functional as F

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


# Fused MoE expert tensors of the HF Qwen3.5 layout have no ``.weight`` suffix.
FUSED_EXPERT_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")


def quantize_to_fp8_blockwise(weight: Tensor, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize a 2D tensor to FP8 e4m3 with per-block scales."""
    if weight.ndim != 2:
        raise ValueError(f"FP8 quantization expects a 2D tensor, got shape={tuple(weight.shape)}")

    rows, cols = weight.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size

    padded = F.pad(weight, (0, pad_cols, 0, pad_rows)).contiguous()
    padded_rows, padded_cols = padded.shape
    blocks = padded.view(
        padded_rows // block_size,
        block_size,
        padded_cols // block_size,
        block_size,
    ).permute(0, 2, 1, 3)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    inv_fp8_max = torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=weight.device)
    max_abs = blocks.float().abs().amax(dim=(2, 3))
    # CUDA divides by a scalar via a float32 reciprocal. Round that reciprocal
    # before multiplying so CPU exports produce the same block scales.
    scales = (max_abs * inv_fp8_max).clamp(min=1e-12)
    blocks_fp8 = (blocks.float() / scales[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    quantized = blocks_fp8.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols].contiguous()
    return quantized, scales.float().contiguous()


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    return (
        name.endswith(".weight")
        and tensor.ndim == 2
        and tensor.is_floating_point()
        and tensor.element_size() > 1
        and not name.endswith(".gate.weight")
        and not any(substring in name for substring in SKIP_SUBSTRINGS)
    )


def is_fused_expert_tensor(tensor: Tensor) -> bool:
    return tensor.ndim == 3 and tensor.is_floating_point() and tensor.element_size() > 1


def fused_expert_prefixes(shapes: dict[str, Sequence[int]]) -> set[str]:
    """Prefixes (ending in ``.experts``) whose ``gate_up_proj`` [E, 2I, H] and ``down_proj`` [E, H, I] are both present.

    The shapes are the gate: gpt-oss stores [E, H, 2I] with interleaved gate and up,
    which never satisfies them, so it is left alone.
    """
    prefixes = set()
    for name, gate_up in shapes.items():
        prefix = name.removesuffix(".gate_up_proj")
        down = shapes.get(f"{prefix}.down_proj")
        if name == prefix or not prefix.endswith(".experts") or down is None or len(gate_up) != 3:
            continue
        num_experts, gate_up_rows, hidden = gate_up
        if tuple(down) == (num_experts, hidden, gate_up_rows // 2) and gate_up_rows % 2 == 0:
            prefixes.add(prefix)
    return prefixes


def unfuse_experts(name: str, tensor: Tensor) -> dict[str, Tensor]:
    """Split a fused expert tensor into the per-expert 2D weights that vLLM and the official FP8 checkpoints use."""
    prefix, _, proj = name.rpartition(".")
    if proj == "down_proj":
        return {f"{prefix}.{expert}.down_proj.weight": weight for expert, weight in enumerate(tensor)}
    gate, up = tensor.chunk(2, dim=1)
    weights = {}
    for expert in range(tensor.shape[0]):
        weights[f"{prefix}.{expert}.gate_proj.weight"] = gate[expert]
        weights[f"{prefix}.{expert}.up_proj.weight"] = up[expert]
    return weights


def quantize_state_dict(
    tensors: dict[str, torch.Tensor], block_size: int, device: str, fused_prefixes: set[str] | None = None
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Quantize the quantizable tensors of one CPU state-dict slice; returns (out, modules kept in bf16).

    Tensors move to ``device`` one at a time (and are popped from the input as they
    are consumed), so peak device memory stays near the largest single weight. Fused
    experts under ``fused_prefixes`` (by default the pairs in ``tensors``) are split into
    per-expert weights first. Any other fused expert tensor stays in bf16 and its layer
    goes into the modules kept in bf16.
    """
    out: dict[str, torch.Tensor] = {}
    modules_to_not_convert: list[str] = []
    if fused_prefixes is None:
        fused_prefixes = fused_expert_prefixes({name: tensor.shape for name, tensor in tensors.items()})
    for name in list(tensors):
        tensor = tensors.pop(name)
        if (
            name.endswith(FUSED_EXPERT_SUFFIXES)
            and name.rpartition(".")[0] in fused_prefixes
            and is_fused_expert_tensor(tensor)
        ):
            weights = unfuse_experts(name, tensor)
        else:
            weights = {name: tensor}
        for weight_name, tensor in weights.items():
            if should_quantize(weight_name, tensor):
                quantized, scales = quantize_to_fp8_blockwise(tensor.to(device), block_size)
                out[weight_name] = quantized.cpu()
                out[weight_name + "_scale_inv"] = scales.cpu()
                continue
            kept_fused_experts = weight_name.endswith(FUSED_EXPERT_SUFFIXES) and is_fused_expert_tensor(tensor)
            if kept_fused_experts:
                modules_to_not_convert.append(weight_name.rpartition(".")[0])
            if weight_name.endswith(".weight") and tensor.ndim == 2 and tensor.is_floating_point():
                modules_to_not_convert.append(weight_name.removesuffix(".weight"))
            if kept_fused_experts or (
                weight_name.endswith(".weight") and tensor.ndim > 2 and tensor.numel() > 1_000_000
            ):
                print(f"Warning: leaving large {tensor.ndim}D tensor unquantized: {weight_name} {tuple(tensor.shape)}")
            out[weight_name] = tensor.cpu()
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

    # Official Qwen3.5 releases store a layer's gate_up_proj and down_proj in different
    # shards, so fused experts are matched by the shapes of all shards first.
    shards = list_shards(input_dir)
    shapes = {}
    for shard_name in shards:
        with safe_open(input_dir / shard_name, framework="pt", device="cpu") as f:
            shapes.update({name: f.get_slice(name).get_shape() for name in f.keys()})
    fused_prefixes = fused_expert_prefixes(shapes)

    for shard_name in shards:
        # Shards land in host memory; quantize_state_dict streams them through the
        # device tensor-by-tensor so a whole shard never sits on the GPU at once.
        with safe_open(input_dir / shard_name, framework="pt", device="cpu") as f:
            tensors = {name: f.get_tensor(name) for name in f.keys()}
        out_shard, shard_modules = quantize_state_dict(tensors, block_size, device, fused_prefixes)
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
