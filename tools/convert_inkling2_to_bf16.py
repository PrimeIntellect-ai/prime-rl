"""Dequantize an Inkling 2 release checkpoint (NVFP4 / MXFP8 routed experts) to bf16.

The releases store the routed experts, ``model.llm.layers.N.mlp.experts.{w13,w2}_weight``, in NVFP4 or
MXFP8, chosen per tensor; every other tensor is plain bf16 or fp32. The trainer reads bf16 experts, so this
writes the same checkpoint with every expert weight dequantized and its scale tensors dropped.

``hf_quant_config.json`` (the release's per-tensor precision metadata) is rewritten to list the experts as
bf16: vLLM builds a layer's experts from it, and serves bf16 experts on GPUs without the Blackwell kernels
of the quantized ones. ``quantization_config`` is dropped from ``config.json``, where it would take
precedence over that file.

NVFP4 (uint8 weight): two E2M1 codes per byte along the input dim, low nibble first. Bit 3 of a code is the
sign and bits 0-2 index (0, 0.5, 1, 1.5, 2, 3, 4, 6). Every 16 inputs share an E4M3 scale (``<name>_scale``)
and every expert a factor ``<name>_amax[e] / <name>_outer_scale_denominator``.

MXFP8 (E4M3 weight): every 32 inputs share the power-of-two scale ``2 ** (s - 127)``, with ``s`` the uint8 in
``<name>_scale``.

Usage (from the prime-rl repo):
    uv run python tools/convert_inkling2_to_bf16.py <model_dir> [output_dir]

Writes to ``<model_dir>-BF16`` by default.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
SCALE_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")
PRECISION_METADATA = "hf_quant_config.json"
QUANTIZED_COMPUTE_DTYPES = ("nvfp4", "mxfp8")
# Experts dequantized at a time: bounds the fp32 intermediate to a few GB.
EXPERT_CHUNK = 16


def dequantize_nvfp4(
    packed: torch.Tensor, block_scales: torch.Tensor, amax: torch.Tensor, outer_scale_denominator: torch.Tensor
) -> torch.Tensor:
    """``packed [E, N, K / 2]`` with ``block_scales [E, N, K / 16]`` and ``amax [E]`` -> bf16 ``[E, N, K]``."""
    magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=packed.device)
    code_values = torch.cat((magnitudes, -magnitudes))
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    weight = code_values[codes].unflatten(-1, (-1, NVFP4_BLOCK)) * block_scales.float().unsqueeze(-1)
    expert_factor = (amax.float() / outer_scale_denominator.float()).view(-1, 1, 1)
    return (weight.flatten(-2) * expert_factor).to(torch.bfloat16)


def dequantize_mxfp8(weight: torch.Tensor, block_exponents: torch.Tensor) -> torch.Tensor:
    """E4M3 ``weight [E, N, K]`` with uint8 ``block_exponents [E, N, K / 32]`` -> bf16 ``[E, N, K]``."""
    block_scales = torch.exp2(block_exponents.float() - 127.0).unsqueeze(-1)
    return (weight.float().unflatten(-1, (-1, MXFP8_BLOCK)) * block_scales).flatten(-2).to(torch.bfloat16)


def is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


def is_expert_scale(name: str) -> bool:
    return is_expert_tensor(name) and name.endswith(SCALE_SUFFIXES)


def is_quantized(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.uint8, torch.float8_e4m3fn)


def list_experts_as_bf16(precision_metadata: dict) -> dict:
    """``hf_quant_config.json`` with the tensors of its NVFP4 / MXFP8 groups moved to one bf16 group."""
    groups = []
    experts: list[str] = []
    for group in precision_metadata["model"]["parameter_groups"]:
        if group["compute_dtype"] in QUANTIZED_COMPUTE_DTYPES:
            experts.extend(group["tensors"])
        else:
            groups.append(group)
    groups.append(
        {
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "required_activation_quantization": None,
            "tensors": sorted(experts),
            "weight_block_shape": [],
            "weight_transform": "cast",
        }
    )
    return {**precision_metadata, "model": {**precision_metadata["model"], "parameter_groups": groups}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="Inkling 2 release checkpoint dir (NVFP4 or MXFP8 experts)")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-BF16")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir or input_dir.with_name(input_dir.name + "-BF16")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tensor_to_shard: dict[str, str] = json.loads((input_dir / "model.safetensors.index.json").read_text())["weight_map"]

    # Scales can live in a different shard than their weight.
    open_shards: dict[str, safe_open] = {}

    def get_tensor(name: str) -> torch.Tensor:
        shard_name = tensor_to_shard[name]
        if shard_name not in open_shards:
            open_shards[shard_name] = safe_open(input_dir / shard_name, framework="pt", device="cpu")
        return open_shards[shard_name].get_tensor(name)

    def dequantize(name: str, weight: torch.Tensor) -> torch.Tensor:
        scales = get_tensor(name + "_scale")
        chunks = []
        if weight.dtype == torch.uint8:
            amax = get_tensor(name + "_amax").to(device)
            denominator = get_tensor(name + "_outer_scale_denominator").to(device)
            for start in range(0, weight.shape[0], EXPERT_CHUNK):
                experts = slice(start, start + EXPERT_CHUNK)
                chunks.append(
                    dequantize_nvfp4(
                        weight[experts].to(device), scales[experts].to(device), amax[experts], denominator
                    ).cpu()
                )
        else:
            for start in range(0, weight.shape[0], EXPERT_CHUNK):
                experts = slice(start, start + EXPERT_CHUNK)
                chunks.append(dequantize_mxfp8(weight[experts].to(device), scales[experts].to(device)).cpu())
        return torch.cat(chunks)

    weight_map: dict[str, str] = {}
    total_size = 0
    num_dequantized = 0
    for shard_name in sorted(set(tensor_to_shard.values())):
        out_shard: dict[str, torch.Tensor] = {}
        for name in sorted(name for name, shard in tensor_to_shard.items() if shard == shard_name):
            if is_expert_scale(name):
                continue
            tensor = get_tensor(name)
            if is_expert_tensor(name) and is_quantized(tensor):
                tensor = dequantize(name, tensor)
                num_dequantized += 1
            out_shard[name] = tensor
        for name, tensor in out_shard.items():
            weight_map[name] = shard_name
            total_size += tensor.nbytes
        save_file(out_shard, output_dir / shard_name, metadata={"format": "pt"})
        print(f"Wrote {shard_name} ({num_dequantized} expert weights dequantized so far)")

    if num_dequantized == 0:
        raise ValueError(f"No quantized expert weights found in {input_dir} - is this an Inkling 2 release?")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    config = json.loads((input_dir / "config.json").read_text())
    config.pop("quantization_config", None)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    precision_metadata = json.loads((input_dir / PRECISION_METADATA).read_text())
    (output_dir / PRECISION_METADATA).write_text(json.dumps(list_experts_as_bf16(precision_metadata), indent=2) + "\n")

    converted = {"model.safetensors.index.json", "config.json", PRECISION_METADATA, *tensor_to_shard.values()}
    for path in input_dir.iterdir():
        # Everything else is copied as is, the speculator's `mtp.safetensors` included.
        if path.is_file() and path.name not in converted:
            shutil.copyfile(path, output_dir / path.name)
    print(f"Done: {num_dequantized} expert weights dequantized -> {output_dir}")


if __name__ == "__main__":
    main()
