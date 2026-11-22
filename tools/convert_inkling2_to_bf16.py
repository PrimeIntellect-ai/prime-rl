"""Dequantize an Inkling 2 release checkpoint (NVFP4 / MXFP8 routed experts) to bf16.

The releases store the routed experts, ``model.llm.layers.N.mlp.experts.{w13,w2}_weight``, in NVFP4 or
MXFP8, chosen per tensor; every other tensor is plain bf16 or fp32. The trainer reads bf16 experts, so this
writes the same checkpoint with every expert weight dequantized and its scale tensors dropped.

Convert ``thinkingmachines/Inkling-2-Small-MXFP8``: an MXFP8 weight is a bf16 number, so that conversion is
exact. An NVFP4 weight is not, and is rounded to the nearest one.

MXFP8 (E4M3 weight): every 32 inputs share the power-of-two scale ``2 ** (s - 127)``, with ``s`` the uint8 in
``<name>_scale``. A NaN code, or a scale for which some code would not be a bf16 number, stops the conversion.

NVFP4 (uint8 weight): two E2M1 codes per byte along the input dim, low nibble first. Bit 3 of a code is the
sign and bits 0-2 index (0, 0.5, 1, 1.5, 2, 3, 4, 6). Every 16 inputs share an E4M3 scale (``<name>_scale``)
and every expert a factor ``<name>_amax[e] / <name>_outer_scale_denominator``.

So that the output is read as bf16:

* ``hf_quant_config.json`` (the release's per-tensor precision metadata) lists the experts as bf16. vLLM
  builds a layer's experts from it, and serves bf16 experts on GPUs without the Blackwell kernels of the
  quantized ones.
* ``config.json`` loses ``quantization_config``, which would take precedence over that file, and gains
  ``transformers_version``: without either, transformers reads the directory as the MXFP8 release.

Every other file is copied, subdirectories included. ``bf16_conversion.json`` records the source revision
and this tool's commit. Each file is written under a temporary name and renamed when complete, so a rerun
into the same output directory continues after the files that are there; ``model.safetensors.index.json``
is written last.

Usage (from the prime-rl repo):
    uv run python tools/convert_inkling2_to_bf16.py <model_dir> [output_dir] [--source-revision <commit>]

Writes to ``<model_dir>-BF16`` by default.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
# 2 ** (s - 127) for every scale byte s: read from a table, so that no device computes a power.
MXFP8_SCALES = torch.tensor([2.0 ** (scale - 127) for scale in range(256)], dtype=torch.float32)
# The scale bytes for which every E4M3 code times its scale is a bf16 number. Below, the smallest codes fall
# between bf16's subnormals; above, the largest overflow.
MXFP8_SCALE_RANGE = (3, 246)
SCALE_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")
EXPERT_FORMATS = {"F8_E4M3": "mxfp8", "U8": "nvfp4"}
INDEX = "model.safetensors.index.json"
PRECISION_METADATA = "hf_quant_config.json"
RECORD = "bf16_conversion.json"
QUANTIZED_COMPUTE_DTYPES = ("nvfp4", "mxfp8")
# The first transformers release with Inkling 2.
TRANSFORMERS_VERSION = "5.23.0"
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
    """E4M3 ``weight [E, N, K]`` with uint8 ``block_exponents [E, N, K / 32]`` -> bf16 ``[E, N, K]``.

    Exact for the inputs `check_mxfp8` accepts: the product of a code and a power of two is an fp32 number,
    and a bf16 number as well.
    """
    block_scales = MXFP8_SCALES.to(weight.device)[block_exponents.long()].unsqueeze(-1)
    return (weight.float().unflatten(-1, (-1, MXFP8_BLOCK)) * block_scales).flatten(-2).to(torch.bfloat16)


def check_mxfp8(name: str, weight: torch.Tensor, block_exponents: torch.Tensor) -> None:
    if block_exponents.dtype != torch.uint8:
        raise ValueError(f"{name}_scale is {block_exponents.dtype}, not uint8")
    if ((weight.view(torch.uint8) & 0x7F) == 0x7F).any():
        raise ValueError(f"{name} holds NaN codes")
    low, high = MXFP8_SCALE_RANGE
    smallest, largest = int(block_exponents.min()), int(block_exponents.max())
    if smallest < low or largest > high:
        raise ValueError(
            f"{name}_scale holds scale bytes from {smallest} to {largest}; "
            f"the bf16 result is exact from {low} to {high} only"
        )


def check_device_arithmetic(device: str) -> None:
    """Every E4M3 code at every scale in range has to dequantize on ``device`` to the bits it gives on the CPU."""
    codes = torch.tensor([code for code in range(256) if code & 0x7F != 0x7F] + [0, 0], dtype=torch.uint8)
    low, high = MXFP8_SCALE_RANGE
    weight = codes.view(torch.float8_e4m3fn).expand(1, high + 1 - low, -1).contiguous()
    block_exponents = torch.arange(low, high + 1, dtype=torch.uint8).view(1, -1, 1).repeat(1, 1, 256 // MXFP8_BLOCK)
    on_cpu = dequantize_mxfp8(weight, block_exponents)
    on_device = dequantize_mxfp8(weight.to(device), block_exponents.to(device)).cpu()
    if not torch.equal(on_device.view(torch.int16), on_cpu.view(torch.int16)):
        raise RuntimeError(f"MXFP8 dequantization on {device} differs from the CPU's")


def is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


def is_expert_scale(name: str) -> bool:
    return is_expert_tensor(name) and name.endswith(SCALE_SUFFIXES)


def read_header(path: Path) -> dict[str, dict]:
    """A safetensors file's tensors: name -> dtype, shape and byte range."""
    with path.open("rb") as file:
        header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
    header.pop("__metadata__", None)
    return header


def quantized_experts(tensor_to_shard: dict[str, str], headers: dict[str, dict[str, dict]]) -> dict[str, str]:
    """The expert weights to dequantize and their format, once every tensor of the index is found in its shard
    and every quantized weight has its scales."""
    missing = [name for name, shard_name in tensor_to_shard.items() if name not in headers[shard_name]]
    formats = {}
    for name in tensor_to_shard:
        dtype = headers[tensor_to_shard[name]].get(name, {}).get("dtype")
        if is_expert_tensor(name) and not is_expert_scale(name) and dtype in EXPERT_FORMATS:
            formats[name] = EXPERT_FORMATS[dtype]
            suffixes = SCALE_SUFFIXES if formats[name] == "nvfp4" else SCALE_SUFFIXES[:1]
            missing += [name + suffix for suffix in suffixes if name + suffix not in tensor_to_shard]
    if missing:
        raise ValueError(f"The checkpoint lacks {len(missing)} tensors, such as {missing[:3]}")
    if not formats:
        raise ValueError("No quantized expert weights found - is this an Inkling 2 release?")
    return formats


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


def source_of(input_dir: Path) -> tuple[str, str | None]:
    """The checkpoint's name and revision. A snapshot directory of the Hub cache,
    ``models--<org>--<name>/snapshots/<commit>``, gives both."""
    if input_dir.parent.name == "snapshots" and input_dir.parent.parent.name.startswith("models--"):
        return input_dir.parent.parent.name.removeprefix("models--").replace("--", "/"), input_dir.name
    return input_dir.name, None


def tool_commit() -> str | None:
    """The commit of the checkout this file is in, marked ``-dirty`` when tracked files differ from it."""
    git = ["git", "-C", str(Path(__file__).resolve().parent)]
    if shutil.which("git") is None:
        return None
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        return None
    changes = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    return head.stdout.strip() + ("-dirty" if changes.stdout.strip() else "")


def partial_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".partial")


def finish(partial: Path) -> None:
    """Give a completed file its name, with the mode of a newly created file (safetensors writes 0600)."""
    umask = os.umask(0)
    os.umask(umask)
    partial.chmod(0o666 & ~umask)
    partial.replace(partial.with_suffix(""))


def write_json(path: Path, content: dict, **kwargs) -> None:
    partial = partial_path(path)
    partial.write_text(json.dumps(content, indent=2, **kwargs) + "\n")
    finish(partial)


def convert(input_dir: Path, output_dir: Path, source_revision: str | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Everything the conversion reads is checked before it writes.
    tensor_to_shard: dict[str, str] = json.loads((input_dir / INDEX).read_text())["weight_map"]
    config = json.loads((input_dir / "config.json").read_text())
    precision_metadata = list_experts_as_bf16(json.loads((input_dir / PRECISION_METADATA).read_text()))
    shard_names = sorted(set(tensor_to_shard.values()))
    formats = quantized_experts(tensor_to_shard, {name: read_header(input_dir / name) for name in shard_names})
    check_device_arithmetic(device)

    source, snapshot_revision = source_of(input_dir)
    record = {
        "source": source,
        "source_revision": source_revision or snapshot_revision,
        "tool": "tools/convert_inkling2_to_bf16.py",
        "tool_commit": tool_commit(),
        "expert_weights": dict(Counter(formats.values())),
        "exact": "nvfp4" not in formats.values(),
    }
    if (output_dir / RECORD).exists():
        if json.loads((output_dir / RECORD).read_text()) != record:
            raise ValueError(
                f"{output_dir} holds the conversion of another source, or by another version of this tool "
                f"(see {RECORD}). Convert into an empty directory."
            )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty and has no {RECORD}. Convert into an empty directory.")
    else:
        write_json(output_dir / RECORD, record)

    if not record["exact"]:
        print(
            f"WARNING: {record['expert_weights']['nvfp4']} of {len(formats)} expert weights are NVFP4. They are "
            "not bf16 numbers, so this conversion rounds them, and they are a coarser copy of the weights than "
            "MXFP8. Convert thinkingmachines/Inkling-2-Small-MXFP8 instead: that conversion is exact.",
            file=sys.stderr,
        )
    if record["source_revision"] is None:
        print(f"The source revision is unknown, so {RECORD} does not record it: pass --source-revision.")

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
        if formats[name] == "nvfp4":
            amax = get_tensor(name + "_amax").to(device)
            denominator = get_tensor(name + "_outer_scale_denominator").to(device)
            for start in range(0, weight.shape[0], EXPERT_CHUNK):
                experts = slice(start, start + EXPERT_CHUNK)
                chunk = dequantize_nvfp4(
                    weight[experts].to(device), scales[experts].to(device), amax[experts], denominator
                )
                if not chunk.isfinite().all():
                    raise ValueError(f"{name} dequantizes to values that are not finite")
                chunks.append(chunk.cpu())
        else:
            for start in range(0, weight.shape[0], EXPERT_CHUNK):
                experts = slice(start, start + EXPERT_CHUNK)
                check_mxfp8(name, weight[experts], scales[experts])
                chunks.append(dequantize_mxfp8(weight[experts].to(device), scales[experts].to(device)).cpu())
        return torch.cat(chunks)

    for shard_name in shard_names:
        if (output_dir / shard_name).exists():
            print(f"Kept {shard_name}")
            continue
        out_shard: dict[str, torch.Tensor] = {}
        for name in sorted(name for name, shard in tensor_to_shard.items() if shard == shard_name):
            if is_expert_scale(name):
                continue
            tensor = get_tensor(name)
            out_shard[name] = dequantize(name, tensor) if name in formats else tensor
        partial = partial_path(output_dir / shard_name)
        save_file(out_shard, partial, metadata={"format": "pt"})
        finish(partial)
        print(f"Wrote {shard_name}")

    converted = {INDEX, "config.json", PRECISION_METADATA, *shard_names}
    for path in sorted(input_dir.rglob("*")):
        relative = path.relative_to(input_dir)
        # Everything else is copied as is, the speculator's `mtp.safetensors` included. `.cache` is the Hub
        # client's record of a download.
        if not path.is_file() or relative.as_posix() in converted or relative.parts[0] == ".cache":
            continue
        if not (output_dir / relative).exists():
            partial = partial_path(output_dir / relative)
            shutil.copyfile(path, partial)
            finish(partial)

    config.pop("quantization_config", None)
    config["transformers_version"] = TRANSFORMERS_VERSION
    write_json(output_dir / "config.json", config)
    write_json(output_dir / PRECISION_METADATA, precision_metadata)

    # The index follows the files that are there, whichever run wrote them.
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name in shard_names:
        for name, tensor in read_header(output_dir / shard_name).items():
            weight_map[name] = shard_name
            total_size += tensor["data_offsets"][1] - tensor["data_offsets"][0]
    write_json(output_dir / INDEX, {"metadata": {"total_size": total_size}, "weight_map": weight_map}, sort_keys=True)
    print(f"Done: {len(formats)} expert weights dequantized -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="Inkling 2 release checkpoint dir (MXFP8 or NVFP4 experts)")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-BF16")
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Hub commit of the checkpoint, to record in the output (default: the one in a Hub cache snapshot path)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir or input_dir.with_name(input_dir.name + "-BF16")
    convert(input_dir, output_dir, args.source_revision)


if __name__ == "__main__":
    main()
