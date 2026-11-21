"""Dequantize an Inkling 2 Mini release checkpoint (NVFP4 / MXFP8 routed experts) to bf16.

The release packs the experts of some layers as NVFP4 and of the others as MXFP8.
``tools/convert_inkling2_mxfp8_to_bf16.py`` is for a release that is MXFP8 throughout (Inkling 2 Small's MXFP8 one).

Only ``model.llm.layers.N.mlp.experts.{w13,w2}_weight`` are packed in the release. They are dequantized and their
scale tensors dropped; every other tensor (the vision tower included) is copied unchanged under its checkpoint
name, and ``w13`` keeps its interleaved gate/up rows. ``quantization_config`` is stripped from ``config.json`` and
``hf_quant_config.json`` is not copied, so a serving stack sees plain bf16 experts. The draft head
(``mtp.safetensors``) and the trainer's conversion caches (``prime/``, ``hf/``) are not copied.

With ``--verify`` every dequantized NVFP4 tensor is re-quantized with its stored scales and must give back the
stored codes, which shows that the bf16 rounding lost none of them. It does not check how the codes are decoded.

Usage (from the prime-rl repo):
    uv run python tools/convert_inkling2_loop_to_bf16.py <model_dir> [output_dir] [--verify]

Writes one shard per decoder layer plus one for everything else, to ``<model_dir>-bf16`` by default.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prime_rl.trainer.models.inkling2_loop.dequantize import (
    EXPERT_WEIGHT_RE,
    dequantize_expert_weight,
    nvfp4_requantization_mismatches,
)

LAYER_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.")
SCALE_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")
SKIPPED = (
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    # Conversion caches that a trainer wrote into the model dir. A copy would be read as the export's own conversion.
    "prime",
    "hf",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="Inkling 2 Mini release checkpoint dir")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-bf16")
    parser.add_argument("--verify", action="store_true", help="check that the bf16 rounding loses no NVFP4 code")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir.with_name(input_dir.name + "-bf16")).resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        # The copy would overwrite the release's config and index.
        parser.error("output_dir must be outside input_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tensor_to_shard = json.loads((input_dir / "model.safetensors.index.json").read_text())["weight_map"]
    groups: dict[str, list[str]] = defaultdict(list)
    for name in tensor_to_shard:
        if ".mlp.experts." in name and name.endswith(SCALE_SUFFIXES):
            continue
        match = LAYER_RE.match(name)
        groups[f"layer-{int(match.group(1)):02d}" if match else "other"].append(name)

    shards = {
        shard: safe_open(input_dir / shard, framework="pt", device=device) for shard in set(tensor_to_shard.values())
    }

    def load(name: str) -> torch.Tensor:
        return shards[tensor_to_shard[name]].get_tensor(name)

    weight_map: dict[str, str] = {}
    total_size = 0
    num_dequantized = 0
    group_names = sorted(groups)
    for index, group in enumerate(group_names):
        shard_name = f"model-{index + 1:05d}-of-{len(group_names):05d}.safetensors"
        out_shard: dict[str, torch.Tensor] = {}
        for name in groups[group]:
            tensor = load(name)
            if EXPERT_WEIGHT_RE.search(name) and tensor.element_size() == 1:
                packed = {name: tensor} | {
                    name + suffix: load(name + suffix) for suffix in SCALE_SUFFIXES if name + suffix in tensor_to_shard
                }
                tensor = dequantize_expert_weight(name, packed)
                if args.verify and packed[name].dtype == torch.uint8:
                    mismatches = nvfp4_requantization_mismatches(
                        tensor,
                        packed[name],
                        packed[name + "_scale"],
                        packed[name + "_amax"],
                        packed[name + "_outer_scale_denominator"],
                    )
                    if mismatches:
                        raise ValueError(f"{name}: {mismatches} NVFP4 codes do not survive re-quantization")
                num_dequantized += 1
            out_shard[name] = tensor.cpu()
            weight_map[name] = shard_name
            total_size += tensor.nbytes
        save_file(out_shard, output_dir / shard_name, metadata={"format": "pt"})
        print(f"Wrote {shard_name} ({group}, {len(out_shard)} tensors, {num_dequantized} dequantized so far)")

    if num_dequantized == 0:
        raise ValueError(f"No packed expert weights found in {input_dir} - is this a release checkpoint?")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    config = json.loads((input_dir / "config.json").read_text())
    config.pop("quantization_config", None)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in input_dir.iterdir():
        if path.name.startswith(".") or path.name in SKIPPED or path.suffix == ".safetensors":
            continue
        if path.is_dir():
            shutil.copytree(path, output_dir / path.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(path, output_dir / path.name)
    print(f"Done: {num_dequantized} expert weights dequantized -> {output_dir}")


if __name__ == "__main__":
    main()
