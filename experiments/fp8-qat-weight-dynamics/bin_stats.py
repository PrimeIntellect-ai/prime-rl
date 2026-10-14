"""Per-family weight RMS and FP8 e4m3 bin width (UE8M0 and amax/448 block scales) for a sample of layers.

Usage (on a GPU node, from the repo root):
    uv run --no-sync python experiments/fp8-qat-weight-dynamics/bin_stats.py <snapshot> [--experts-per-layer 16]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

BLOCK = 128
LAYERS = (0, 16, 32, 47)
FAMILY = re.compile(
    r"model\.layers\.(\d+)\.(?:self_attn\.([qkvo])_proj|mlp\.experts\.(\d+)\.(gate|up|down)_proj)\.weight$"
)


def block_scales(w: torch.Tensor, ue8m0: bool) -> torch.Tensor:
    rows, cols = w.shape
    amax = w.abs().view(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK).amax(dim=(1, 3))
    scale = torch.clamp(amax / 448.0, min=1e-4)
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)


def e4m3_bin(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled = (w / scale).abs()
    exponent = torch.floor(torch.log2(scaled.clamp(min=2.0**-6)))
    return torch.exp2(exponent - 3) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--experts-per-layer", type=int, default=16)
    args = parser.parse_args()

    samples: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    for shard in sorted(args.snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cuda") as f:
            for name in f.keys():
                match = FAMILY.match(name)
                if match is None or int(match.group(1)) not in LAYERS:
                    continue
                if match.group(3) is not None and int(match.group(3)) >= args.experts_per_layer:
                    continue
                family = f"attn.{match.group(2)}" if match.group(2) else f"expert.{match.group(4)}"
                w = f.get_tensor(name).float()
                samples[family]["w"].append(w.flatten())
                for grid, ue8m0 in (("ue8m0", True), ("amax", False)):
                    samples[family][f"bin_{grid}"].append(e4m3_bin(w, block_scales(w, ue8m0)).flatten())

    report = {}
    for family in sorted(samples):
        w = torch.cat(samples[family]["w"])
        row = {"rms_w": float(w.square().mean().sqrt()), "median_abs_w": float(w.abs().median())}
        for grid in ("ue8m0", "amax"):
            bin_width = torch.cat(samples[family][f"bin_{grid}"])
            row[f"median_bin_{grid}"] = float(bin_width.median())
            row[f"rms_bin_{grid}"] = float(bin_width.square().mean().sqrt())
        report[family] = row
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
