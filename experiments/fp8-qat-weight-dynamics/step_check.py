"""Quick per-family weight-shift check of an fp32 master export against its start snapshot.

Reports, over the tensors present in the export: share changed in bf16, median |delta|/bin on the start's UE8M0
and amax grids, and under UE8M0 round-to-nearest serving the erased and crossed shares. "Erased" is changed in bf16
but identical after serving, "crossed" is different after serving.

Usage (on a GPU node, from the repo root):
    uv run --no-sync python experiments/fp8-qat-weight-dynamics/step_check.py <start_snapshot> <export_dir>
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from bin_stats import block_scales, e4m3_bin
from safetensors import safe_open

FAMILY = re.compile(r"model\.layers\.\d+\.(?:self_attn\.([qkvo])_proj|mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight$")


def serve_ue8m0_rtn(w: torch.Tensor) -> torch.Tensor:
    scale = block_scales(w, ue8m0=True)
    return (w / scale).to(torch.float8_e4m3fn).float() * scale


def quantile(values: list[torch.Tensor], q: float) -> float:
    ordered = torch.cat(values).sort().values
    return float(ordered[round(q * (ordered.numel() - 1))])


def tensor_locations(directory: Path) -> dict[str, Path]:
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    return {name: directory / shard for name, shard in weight_map.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", type=Path)
    parser.add_argument("export", type=Path)
    args = parser.parse_args()

    start_files = tensor_locations(args.start)
    export_files = tensor_locations(args.export)
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ratios: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    generator = torch.Generator(device="cuda").manual_seed(0)
    for name in sorted(export_files):
        match = FAMILY.match(name)
        if match is None:
            continue
        family = f"attn.{match.group(1)}" if match.group(1) else f"expert.{match.group(2)}"
        with safe_open(start_files[name], framework="pt", device="cuda") as f:
            start = f.get_tensor(name).float()
        with safe_open(export_files[name], framework="pt", device="cuda") as f:
            master = f.get_tensor(name).float()
        delta = master - start
        changed = master.to(torch.bfloat16).float() != start
        crossed = serve_ue8m0_rtn(master.to(torch.bfloat16).float()) != serve_ue8m0_rtn(start)
        row = sums[family]
        row["elements"] += start.numel()
        row["changed"] += int(changed.sum())
        row["erased"] += int((changed & ~crossed).sum())
        row["crossed"] += int(crossed.sum())
        row["sum_sq_delta"] += float(delta.square().sum())
        keep = torch.rand(start.numel(), device="cuda", generator=generator) < 1e-2
        for grid, ue8m0 in (("ue8m0", True), ("amax", False)):
            ratio = (delta.abs() / e4m3_bin(start, block_scales(start, ue8m0))).flatten()[keep]
            ratios[family][grid].append(ratio.cpu())

    report = {}
    for family, row in sorted(sums.items()):
        report[family] = {
            "elements": int(row["elements"]),
            "changed_share": row["changed"] / row["elements"],
            "erased_share_of_changed": row["erased"] / max(row["changed"], 1),
            "crossed_share": row["crossed"] / row["elements"],
            "rms_delta": (row["sum_sq_delta"] / row["elements"]) ** 0.5,
            **{f"median_delta_over_bin_{grid}": quantile(r, 0.5) for grid, r in ratios[family].items()},
            **{f"p90_delta_over_bin_{grid}": quantile(r, 0.9) for grid, r in ratios[family].items()},
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
