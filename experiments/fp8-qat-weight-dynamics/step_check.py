"""Weight-shift statistics (see weight_shift_stats.py) of an fp32 master export against its start snapshot.

Usage (on a GPU node, from the repo root):
    uv run --no-sync python experiments/fp8-qat-weight-dynamics/step_check.py <start_snapshot> <export_dir> <out.json>
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from weight_shift_stats import WeightShiftStats, family_of, report


def tensor_locations(directory: Path) -> dict[str, Path]:
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    return {name: directory / shard for name, shard in weight_map.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    start_files = tensor_locations(args.start)
    export_files = tensor_locations(args.export)
    stats = WeightShiftStats()
    for index, name in enumerate(sorted(export_files)):
        family = family_of(name)
        if family is None:
            continue
        with safe_open(start_files[name], framework="pt", device="cuda") as f:
            start = f.get_tensor(name)
        with safe_open(export_files[name], framework="pt", device="cuda") as f:
            master = f.get_tensor(name)
        stats.add(family, start, master, seed_offset=index)
    args.out.write_text(json.dumps(report(stats), indent=2))


if __name__ == "__main__":
    main()
