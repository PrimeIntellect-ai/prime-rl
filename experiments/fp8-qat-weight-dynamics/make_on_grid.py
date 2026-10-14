"""Snap the trainer-FP8-covered weights of a Qwen3-MoE HF snapshot onto the UE8M0 128x128 grid.

Usage (on a GPU node, from the repo root):
    uv run --no-sync python experiments/fp8-qat-weight-dynamics/make_on_grid.py <src_snapshot> <out_dir>
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prime_rl.trainer.models.kernels.fp8_utils import per_block_cast_to_fp8_triton

BLOCK = 128
QUANTIZED_NAME = re.compile(
    r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight$"
)
SMALL_AMAX = 448.0 * 2.0**-14


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = q.shape
    blocks = q.float().view(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK) * scale[:, None, :, None]
    return blocks.view(rows, cols).to(torch.bfloat16)


def snap_to_ue8m0(w: torch.Tensor) -> torch.Tensor:
    assert w.dim() == 2 and w.shape[0] % BLOCK == 0 and w.shape[1] % BLOCK == 0, w.shape
    return dequantize(*per_block_cast_to_fp8_triton(w, True, BLOCK))


def block_amax(w: torch.Tensor) -> torch.Tensor:
    rows, cols = w.shape
    return w.float().abs().view(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK).amax(dim=(1, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.out / "prime", ignore_errors=True)
    shutil.rmtree(args.out / "hf", ignore_errors=True)
    for path in args.src.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copyfile(path, args.out / path.name)

    stats = dict(tensors=0, elements=0, changed=0, blocks=0, small_amax_blocks=0, roundtrip_failures=0)
    sum_sq_delta = 0.0
    sum_sq_w = 0.0
    for shard in sorted(args.src.glob("*.safetensors")):
        out_tensors = {}
        with safe_open(shard, framework="pt", device="cuda") as f:
            metadata = f.metadata()
            for name in f.keys():
                w = f.get_tensor(name)
                if not QUANTIZED_NAME.match(name):
                    out_tensors[name] = w.cpu()
                    continue
                assert w.dtype == torch.bfloat16, (name, w.dtype)
                w_on = snap_to_ue8m0(w)
                stats["roundtrip_failures"] += int((snap_to_ue8m0(w_on) != w_on).sum())
                amax = block_amax(w)
                stats["blocks"] += amax.numel()
                stats["small_amax_blocks"] += int((amax < SMALL_AMAX).sum())
                stats["tensors"] += 1
                stats["elements"] += w.numel()
                stats["changed"] += int((w_on != w).sum())
                sum_sq_delta += float((w_on.float() - w.float()).square().sum())
                sum_sq_w += float(w.float().square().sum())
                out_tensors[name] = w_on.cpu()
        save_file(out_tensors, args.out / shard.name, metadata=metadata)
        print(f"{shard.name}: {len(out_tensors)} tensors", flush=True)

    untouched_mismatches = 0
    for shard in sorted(args.src.glob("*.safetensors")):
        with (
            safe_open(shard, framework="pt", device="cpu") as src,
            safe_open(args.out / shard.name, framework="pt", device="cpu") as out,
        ):
            assert set(src.keys()) == set(out.keys()), shard.name
            for name in src.keys():
                if not QUANTIZED_NAME.match(name) and not torch.equal(src.get_tensor(name), out.get_tensor(name)):
                    untouched_mismatches += 1

    report = {
        **stats,
        "untouched_mismatches": untouched_mismatches,
        "changed_share": stats["changed"] / stats["elements"],
        "small_amax_block_share": stats["small_amax_blocks"] / stats["blocks"],
        "relative_rms_delta": (sum_sq_delta / sum_sq_w) ** 0.5,
    }
    (args.out / "on_grid_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert stats["roundtrip_failures"] == 0 and untouched_mismatches == 0


if __name__ == "__main__":
    main()
