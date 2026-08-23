"""Convert a DCP trainer checkpoint into blockwise-FP8 HF weights.

Chains ``dcp_to_bf16`` and ``bf16_to_fp8``: the bf16 export lands at
``<ckpt_dir>/weights`` and the fp8 dir next to it.

Usage (from the prime-rl repo; multi-rank speeds up the bf16 export):
    uv run python tools/converters/dcp_to_fp8.py <run>/checkpoints/step_{n} [output_dir]
    uv run torchrun --nproc-per-node 8 tools/converters/dcp_to_fp8.py \
        <run>/checkpoints/step_{n} [output_dir]

Writes to ``<ckpt_dir>/weights-FP8`` by default.
"""

import argparse
from pathlib import Path

from bf16_to_fp8 import convert as bf16_to_fp8
from dcp_to_bf16 import convert as dcp_to_bf16

from prime_rl.trainer.world import get_world


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <ckpt_dir>/weights-FP8")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    bf16_dir = dcp_to_bf16(args.ckpt_dir)
    if get_world().is_master:
        bf16_to_fp8(bf16_dir, args.output_dir, args.block_size)


if __name__ == "__main__":
    main()
