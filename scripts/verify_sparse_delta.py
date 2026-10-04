#!/usr/bin/env python3
from __future__ import annotations

import argparse

from prime_rl.utils.delta import verify_sparse_delta_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="path to the base model.safetensors")
    parser.add_argument("--target", required=True, help="path to the target model.safetensors")
    parser.add_argument("--delta", required=True, help="path to the sparse delta.safetensors")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--max-report", type=int, default=10)
    parser.add_argument("--include-bias", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    result = verify_sparse_delta_file(
        args.base,
        args.target,
        args.delta,
        atol=args.atol,
        rtol=args.rtol,
        max_report=args.max_report,
        include_bias=args.include_bias,
    )
    if result.ok:
        print(f"OK: all tensors matched. max_diff={result.max_diff:.3e}")
        return

    print("FAILED: mismatched tensors:")
    for mismatch in result.mismatches:
        print(f"  {mismatch.name}: max_diff={mismatch.max_diff:.3e}")
    print(f"max_diff={result.max_diff:.3e}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
