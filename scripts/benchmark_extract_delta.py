#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

from prime_rl.utils.delta import ModelDeltaManager, count_sparse_delta_values


def run_extract(label: str, extract: Callable[[Path], None], output_path: Path) -> float:
    start = time.perf_counter()
    extract(output_path)
    elapsed = time.perf_counter() - start
    size_mb = os.path.getsize(output_path) / (1024**2)
    nnz = count_sparse_delta_values(output_path)
    print(f"{label}: {elapsed:.2f}s, {size_mb:.2f} MB, nnz={nnz}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="path to the base model.safetensors")
    parser.add_argument("--target", required=True, help="path to the target model.safetensors")
    parser.add_argument("--out-dir", default="outputs/delta_benchmark")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manager = ModelDeltaManager()

    fused_times: list[float] = []
    sparse_times: list[float] = []
    for index in range(args.repeat):
        fused_path = out_dir / f"delta_fused_sparse_{index}.safetensors"
        sparse_path = out_dir / f"delta_sparse_varint_{index}.safetensors"

        fused_times.append(
            run_extract(
                f"[{index + 1}/{args.repeat}] extract_delta_fused_sparse",
                lambda output_path: manager.extract_delta_fused_sparse(
                    base_model_path=args.base,
                    finetuned_model_path=args.target,
                    delta_output_path=output_path,
                    threshold=0.0,
                    include_bias=False,
                ),
                fused_path,
            )
        )
        sparse_times.append(
            run_extract(
                f"[{index + 1}/{args.repeat}] extract_sparse_delta",
                lambda output_path: manager.extract_sparse_delta(
                    base_model_path=args.base,
                    finetuned_model_path=args.target,
                    delta_output_path=output_path,
                ),
                sparse_path,
            )
        )

    if args.repeat > 1:
        print(
            f"avg fused: {sum(fused_times) / len(fused_times):.2f}s, "
            f"avg sparse: {sum(sparse_times) / len(sparse_times):.2f}s"
        )


if __name__ == "__main__":
    main()
