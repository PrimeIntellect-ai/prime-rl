"""What a wider slot width costs the cudnn_flashmla backend: its first-call time (a cuDNN compile per new width)
and its steady forward+backward time, on corpus items forced to run at each width.

usage (repo root, on an otherwise idle GPU):
  uv run --no-sync python benchmarks/scripts/dsv4_sparse_attn/slot_width_cost.py [ITEM_ID ...]
"""

import argparse
import time

import torch
from common import DEFAULT_CORPUS_DIR, SM_SCALE, load_indices, load_manifest, make_inputs, select_items

from prime_rl.trainer.models.kernels.deepseek_v4 import dsv4_sparse_attn as ops

WIDTHS = (128, 256, 384, 512, 640, 768, 1024, 1536, 2048)
TIMED_CALLS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "items", nargs="*", default=["short-16384-hca-cp1", "single-16384-csa-cp1", "heavy-65536-hca-cp1"]
    )
    args = parser.parse_args()
    for item in select_items(load_manifest(DEFAULT_CORPUS_DIR), args.items):
        indices = load_indices(DEFAULT_CORPUS_DIR, item)
        inputs = make_inputs(item, requires_grad=True)
        for width in (width for width in WIDTHS if width >= item.n_slots):
            ops.cudnn_flashmla_slot_width = lambda _n_slots, width=width: width

            def step():
                out, _ = ops.dsv4_sparse_attn(
                    inputs["q"], inputs["kv"], indices, inputs["sinks"], SM_SCALE, backend="cudnn_flashmla"
                )
                return torch.autograd.grad(out, (inputs["q"], inputs["kv"], inputs["sinks"]), inputs["grad_out"])

            torch.cuda.synchronize()
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            first_call_s = time.perf_counter() - start
            times_us = []
            for _ in range(TIMED_CALLS):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                step()
                end.record()
                torch.cuda.synchronize()
                times_us.append(begin.elapsed_time(end) * 1e3)
            median_us = sorted(times_us)[TIMED_CALLS // 2]
            print(
                f"{item.id} n_slots={item.n_slots} width={width}: first call {first_call_s:.2f} s, "
                f"fwd+bwd median {median_us:.0f} us",
                flush=True,
            )


if __name__ == "__main__":
    main()
