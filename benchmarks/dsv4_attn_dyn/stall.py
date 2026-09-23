"""Time the host stall of the DSv4 sparse attention op on the first call at each gather width.

Set `TILELANG_CACHE_DIR` before launching: an empty directory measures a cold compile, and a
second process pointed at the same directory measures a warm disk-cache load. Within one process
the second call at a width measures the in-process memo hit.
"""

import argparse
import json
import time

import torch

from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn

HEADS = 64
DIM = 512


def make_inputs(seq_len: int, n_kv: int, width: int, device: torch.device):
    q = torch.randn(1, seq_len, HEADS, DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(1, n_kv, 1, DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.randint(0, n_kv, (1, seq_len, 1, width), device=device, dtype=torch.int32)
    indices[..., width // 2 :] = -1
    sinks = torch.zeros(HEADS, device=device, dtype=torch.float32, requires_grad=True)
    return q, kv, indices, sinks


def timed(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("widths", type=int, nargs="+")
    parser.add_argument("--seq-len", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    warm_q, warm_kv, warm_idx, warm_sinks = make_inputs(256, 256, 64, device)
    dsv4_sparse_attn(warm_q, warm_kv, warm_idx, warm_sinks)[0].sum().backward()

    rows = []
    for width in args.widths:
        q, kv, indices, sinks = make_inputs(args.seq_len, args.seq_len, width, device)
        state = {}

        def fwd():
            state["out"] = dsv4_sparse_attn(q, kv, indices, sinks)[0]

        def bwd():
            state["out"].backward(torch.ones_like(state["out"]))

        first_fwd = timed(fwd)
        first_bwd = timed(bwd)
        repeat_fwd = timed(fwd)
        repeat_bwd = timed(bwd)
        rows.append(
            dict(
                label=args.label,
                width=width,
                first_fwd_s=first_fwd,
                first_bwd_s=first_bwd,
                repeat_fwd_s=repeat_fwd,
                repeat_bwd_s=repeat_bwd,
            )
        )
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
