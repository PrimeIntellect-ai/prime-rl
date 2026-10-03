"""Microbenchmark + correctness check for the DSA sparse MLA kernels (GLM-5 / DeepSeek-V3.2 shapes).

Usage:
    uv run python benchmarks/scripts/bench_sparse_mla.py
    uv run python benchmarks/scripts/bench_sparse_mla.py --seqlens 8192 16384 --check-seqlen 4096
"""

import argparse
import math

import torch

from prime_rl.trainer.models.kernels import sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla

H = 64
D_QK = 576
D_V = 512


def make_indices(seqlen: int, topk: int, doc_len: int, device: str, generator: torch.Generator) -> torch.Tensor:
    """Causal, document-masked top-k indices in the layout produced by the indexer.

    Tokens with fewer than `topk` visible keys get every visible key plus the sentinel
    (`seqlen`, i.e. the extra zero KV row) for the remaining slots; later tokens get a
    random subset of their visible keys. Order inside a row is random.
    """
    sentinel = seqlen
    positions = torch.arange(seqlen, device=device)
    doc_start = (positions // doc_len) * doc_len
    indices = torch.empty(seqlen, topk, dtype=torch.int32, device=device)
    chunk = 512
    for start in range(0, seqlen, chunk):
        rows = positions[start : start + chunk]
        scores = torch.rand(len(rows), seqlen, device=device, generator=generator)
        visible = (positions[None, :] <= rows[:, None]) & (positions[None, :] >= doc_start[rows][:, None])
        scores = torch.where(visible, scores, -1.0)
        top_scores, top_idx = scores.topk(topk, dim=-1)
        top_idx = torch.where(top_scores >= 0, top_idx, sentinel)
        perm = torch.rand(len(rows), topk, device=device, generator=generator).argsort(dim=-1)
        indices[start : start + chunk] = top_idx.gather(1, perm).int()
    return indices.view(1, seqlen, 1, topk)


def make_inputs(seqlen: int, topk: int, doc_len: int, seed: int = 0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, seqlen, H, D_QK, device=device, dtype=torch.bfloat16, generator=g)
    kv = torch.randn(1, seqlen + 1, 1, D_QK, device=device, dtype=torch.bfloat16, generator=g)
    kv[:, -1] = 0
    grad_out = torch.randn(1, seqlen, H, D_V, device=device, dtype=torch.bfloat16, generator=g)
    indices = make_indices(seqlen, topk, doc_len, device, g)
    return q, kv, grad_out, indices


def reference_grads(q, kv, grad_out, indices, sm_scale, chunk: int = 128):
    """fp32 autograd reference: gather KV by index, mask sentinel, softmax, matmuls."""
    _, seqlen, _, _ = q.shape
    sentinel = kv.shape[1] - 1
    kv32 = kv[0, :, 0].float().requires_grad_(True)
    dq = torch.empty(seqlen, H, D_QK, device=q.device, dtype=torch.float32)
    for start in range(0, seqlen, chunk):
        idx = indices[0, start : start + chunk, 0].long()
        q32 = q[0, start : start + chunk].float().requires_grad_(True)
        kv_g = kv32[idx]  # [c, topk, 576]
        scores = torch.einsum("chd,ckd->chk", q32, kv_g) * sm_scale
        scores = scores.masked_fill((idx == sentinel)[:, None, :], float("-inf"))
        probs = scores.softmax(dim=-1)
        out = torch.einsum("chk,ckd->chd", probs, kv_g[..., :D_V])
        out.backward(grad_out[0, start : start + chunk].float())
        dq[start : start + chunk] = q32.grad
    dkv = kv32.grad.clone()
    dkv[sentinel] = 0
    return dq.unsqueeze(0), dkv.view(1, -1, 1, D_QK)


def compare(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    got, ref = got.float(), ref.float()
    abs_err = (got - ref).abs().max().item()
    rel_err = ((got - ref).norm() / ref.norm()).item()
    print(f"  {name}: max_abs={abs_err:.3e} ref_max={ref.abs().max().item():.3e} rel_l2={rel_err:.3e}")
    assert rel_err < 1e-2, f"{name} rel_l2 {rel_err} too large"


def time_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def backends():
    return {"tilelang": sparse_mla_bwd.tilelang_sparse_mla_backward, "default": sparse_mla_bwd.sparse_mla_backward}


def print_device() -> None:
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())


def check(seqlen: int, topk: int, doc_len: int) -> None:
    sm_scale = D_QK**-0.5
    q, kv, grad_out, indices = make_inputs(seqlen, topk, doc_len, seed=1)
    out, lse = sparse_mla(q, kv, indices, sm_scale)
    dq_ref, dkv_ref = reference_grads(q, kv, grad_out, indices, sm_scale)
    for name, fn in backends().items():
        print(f"[check S={seqlen} topk={topk} doc_len={doc_len}] {name}")
        dq, dkv = fn(q, kv, out, grad_out, indices, lse, sm_scale)
        compare("dQ", dq, dq_ref)
        compare("dKV", dkv, dkv_ref)


def check_batched_cp_shard(seqlen: int, topk: int, doc_len: int) -> None:
    """B=2 with a CP-style query shard (second half of the queries, full KV): every backend vs tilelang."""
    sm_scale = D_QK**-0.5
    inputs = [make_inputs(seqlen, topk, doc_len, seed=seed) for seed in (2, 3)]
    half = seqlen // 2
    q, kv, grad_out, indices = (torch.cat(ts).contiguous() for ts in zip(*inputs))
    q, grad_out, indices = (t[:, half:].contiguous() for t in (q, grad_out, indices))
    out, lse = sparse_mla(q, kv, indices, sm_scale)
    dq_ref, dkv_ref = sparse_mla_bwd.tilelang_sparse_mla_backward(q, kv, out, grad_out, indices, lse, sm_scale)
    for name, fn in backends().items():
        print(f"[check B=2 S_q={seqlen - half} S_kv={seqlen + 1}] {name} vs tilelang")
        dq, dkv = fn(q, kv, out, grad_out, indices, lse, sm_scale)
        compare("dQ", dq, dq_ref)
        compare("dKV", dkv, dkv_ref)


def bench(seqlen: int, topk: int, doc_len: int) -> None:
    sm_scale = D_QK**-0.5
    q, kv, grad_out, indices = make_inputs(seqlen, topk, doc_len)
    fwd_flops = 2 * seqlen * H * topk * (D_QK + D_V)
    bwd_flops = 2.5 * fwd_flops

    fwd_ms = time_ms(lambda: sparse_mla(q, kv, indices, sm_scale))
    print(
        f"[bench S={seqlen} topk={topk} doc_len={doc_len}] fwd: {fwd_ms:.3f} ms {fwd_flops / fwd_ms / 1e9:.1f} TFLOP/s"
    )
    out, lse = sparse_mla(q, kv, indices, sm_scale)
    for name, fn in backends().items():
        bwd_ms = time_ms(lambda: fn(q, kv, out, grad_out, indices, lse, sm_scale))
        print(f"  bwd {name}: {bwd_ms:.3f} ms {bwd_flops / bwd_ms / 1e9:.1f} TFLOP/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlens", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--doc-len", type=int, default=None, help="packed document length (default: one document)")
    parser.add_argument("--check-seqlen", type=int, default=4096)
    parser.add_argument("--check-doc-len", type=int, default=1536)
    args = parser.parse_args()
    print_device()

    if args.check_seqlen > 0:
        check(args.check_seqlen, args.topk, args.check_doc_len)
        check_batched_cp_shard(args.check_seqlen, args.topk, args.check_doc_len)
    for seqlen in args.seqlens:
        bench(seqlen, args.topk, args.doc_len or seqlen)
        if args.doc_len is None:
            bench(seqlen, args.topk, math.gcd(seqlen, 4096))


if __name__ == "__main__":
    main()
