#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from prime_rl.trainer.rl.chunked_logprobs import chunked_selective_log_softmax


class ChunkedLogprobsLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int, *, temperature: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return chunked_selective_log_softmax(
            hidden,
            self.weight,
            labels,
            temperature=self.temperature,
            chunk_size=self.chunk_size,
        )


class BaselineLogprobsLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, temperature: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # hidden: [B, S, H], labels: [B, S]
        logits = super().forward(hidden)  # [B, S, V] in model dtype (bf16/fp16)
        logits_f = logits.to(torch.float32).mul_(1.0 / float(self.temperature))
        lp = torch.log_softmax(logits_f, dim=-1)
        return lp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


@dataclass
class BenchResult:
    name: str
    fwd_ms: float
    bwd_ms: float
    peak_mem_mb: float
    out_max_abs: float
    grad_hidden_max_abs: float
    grad_weight_max_abs: float


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _events_timing(fn):
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        return out, (t1 - t0) * 1000.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def run_one(
    *,
    name: str,
    module: torch.nn.Module,
    hidden_init: torch.Tensor,
    labels: torch.Tensor,
    warmup: int,
    iters: int,
):
    # Fresh tensors per run
    hidden = hidden_init.detach().clone().requires_grad_(True)
    module.train()

    def _zero_grads():
        hidden.grad = None
        module.zero_grad(set_to_none=True)

    # Warmup
    for _ in range(warmup):
        out = module(hidden, labels)
        out.sum().backward()
        _zero_grads()
    _sync()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _sync()

    fwd_ms_total = 0.0
    bwd_ms_total = 0.0
    out = None
    for _ in range(max(iters, 1)):
        # Forward
        out, fwd_ms = _events_timing(lambda: module(hidden, labels))
        fwd_ms_total += float(fwd_ms)

        # Backward
        _, bwd_ms = _events_timing(lambda: out.sum().backward())
        bwd_ms_total += float(bwd_ms)

        _zero_grads()
    assert out is not None

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # Basic sanity
    # Re-run one backward to populate grads for reporting
    _zero_grads()
    out = module(hidden, labels)
    out.sum().backward()
    out_max_abs = float(out.detach().abs().max().item())
    grad_hidden_max_abs = float(hidden.grad.detach().abs().max().item())
    grad_weight_max_abs = float(module.weight.grad.detach().abs().max().item())  # type: ignore[attr-defined]

    return (
        out.detach(),
        hidden.grad.detach(),
        module.weight.grad.detach(),
        BenchResult(  # type: ignore[attr-defined]
            name=name,
            fwd_ms=float(fwd_ms_total / max(iters, 1)),
            bwd_ms=float(bwd_ms_total / max(iters, 1)),
            peak_mem_mb=float(peak_mem_mb),
            out_max_abs=out_max_abs,
            grad_hidden_max_abs=grad_hidden_max_abs,
            grad_weight_max_abs=grad_weight_max_abs,
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b", type=int, default=2)
    p.add_argument("--s", type=int, default=1024, help="Sequence length (we benchmark S-1 predicted positions).")
    p.add_argument("--h", type=int, default=4096)
    p.add_argument("--v", type=int, default=131072)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    b, s, h, v = args.b, args.s, args.h, args.v
    s_pred = s - 1

    # Pred hidden predicts next tokens
    hidden = torch.randn(b, s_pred, h, device=device, dtype=dtype)
    labels = torch.randint(0, v, (b, s_pred), device=device, dtype=torch.long)
    # Chunked implementation currently supports weight-only (no bias), so benchmark with bias=False.
    linear = torch.nn.Linear(h, v, bias=False, device=device, dtype=dtype)
    baseline = BaselineLogprobsLinear(h, v, temperature=args.temp)
    baseline.weight = torch.nn.Parameter(linear.weight.detach().clone().requires_grad_(True))
    chunked = ChunkedLogprobsLinear(h, v, args.chunk, temperature=args.temp)
    chunked.weight = torch.nn.Parameter(linear.weight.detach().clone().requires_grad_(True))
    del linear

    out0, gh0, gw0, r0 = run_one(
        name="baseline_full_logits",
        module=baseline,
        hidden_init=hidden,
        labels=labels,
        warmup=args.warmup,
        iters=args.iters,
    )
    out1, gh1, gw1, r1 = run_one(
        name="chunked_online_logsumexp",
        module=chunked,
        hidden_init=hidden,
        labels=labels,
        warmup=args.warmup,
        iters=args.iters,
    )

    # Correctness checks (forward + backward)
    out_max_err = float((out0 - out1).abs().max().item())
    gh_max_err = float((gh0.float() - gh1.float()).abs().max().item())
    gw_max_err = float((gw0.float() - gw1.float()).abs().max().item())

    print("=== Shapes ===")
    print(f"hidden: {tuple(hidden.shape)} dtype={dtype} device={device}")
    print(f"weight: {tuple(baseline.weight.shape)} dtype={dtype}")
    print(f"labels: {tuple(labels.shape)}")
    print()
    print("=== Results ===")
    for r in (r0, r1):
        print(
            f"{r.name:24s} | fwd {r.fwd_ms:8.3f} ms | bwd {r.bwd_ms:8.3f} ms | "
            f"peak {r.peak_mem_mb:10.1f} MB | |out|max {r.out_max_abs:.6g}"
        )
    print()
    print("=== Correctness (max abs) ===")
    print(f"out:        {out_max_err:.6g}")
    print(f"grad_hidden:{gh_max_err:.6g}")
    print(f"grad_weight:{gw_max_err:.6g}")


if __name__ == "__main__":
    main()
