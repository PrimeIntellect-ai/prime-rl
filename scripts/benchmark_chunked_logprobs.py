#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from prime_rl.trainer.rl.loss import compute_entropy


def _online_logsumexp_and_weighted_update(
    m: torch.Tensor, s: torch.Tensor, t: torch.Tensor, chunk_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Online logsumexp + weighted-sum accumulator for entropy.

    Maintains:
      m: running max
      s: running sum(exp(x - m))
      t: running sum(exp(x - m) * x)
    """
    chunk_m = torch.amax(chunk_logits, dim=-1)  # [N]
    m_new = torch.maximum(m, chunk_m)  # [N]
    exp_old = torch.exp(m - m_new)

    chunk_exp = torch.exp(chunk_logits - m_new.unsqueeze(-1))  # [N, C]
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    t_new = t * exp_old + (chunk_exp * chunk_logits).sum(dim=-1)
    return m_new, s_new, t_new


class _ChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: float,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (per-token logprobs, per-token entropy) without materializing [N, V].

        Important: entropy is computed from the *same* per-chunk logits used for the softmax
        normalization (no extra W @ hidden matmul).
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert chunk_size > 0

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]

        # Running stats in fp32.
        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)
        t = torch.zeros((n,), device=device, dtype=torch.float32)
        target_logits = torch.empty((n,), device=device, dtype=torch.float32)

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]
            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32).mul_(inv_temperature)  # [N, C] fp32

            # Shared intermediates for logZ and entropy stats.
            m, s, t = _online_logsumexp_and_weighted_update(m, s, t, logits_f)

            # Fill target logits for labels that fall in this chunk.
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                target_logits[mask] = logits_f[mask, idx]

        logz = m + torch.log(s)
        logprobs = target_logits - logz
        entropy = logz - (t / s)

        # Save for backward (recompute logits per chunk for grad)
        ctx.save_for_backward(hidden, weight, labels, logz)
        ctx.inv_temperature = inv_temperature
        ctx.chunk_size = chunk_size

        # Return fp32 for numerical stability (matching baseline behavior).
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):  # type: ignore[override]
        # Entropy is treated as a detached metric in this PoC (no gradient support).
        if grad_entropy is not None:
            # If someone wires entropy into the loss accidentally, fail loudly.
            if torch.any(grad_entropy != 0):
                raise RuntimeError("PoC: entropy output is non-differentiable (grad_entropy must be zero/unused).")

        hidden, weight, labels, logz = ctx.saved_tensors
        inv_temperature: float = ctx.inv_temperature
        chunk_size: int = ctx.chunk_size

        n, h = hidden.shape
        vocab = weight.shape[0]

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        g = grad_logprobs.to(torch.float32)  # [N] fp32 for stable scaling

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]

            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32).mul_(inv_temperature)  # [N, C] fp32

            # p = softmax(logits_f) chunk = exp(logits_f - logz)
            p = torch.exp(logits_f - logz.unsqueeze(-1))  # [N, C] fp32

            # dL/dlogits = g * (1_{label} - p)
            grad_logits = (-g).unsqueeze(-1) * p  # [N, C] fp32
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                grad_logits[mask, idx] += g[mask]

            # Chain through temperature scaling: logits_f = logits * inv_temperature
            grad_logits.mul_(inv_temperature)

            grad_hidden.add_(grad_logits.to(hidden.dtype) @ w_chunk)
            grad_w_chunk = grad_logits.to(weight.dtype).t() @ hidden  # [C, H]
            grad_weight[start:end].add_(grad_w_chunk)

        return grad_hidden, grad_weight, None, None, None


def chunked_selective_log_softmax_and_entropy(
    hidden: torch.Tensor,  # [B, S, H] or [N, H]
    weight: torch.Tensor,  # [V, H]
    labels: torch.Tensor,  # [B, S] or [N]
    *,
    temperature: float = 1.0,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PoC: compute (logprobs, entropy) for labels under softmax(hidden @ weight^T / temperature),
    without materializing full logits [*, V], and without recomputing W@hidden for entropy.
    """
    inv_t = 1.0 / float(temperature)
    if hidden.dim() == 3:
        b, s, h = hidden.shape
        hidden_2d = hidden.reshape(b * s, h).contiguous()
        labels_1d = labels.reshape(b * s).contiguous()
        lp, ent = _ChunkedLogProbEntropyFn.apply(hidden_2d, weight, labels_1d, inv_t, int(chunk_size))
        return lp.reshape(b, s), ent.reshape(b, s)
    if hidden.dim() == 2:
        return _ChunkedLogProbEntropyFn.apply(hidden.contiguous(), weight, labels.contiguous(), inv_t, int(chunk_size))
    raise ValueError(f"expected hidden dim 2 or 3, got {hidden.dim()}")


class ChunkedLogprobsLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int, *, temperature: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return chunked_selective_log_softmax_and_entropy(
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

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: [B, S, H], labels: [B, S]
        logits = super().forward(hidden)  # [B, S, V] in model dtype (bf16/fp16)
        logits_f = logits.to(torch.float32).mul_(1.0 / float(self.temperature))
        lp_full = torch.log_softmax(logits_f, dim=-1)
        logprobs = lp_full.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        entropy = compute_entropy(logits_f)
        return logprobs, entropy


@dataclass
class BenchResult:
    name: str
    fwd_ms: float
    bwd_ms: float
    peak_mem_mb: float
    out_max_abs: float
    entropy_max_abs: float
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
        logprobs = out[0] if isinstance(out, tuple) else out
        logprobs.sum().backward()
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
        logprobs = out[0] if isinstance(out, tuple) else out
        _, bwd_ms = _events_timing(lambda: logprobs.sum().backward())
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
    logprobs = out[0] if isinstance(out, tuple) else out
    entropy = out[1] if isinstance(out, tuple) else None
    logprobs.sum().backward()
    out_max_abs = float(logprobs.detach().abs().max().item())
    entropy_max_abs = float(entropy.detach().abs().max().item()) if entropy is not None else float("nan")
    grad_hidden_max_abs = float(hidden.grad.detach().abs().max().item())
    grad_weight_max_abs = float(module.weight.grad.detach().abs().max().item())  # type: ignore[attr-defined]

    return (
        (logprobs.detach(), entropy.detach()) if entropy is not None else logprobs.detach(),
        hidden.grad.detach(),
        module.weight.grad.detach(),
        BenchResult(  # type: ignore[attr-defined]
            name=name,
            fwd_ms=float(fwd_ms_total / max(iters, 1)),
            bwd_ms=float(bwd_ms_total / max(iters, 1)),
            peak_mem_mb=float(peak_mem_mb),
            out_max_abs=out_max_abs,
            entropy_max_abs=entropy_max_abs,
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
    p.add_argument("--atol", type=float, default=1e-3, help="Abs tolerance for correctness assertions.")
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
        name="chunked_logprobs+entropy",
        module=chunked,
        hidden_init=hidden,
        labels=labels,
        warmup=args.warmup,
        iters=args.iters,
    )

    # Correctness checks (forward + backward)
    assert isinstance(out0, tuple) and isinstance(out1, tuple)
    lp0, ent0 = out0
    lp1, ent1 = out1

    out_max_err = float((lp0 - lp1).abs().max().item())
    ent_max_err = float((ent0 - ent1).abs().max().item())
    gh_max_err = float((gh0.float() - gh1.float()).abs().max().item())
    gw_max_err = float((gw0.float() - gw1.float()).abs().max().item())

    atol = float(args.atol)
    assert out_max_err <= atol, f"logprobs max abs error {out_max_err} > atol {atol}"
    assert ent_max_err <= atol, f"entropy max abs error {ent_max_err} > atol {atol}"
    assert gh_max_err <= atol, f"grad_hidden max abs error {gh_max_err} > atol {atol}"
    assert gw_max_err <= atol, f"grad_weight max abs error {gw_max_err} > atol {atol}"

    print("=== Shapes ===")
    print(f"hidden: {tuple(hidden.shape)} dtype={dtype} device={device}")
    print(f"weight: {tuple(baseline.weight.shape)} dtype={dtype}")
    print(f"labels: {tuple(labels.shape)}")
    print()
    print("=== Results ===")
    for r in (r0, r1):
        print(
            f"{r.name:24s} | fwd {r.fwd_ms:8.3f} ms | bwd {r.bwd_ms:8.3f} ms | "
            f"peak {r.peak_mem_mb:10.1f} MB | |out|max {r.out_max_abs:.6g} | |ent|max {r.entropy_max_abs:.6g}"
        )
    print()
    print("=== Correctness (max abs) ===")
    print(f"out:        {out_max_err:.6g}")
    print(f"entropy:    {ent_max_err:.6g}")
    print(f"grad_hidden:{gh_max_err:.6g}")
    print(f"grad_weight:{gw_max_err:.6g}")


if __name__ == "__main__":
    main()
