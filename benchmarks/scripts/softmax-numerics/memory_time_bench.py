"""Peak GPU memory and time of one LM-head forward+backward per path, for the softmax-numerics memory A/B. Run from
the checkout under test on one GPU, e.g.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --no-sync python \
        benchmarks/scripts/softmax-numerics/memory_time_bench.py --tag main --out main.jsonl
Qwen3-1.7B head shape: hidden 2048, vocab 151936, bf16 weights, token chunk 8192. Peaks are
torch.cuda.max_memory_allocated minus what was allocated before the step (hidden, weight, labels, support), reported
in GB and in units of one fp32 [token chunk, vocab] buffer (fused paths) or one fp32 [tokens, vocab] buffer
(non-fused paths).
"""

import argparse
import json

import torch

from prime_rl.trainer.models.layers import lm_head as LH
from prime_rl.trainer.rl import loss as RL

H, V, CHUNK, K = 2048, 151936, 8192, 20
FUSED_CASES = ["rl_full_vocab", "rl_support_not_replayed", "rl_replay_dense", "rl_replay_sparse", "sft_quack"]
FUSED_CASES += ["sft_fallback"]
VANILLA_CASES = ["vanilla_rl", "vanilla_rl_support", "vanilla_sft"]


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def peaks(clear_grads, forward, backward):
    """Forward peak and whole-step peak above what is allocated before the step, gradients cleared."""
    clear_grads()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loss = forward()
    torch.cuda.synchronize()
    forward_peak = torch.cuda.max_memory_allocated() - base
    backward(loss)
    torch.cuda.synchronize()
    return forward_peak, torch.cuda.max_memory_allocated() - base


def make_case(case, n):
    torch.manual_seed(0)
    hidden = (torch.randn(1, n, H, device="cuda") * 0.5).bfloat16().requires_grad_(True)
    labels = torch.randint(0, V, (1, n), device="cuda")
    temperature = torch.ones(1, n, device="cuda")
    ids = torch.cat([labels.view(-1, 1), torch.randint(0, V, (n, K - 1), device="cuda")], dim=1).int().view(-1)
    offsets = torch.arange(n + 1) * K

    def support(replayed, sparse):
        return LH.TokenSupport(
            ids=ids, offsets=offsets, replayed=torch.full((n,), replayed, device="cuda"), sparse_backward=sparse
        )

    if case in VANILLA_CASES:
        head = LH.VanillaOutputLinear(H, V).cuda().bfloat16()
    else:
        head = LH.FusedOutputLinear(H, V, chunk_size=CHUNK).cuda().bfloat16()
    with torch.no_grad():
        head.weight.normal_(0, 0.02)
    sup = {
        "rl_support_not_replayed": support(False, False),
        "rl_replay_dense": support(True, False),
        "rl_replay_sparse": support(True, True),
        "vanilla_rl_support": support(True, False),
    }.get(case)

    def clear_grads():
        hidden.grad = head.weight.grad = None

    def forward():
        clear_grads()
        if case.startswith("rl_"):
            out = head(hidden, labels, temperature=temperature, support=sup)
            loss = out["logprobs"].sum()
            return loss + out["support_logprobs"].sum() * 0.5 if sup is not None else loss
        if case.startswith("sft_") or case == "vanilla_sft":
            return head(hidden, labels)["loss"]
        # Mirrors train.py's VanillaOutputLinear branch
        logits = head(hidden)["logits"]
        scaled_logits = logits / temperature.unsqueeze(-1)
        if sup is not None:
            logprobs, support_logprobs = RL.selective_log_softmax_with_support(scaled_logits, labels, sup)
            loss = logprobs.sum() + support_logprobs.sum() * 0.5
        else:
            loss = RL.selective_log_softmax(scaled_logits, labels).sum()
        RL.compute_entropy(scaled_logits)
        return loss

    def backward(loss):
        loss.backward()

    return clear_grads, forward, backward


def run_case(case, n, args):
    clear_grads, forward, backward = make_case(case, n)
    unit = 4 * V * (n if case in VANILLA_CASES else min(CHUNK, n))
    try:
        ms = time_ms(lambda: backward(forward()), args.warmup, args.iters)
        forward_peak, step_peak = peaks(clear_grads, forward, backward)
        row = {"ms": round(ms, 2), "fwd_peak_gb": round(forward_peak / 1e9, 3), "peak_gb": round(step_peak / 1e9, 3)}
        row |= {"fwd_peak_units": round(forward_peak / unit, 2), "peak_units": round(step_peak / unit, 2)}
    except torch.OutOfMemoryError:
        row = {"oom": True}
    del clear_grads, forward, backward
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16384, 32768, 65536])
    parser.add_argument("--cases", nargs="+", default=FUSED_CASES + VANILLA_CASES)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()
    # Checkouts without quack's kernel (this PR) run sft_fallback only
    supported = getattr(LH, "_quack_cross_entropy_supported", None)
    quack = supported is not None and supported(torch.empty(1, device="cuda"))
    for n in args.tokens:
        for case in args.cases:
            if case == "sft_quack" and not quack:
                continue
            if supported is not None:
                LH._quack_cross_entropy_supported = (lambda _: False) if case == "sft_fallback" else supported
            row = {"tag": args.tag, "gpu": torch.cuda.get_device_name(), "tokens": n, "case": case}
            row |= run_case(case, n, args)
            print(json.dumps(row), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
