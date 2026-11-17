"""Precision of every prime-rl logprob and cross-entropy site against an fp64 reference, near probability 1.

Run from the prime-rl checkout under test, on one GPU:
    uv run --no-sync python benchmarks/scripts/softmax-numerics/precision_sweep.py --tag branch --out branch.jsonl

One token per call. Scaled logits: the label sits at z_max, the other V-1 entries at z_max - c - u_j with
u_j ~ U(0, 4) and c chosen so the effective margin delta = -LSE_{j != t}(s_j - s_t) hits a target. The reference
is computed in fp64 from the fp32 scaled logits each site actually sees, with log1p so it cannot collapse.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from prime_rl.inference.vllm import paired_capture
from prime_rl.trainer.models.layers import lm_head as LH
from prime_rl.trainer.rl import loss as RL

V = 32768
HEAD_K = 20
DELTAS = [2, 4, 6, 8, 10, 11, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 18, 20, 25, 30, 40, 60, 80, 100]
Z_MAXES = [0.0, 8.0, 20.0, 40.0]
TEMPERATURES = [1.0, 0.6]
POSITIONS = {"chunk0": 5, "chunk3": V - 7}


def rel(a: float, b: float) -> float:
    if b == 0:
        return 0.0 if a == 0 else math.inf
    return abs(a - b) / abs(b)


def desired_scaled(z_max: float, delta: float, label: int, u: torch.Tensor) -> torch.Tensor:
    others = torch.cat([u[:label], u[label + 1 :]])
    c = delta + torch.logsumexp(-others, 0).item()
    s = z_max - c - u
    s[label] = z_max
    return s


def reference(scaled: torch.Tensor, label: int, support: torch.Tensor | None = None) -> dict:
    """fp64 logprob, logit gradient of logp_t, and entropy over ``support`` (all ids when None)."""
    s = scaled.double().cpu()
    keep = torch.ones_like(s, dtype=torch.bool) if support is None else torch.zeros_like(s, dtype=torch.bool)
    if support is not None:
        keep[support.cpu().long()] = True
    d = torch.where(keep, s - s[label], -math.inf)
    d[label] = -math.inf
    lse_rest = torch.logsumexp(d, 0)
    log1p_rest = torch.log1p(torch.exp(lse_rest))
    logp = torch.where(keep, d - log1p_rest, -math.inf)
    logp[label] = -log1p_rest
    p = torch.exp(logp)
    grad = -p
    one_minus_pt = (torch.exp(lse_rest) / (1 + torch.exp(lse_rest))).item()
    grad[label] = one_minus_pt
    ent = -(torch.where(keep, p * logp, 0.0)).sum().item()
    return dict(
        delta=-lse_rest.item(), logp=logp[label].item(), gt=one_minus_pt, grad=grad, ent=ent, logp_all=logp, p=p
    )


def summarize_grad(grad_hat: torch.Tensor, ref: dict, label: int) -> dict:
    g = grad_hat.double().cpu()
    total = 2 * ref["gt"]
    return dict(
        gt_hat=g[label].item(),
        gt_rel=rel(g[label].item(), ref["gt"]),
        zero_sum=abs(g.sum().item()) / total if total > 0 else 0.0,
    )


def fused_head(raw: torch.Tensor, label: int, temperature: float, support_ids=None, replayed=False, sparse=False):
    head = LH.FusedOutputLinear(1, V, chunk_size=1).cuda().float()
    with torch.no_grad():
        head.weight.copy_(raw.view(V, 1))
    hidden = torch.ones(1, 1, 1, device="cuda")
    labels = torch.tensor([[label]], device="cuda")
    temp = torch.full((1, 1), temperature, device="cuda")
    support = None
    if support_ids is not None:
        support = LH.TokenSupport(
            ids=support_ids.int().cuda(),
            offsets=torch.tensor([0, support_ids.numel()]),
            replayed=torch.tensor([replayed], device="cuda"),
            sparse_backward=sparse,
        )
    out = head(hidden, labels, temperature=temp, support=support)
    out["logprobs"].sum().backward()
    inv_t = 1.0 / temp.reshape(1)
    scaled = raw.cuda() * inv_t
    grad_scaled = head.weight.grad.view(V).double() / inv_t.double()
    return out, scaled, grad_scaled


def autograd_site(fn, scaled: torch.Tensor):
    leaf = scaled.detach().clone().requires_grad_(True)
    logp = fn(leaf)
    logp.sum().backward()
    return logp.detach(), leaf.grad


def pack_sampler_head(scaled: torch.Tensor, label: int) -> torch.Tensor:
    """The real ``pack_sampler_rows`` on one request: logprobs of [sampled, top HEAD_K - 1 ids...]."""
    import numpy as np

    _, packed, _ = paired_capture.pack_sampler_rows(
        scaled.view(1, V),
        torch.tensor([0, 1], device="cuda"),
        torch.tensor([1], device="cuda"),
        torch.tensor([label], device="cuda"),
        np.array([HEAD_K]),
    )
    bits = (packed >> 32).to(torch.int32) ^ paired_capture.NEG_INF_BITS
    values = bits.view(torch.float32).view(-1)
    return values


def run_point(z_max, delta, temperature, pos, label, u, rows, tag):
    s_desired = desired_scaled(z_max, delta, label, u)
    raw = (s_desired * temperature).float().cuda()
    t_tensor = torch.tensor(temperature, device="cuda")
    base = dict(tag=tag, z_max=z_max, delta_target=delta, temperature=temperature, pos=pos)

    def emit(site, scaled, logp_hat, grad_hat=None, ent_hat=None, support=None):
        ref = reference(scaled, label, support)
        rec = dict(base, site=site, delta=ref["delta"], logp_ref=ref["logp"], logp_hat=float(logp_hat), gt_ref=ref["gt"])
        rec["logp_rel"] = rel(float(logp_hat), ref["logp"])
        if grad_hat is not None:
            rec.update(summarize_grad(grad_hat, ref, label))
        if ent_hat is not None:
            rec["ent_ref"] = ref["ent"]
            rec["ent_hat"] = float(ent_hat)
            rec["ent_rel"] = rel(float(ent_hat), ref["ent"])
        rows.append(rec)
        return ref

    # Fused RL head, full vocab
    out, scaled, g = fused_head(raw, label, temperature)
    emit("fused_head", scaled, out["logprobs"].item(), g, out["entropy"].item())

    # Fused RL head, replayed support (label + 31 random ids), dense and sparse backward
    gen = torch.Generator().manual_seed(1)
    others = torch.randperm(V, generator=gen)[:40]
    others = others[others != label][:31]
    support_ids = torch.cat([torch.tensor([label]), others])
    for sparse in (False, True):
        out, scaled, g = fused_head(raw, label, temperature, support_ids, replayed=True, sparse=sparse)
        emit(f"fused_head_replay{'_sparse' if sparse else ''}", scaled, out["logprobs"].item(), g, support=support_ids)

    # Vanilla RL head
    scaled_v = raw / t_tensor
    idx = torch.tensor([[label]], device="cuda")
    logp, g = autograd_site(lambda x: RL.selective_log_softmax(x.view(1, 1, V), idx), scaled_v)
    ent = RL.compute_entropy(scaled_v.view(1, 1, V)).item()
    emit("selective_log_softmax", scaled_v, logp.item(), g, ent)

    sup = LH.TokenSupport(
        ids=support_ids.int().cuda(), offsets=torch.tensor([0, support_ids.numel()]), replayed=torch.tensor([True], device="cuda")
    )
    logp, g = autograd_site(lambda x: RL.selective_log_softmax_with_support(x.view(1, 1, V), idx, sup)[0], scaled_v)
    emit("selective_log_softmax_replay", scaled_v, logp.item(), g, support=support_ids)

    # Sampler forms: vLLM's compute_logprobs (log_softmax) and prime-rl's pack_sampler_rows (gather - logsumexp)
    emit("vllm_log_softmax", scaled_v, scaled_v.log_softmax(-1)[label].item())
    emit("pack_sampler_rows", scaled_v, pack_sampler_head(scaled_v, label)[0].item())

    if temperature == 1.0:
        # SFT cross-entropy: fallback, quack, eager F.cross_entropy (logp = -loss, dlogp = -dloss)
        labels1 = torch.tensor([label], device="cuda")
        for name, force_fallback in (("sft_cross_entropy_fallback", True), ("sft_cross_entropy_quack", False)):
            logits = raw.view(1, V).clone()
            # Checkouts without quack's kernel (this PR) only have the fallback
            orig = getattr(LH, "_quack_cross_entropy_supported", None)
            if orig is None and not force_fallback:
                continue
            if not force_fallback and not orig(logits):
                continue
            if orig is not None and force_fallback:
                LH._quack_cross_entropy_supported = lambda _: False
            try:
                loss = torch.empty(1, device="cuda")
                grad = LH._cross_entropy(logits, labels1, loss, True)
            finally:
                if orig is not None:
                    LH._quack_cross_entropy_supported = orig
            emit(name, raw, -loss.item(), -grad.view(V))
        logp, g = autograd_site(lambda z: -F.cross_entropy(z.view(1, V), labels1, reduction="sum"), raw)
        emit("sft_F_cross_entropy", raw, logp.item(), g)
        if hasattr(LH, "cross_entropy_sum"):
            logp, g = autograd_site(lambda z: -LH.cross_entropy_sum(z.view(1, 1, V), labels1.view(1, 1)), raw)
            emit("sft_cross_entropy_sum", raw, logp.item(), g)

    # Proposed forms (autograd), for comparison
    def softplus_form(z):
        t = z[label]
        rest = z.masked_fill(torch.arange(V, device=z.device) == label, -math.inf).logsumexp(-1)
        return -F.softplus(rest - t)

    def split_form(z):
        m = z.detach().max()
        return (z[label] - m) - (z - m).exp().sum().log()

    logp, g = autograd_site(softplus_form, scaled_v)
    emit("proposed_softplus", scaled_v, logp.item(), g)
    logp, g = autograd_site(split_form, scaled_v)
    emit("proposed_split", scaled_v, logp.item(), g)

    # IcePop tail mass: head = top-K ids by scaled logit (label included), tail = 1 - sum(head probs)
    head_ids = scaled_v.topk(HEAD_K).indices
    ref = reference(scaled_v, label)
    keep = torch.ones(V, dtype=torch.bool)
    keep[head_ids.cpu()] = False
    tail_ref = ref["p"][keep].sum().item()
    rows_idx = torch.zeros(HEAD_K, dtype=torch.long, device="cuda")
    trainer_out, _, _ = fused_head(raw, label, temperature, head_ids.cpu(), replayed=False)
    sources = {
        "trainer_fused_head": trainer_out["support_logprobs"].detach(),
        "sampler_vllm_log_softmax": scaled_v.log_softmax(-1)[head_ids],
        "sampler_pack_rows": pack_sampler_head(scaled_v, label)[1:],
    }
    helper = getattr(RL, "_segment_complement_mass", None)
    for src, head_lp in sources.items():
        old = (1.0 - torch.zeros(1, device="cuda").index_add(0, rows_idx, head_lp.exp())).item()
        rec = dict(base, site=f"icepop_tail/{src}", delta=ref["delta"], tail_ref=tail_ref, tail_old=old)
        rec["tail_old_rel"] = rel(old, tail_ref)
        if helper is not None:
            new = helper(head_lp, rows_idx, 1).item()
            rec["tail_new"] = new
            rec["tail_new_rel"] = rel(new, tail_ref)
        rows.append(rec)


def onset(recs, key, thresh):
    for r in sorted(recs, key=lambda r: r["delta"]):
        if r.get(key) is not None and r[key] > thresh:
            return r["delta"]
    return None


def fmt(x):
    if x is None:
        return "-"
    return f"{x:.3g}"


def print_summary(rows):
    sites = sorted({r["site"] for r in rows if not r["site"].startswith("icepop")})
    print("\nOnsets (scaled margin delta where the relative error first exceeds 0.5) and precision before onset")
    print(f"{'site':34} {'z_max':>5} {'T':>4} {'logp onset':>10} {'grad onset':>10} {'logp rel<10':>11} {'zero-sum':>9}")
    for site in sites:
        for z_max in Z_MAXES:
            for t in TEMPERATURES:
                recs = [r for r in rows if r["site"] == site and r["z_max"] == z_max and r["temperature"] == t]
                if not recs:
                    continue
                lo = onset(recs, "logp_rel", 0.5)
                go = onset(recs, "gt_rel", 0.5) if "gt_rel" in recs[0] else None
                pre = max((r["logp_rel"] for r in recs if r["delta"] < 10), default=None)
                zs = max((r.get("zero_sum", 0.0) for r in recs if r["delta"] < 90), default=None)
                print(f"{site:34} {z_max:5.0f} {t:4.1f} {fmt(lo):>10} {fmt(go):>10} {fmt(pre):>11} {fmt(zs):>9}")
    print("\nIcePop tail relative error (max over the delta grid where tail_ref > 1e-7)")
    for site in sorted({r["site"] for r in rows if r["site"].startswith("icepop")}):
        for z_max in Z_MAXES:
            recs = [r for r in rows if r["site"] == site and r["z_max"] == z_max and r["tail_ref"] > 1e-7]
            old = max((r["tail_old_rel"] for r in recs), default=None)
            new = max((r["tail_new_rel"] for r in recs if "tail_new_rel" in r), default=None)
            print(f"{site:40} z_max={z_max:4.0f} old={fmt(old):>9} new={fmt(new):>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name()}")
    u = torch.rand(V, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 4
    rows = []
    for z_max in Z_MAXES:
        for t in TEMPERATURES:
            for pos, label in POSITIONS.items():
                for delta in DELTAS:
                    run_point(z_max, delta, t, pos, label, u, rows, args.tag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print_summary(rows)


if __name__ == "__main__":
    main()
