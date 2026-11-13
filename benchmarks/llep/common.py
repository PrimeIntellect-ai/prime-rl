"""Routing scenarios, the reference MoE and metrics shared by the LLEP benchmarks."""

import math

import torch
import torch.nn.functional as F


def make_routing(
    scenario: str, num_tokens: int, num_experts: int, top_k: int, rank: int, num_ranks: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(top_scores [T, K] float32, topk_idx [T, K] int64)`` for this rank.

    - ``balanced``: global token ``i`` takes experts ``i*K .. i*K+K-1`` (mod N).
    - ``hot:<pct>:<h>``: ``pct``% of each rank's tokens are hot. With ``h >= K`` a hot token takes K distinct
      experts among the ``h`` hot experts ``0..h-1``; with ``h < K`` it takes all of them plus cold ones. Every
      other route goes to the cold experts in a round robin, so routes stay distinct per token.
    - ``orig:<pct>:<h>``: the LeastLoadedEP simulation: a hot token sends its k-th route to hot expert
      ``k % h`` (duplicates when ``h < K``), a cold token's routes walk the cold experts.
    - ``spread:<pct>:<h>``: as ``hot`` with ``h >= K``, but the hot experts are ``h`` random experts (seed 0), so they
      spread over the ranks as a trained router's popular experts do.
    - ``zipf:<s>``: experts drawn without replacement with probability ``∝ (j+1)^-s`` over a random permutation.
    """
    gen = torch.Generator().manual_seed(seed * 1000003 + rank)
    first = rank * num_tokens
    i = torch.arange(first, first + num_tokens).unsqueeze(1)
    j = torch.arange(top_k).unsqueeze(0)
    kind, *args = scenario.split(":")
    if kind == "balanced":
        idx = (i * top_k + j) % num_experts
    elif kind == "spread":
        pct, h = float(args[0]), int(args[1])
        perm = torch.randperm(num_experts, generator=torch.Generator().manual_seed(seed))
        hot_experts, cold_experts = perm[:h], perm[h:]
        local = i - first
        hot = hot_experts[(local * top_k + j) % h]
        cold = cold_experts[(i * top_k + j) % (num_experts - h)]
        idx = torch.where(local < int(num_tokens * pct / 100), hot, cold)
    elif kind in ("hot", "orig"):
        pct, h = float(args[0]), int(args[1])
        num_hot = int(num_tokens * pct / 100)
        local = i - first
        cold = h + (i * top_k + j) % (num_experts - h)
        if kind == "orig":
            hot = (j % h).expand(num_tokens, -1)
        elif h >= top_k:
            hot = (local * top_k + j) % h
        else:
            hot = torch.where(j < h, j, h + (i * top_k + j) % (num_experts - h))
        idx = torch.where(local < num_hot, hot, cold)
    elif kind == "zipf":
        s = float(args[0])
        perm = torch.randperm(num_experts, generator=torch.Generator().manual_seed(seed))
        probs = torch.empty(num_experts)
        probs[perm] = (torch.arange(num_experts, dtype=torch.float64) + 1).pow(-s).float()
        idx = torch.multinomial(probs.expand(num_tokens, -1), top_k, generator=gen)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    logits = torch.randn(num_tokens, top_k, generator=gen)
    scores = F.softmax(logits, dim=-1) * 1.5
    return scores.float().cuda(), idx.to(torch.int64).cuda()


def global_loads(topk_idx: torch.Tensor, num_experts: int, group) -> torch.Tensor:
    loads = torch.bincount(topk_idx.reshape(-1), minlength=num_experts).float()
    torch.distributed.all_reduce(loads, group=group)
    return loads


def imbalance_metrics(loads: torch.Tensor, top_k: int, num_ranks: int) -> dict[str, float]:
    """``max_vio`` as prime-rl logs it (top-k busiest experts dropped), the raw ``max/mean - 1``, and the
    busiest rank's load over the mean under standard EP."""
    ordered = loads.sort(descending=True).values
    rest = ordered[top_k:]
    rank_loads = loads.view(num_ranks, -1).sum(dim=1)
    return {
        "max_vio": ((rest.max() - rest.mean()) / rest.mean()).item(),
        "max_vio_raw": (loads.max() / loads.mean() - 1).item(),
        "rank_imbalance": (rank_loads.max() / rank_loads.mean()).item(),
    }


def clamped_swiglu(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def expert_ffn(x, num_tokens_per_expert, gate_up, down, limit):
    """Grouped SwiGLU expert MLP; ``gate_up [E, 2I, H]`` with gate rows first, ``down [E, H, I]``."""
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = torch._grouped_mm(x, gate_up.transpose(-2, -1), offsets)
    return torch._grouped_mm(clamped_swiglu(h, limit), down.transpose(-2, -1), offsets)


def reference_moe(x, top_scores, topk_idx, gate_up_full, down_full, limit) -> torch.Tensor:
    """fp32 MoE over all experts, one expert at a time."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in topk_idx.unique().tolist():
        token, slot = (topk_idx == e).nonzero(as_tuple=True)
        h = xf[token] @ gate_up_full[e].float().t()
        y = clamped_swiglu(h, limit) @ down_full[e].float().t()
        out.index_add_(0, token, y * top_scores[token, slot].unsqueeze(1))
    return out


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def gib(num_bytes: float) -> float:
    return num_bytes / 2**30


def ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)
