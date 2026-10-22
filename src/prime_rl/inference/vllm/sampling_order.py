import torch
from torch import Tensor


def reference_top_p_then_top_k(logits: Tensor, k: Tensor | None, p: Tensor | None) -> Tensor:
    """Select the full-distribution nucleus, then cap its size, including ties."""
    if k is None and p is None:
        return logits
    sorted_logits, indices = logits.sort(dim=-1, stable=True)
    remove = torch.zeros_like(sorted_logits, dtype=torch.bool)
    if p is not None:
        # Compute mass before applying the top-k cap. Keep the crossing token.
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= 1 - p.unsqueeze(-1)
        remove[:, -1] = False
    if k is not None:
        ranks = torch.arange(logits.shape[-1], device=logits.device)
        remove |= ranks.unsqueeze(0) < logits.shape[-1] - k.unsqueeze(-1)
    sorted_logits.masked_fill_(remove, -torch.inf)
    return logits.scatter_(dim=-1, index=indices, src=sorted_logits)


def apply_top_p_then_top_k(logits: Tensor, k: Tensor | None, p: Tensor | None) -> Tensor:
    """Full-vocabulary nucleus mass, sorting only bounded candidates."""
    # For a single row, the full sort is cheaper than bounded-path dispatch.
    if k is None or logits.shape[0] == 1:
        return reference_top_p_then_top_k(logits, k, p)
    width = int(k.max().item())
    if width >= logits.shape[-1]:
        return reference_top_p_then_top_k(logits, k, p)
    # One extra candidate detects ties that cross the extraction boundary.
    values, ids = logits.topk(width + 1, dim=-1)
    if (values[:, width - 1] == values[:, width]).any().item():
        return reference_top_p_then_top_k(logits, k, p)
    values, ids = values[:, :width], ids[:, :width]
    # Stable descending score/token-ID ordering matches the reference's
    # ascending stable sort, including a nucleus boundary inside a tie.
    by_id = ids.argsort(dim=-1, descending=True)
    ids, values = ids.gather(-1, by_id), values.gather(-1, by_id)
    by_value = values.argsort(dim=-1, descending=True, stable=True)
    ids, values = ids.gather(-1, by_value), values.gather(-1, by_value)
    remove = torch.arange(width, device=logits.device).unsqueeze(0) >= k.unsqueeze(-1)
    if p is not None:
        probs = (values - logits.logsumexp(dim=-1, keepdim=True)).exp()
        # Exclusive cumulative mass keeps the crossing token.
        before = torch.cat((torch.zeros_like(probs[:, :1]), probs.cumsum(-1)[:, :-1]), dim=-1)
        remove |= (before >= p.unsqueeze(-1)) & (p.unsqueeze(-1) < 1)
    values = values.masked_fill(remove, -torch.inf)
    logits.fill_(-torch.inf)
    return logits.scatter_(dim=-1, index=ids, src=values)


def patch_top_p_then_top_k() -> None:
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.worker.gpu.sample import sampler, states

    # V2 imports this function by value in both its normal and fallback paths.
    topk_topp_sampler.apply_top_k_top_p = apply_top_p_then_top_k
    states.apply_top_k_top_p = apply_top_p_then_top_k
    sampler.apply_top_k_top_p = apply_top_p_then_top_k
