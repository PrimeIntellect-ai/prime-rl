import torch
from torch import Tensor


def apply_top_p_then_top_k(logits: Tensor, k: Tensor | None, p: Tensor | None) -> Tensor:
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


def patch_top_p_then_top_k() -> None:
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.worker.gpu.sample import sampler, states

    # V2 imports this function by value in both its normal and fallback paths.
    topk_topp_sampler.apply_top_k_top_p = apply_top_p_then_top_k
    states.apply_top_k_top_p = apply_top_p_then_top_k
    sampler.apply_top_k_top_p = apply_top_p_then_top_k
