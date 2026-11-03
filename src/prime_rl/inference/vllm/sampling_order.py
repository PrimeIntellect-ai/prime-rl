from contextvars import ContextVar

import torch
from torch import Tensor

# The batch's largest top-k, read from host state so the bounded path never syncs.
_batch_top_k: ContextVar[int | None] = ContextVar("prl_batch_top_k", default=None)


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
        remove |= (k.unsqueeze(-1) != -1) & (ranks.unsqueeze(0) < logits.shape[-1] - k.unsqueeze(-1))
    sorted_logits.masked_fill_(remove, -torch.inf)
    return logits.scatter_(dim=-1, index=indices, src=sorted_logits)


def apply_top_p_then_top_k(logits: Tensor, k: Tensor | None, p: Tensor | None) -> Tensor:
    """Full-vocabulary nucleus mass, sorting only bounded candidates."""
    # For a single row, the full sort is cheaper than bounded-path dispatch.
    if k is None or logits.shape[0] == 1:
        return reference_top_p_then_top_k(logits, k, p)
    k = torch.where(k == -1, logits.shape[-1], k)
    width = _batch_top_k.get()
    if width is None:
        width = int(k.max().item())
    if width >= logits.shape[-1]:
        return reference_top_p_then_top_k(logits, k, p)
    # Rank by (score, token ID) descending, the reference's stable ascending
    # order reversed, so boundary ties need no data-dependent fallback.
    bits = logits.float().view(torch.int32)
    keys = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).long() << 32
    keys |= torch.arange(logits.shape[-1], device=logits.device)
    ids = keys.topk(width, dim=-1).values & 0xFFFFFFFF
    values = logits.gather(-1, ids)
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

    sample_random = sampler.Sampler._sample_random

    def _sample_random(self, processed_logits, expanded_idx_mapping, idx_mapping_np, *args):
        token = _batch_top_k.set(int(self.sampling_states.top_k.np[idx_mapping_np].max()))
        try:
            return sample_random(self, processed_logits, expanded_idx_mapping, idx_mapping_np, *args)
        finally:
            _batch_top_k.reset(token)

    sampler.Sampler._sample_random = _sample_random
