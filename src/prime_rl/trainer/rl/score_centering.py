"""Score centering under a top-k sampler head and a proportional trainer tail.

Reference: arXiv:2609.20807, equations 9 and 11. IPO's absolute-probability
mask varies within the tail, so its correction must include tail tokens.
"""

import torch
from torch import Tensor


def score_correction(logp: Tensor, head_ids: Tensor, head_logq: Tensor, ipo_eps: float | None) -> Tensor:
    valid = head_ids >= 0
    ids = head_ids.clamp_min(0).long()
    head_logp = logp.gather(-1, ids)
    with torch.no_grad():
        p = logp.exp()
        p_head = torch.where(valid, head_logp.exp(), 0.0)
        q_head = torch.where(valid, head_logq.exp(), 0.0)
        rho = (1 - q_head.sum(-1)).clamp_min(1e-6) / (1 - p_head.sum(-1)).clamp_min(1e-6)
        if ipo_eps is None:
            residual = q_head - rho.unsqueeze(-1) * p_head
        else:
            # q*w = p for accepted tokens and zero for rejected tokens.
            # Subtract p everywhere: its expected score is identically zero.
            q = rho.unsqueeze(-1) * p
            q.scatter_add_(-1, ids, q_head - q.gather(-1, ids) * valid)
            rejected = (p - q).abs() > ipo_eps
            residual_full = -p * rejected
    if ipo_eps is None:
        return (residual * head_logp).sum(-1)
    return (residual_full * logp).sum(-1)


def chunked_score_logprobs(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    temperature: Tensor,
    head_ids: Tensor,
    head_logq: Tensor,
    chunk_size: int,
    ipo_eps: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    from torch.utils.checkpoint import checkpoint

    shape = labels.shape
    hidden = hidden.flatten(0, 1)
    labels = labels.flatten()
    temperature = temperature.flatten()
    head_ids = head_ids.flatten(0, 1)
    head_logq = head_logq.flatten(0, 1)

    def compute(h, w, target, temp, ids, logq):
        logits = (h @ w.t()).float() / temp.unsqueeze(-1)
        logp = logits.log_softmax(-1)
        token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        correction = score_correction(logp, ids, logq, ipo_eps)
        with torch.no_grad():
            entropy = -(logp.exp() * logp).sum(-1)
        return token_logp, entropy, correction

    outputs = [
        checkpoint(
            compute,
            hidden[start : start + chunk_size],
            weight,
            labels[start : start + chunk_size],
            temperature[start : start + chunk_size],
            head_ids[start : start + chunk_size],
            head_logq[start : start + chunk_size],
            use_reentrant=False,
        )
        for start in range(0, len(labels), chunk_size)
    ]
    return tuple(torch.cat(parts).reshape(shape) for parts in zip(*outputs))
