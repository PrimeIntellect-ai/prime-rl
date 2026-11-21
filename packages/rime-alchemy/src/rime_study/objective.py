"""Differentiable, token-normalized asymmetric PPO with sampled reference KL.

PyTorch is imported only by callers running a training backend. Prompt tokens
must be removed before this function; old/ref log probabilities are detached.
"""


def token_objective(current, old, reference, advantage, *, clip_low=0.2, clip_high=0.28, kl_coef=0.01):
    import torch

    if not 0 < clip_low < 1 or clip_high <= 0 or kl_coef < 0:
        raise ValueError("Invalid clipping or KL configuration")
    if current.ndim != 1 or current.numel() == 0 or current.shape != old.shape or current.shape != reference.shape:
        raise ValueError("Expected aligned, nonempty completion log-probability vectors")
    if not all(torch.isfinite(x).all() for x in (current, old, reference)):
        raise ValueError("Nonfinite token log probabilities")
    ratio = (current - old.detach()).exp()
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    delta = reference.detach() - current
    kl = delta.exp() - delta - 1
    return -surrogate + kl_coef * kl


def update_shared_policy(backend, samples, optimizer, *, clip_low=0.2, clip_high=0.28, kl_coef=0.01, max_grad_norm=1.0):
    """One on-policy update, accumulating ACT/WRITE into the SAME parameters.

    Token normalization spans all completions, not a mean of sequence means.
    No optimizer update occurs inside collection or between the two roles.
    """
    import torch

    if {sample.role for sample in samples} != {"actor", "writer"}:
        raise ValueError("Joint update requires actor and writer samples")
    total_tokens = sum(len(s.completion.old_logprobs) for s in samples)
    optimizer.zero_grad(set_to_none=True)
    losses = {"actor": 0.0, "writer": 0.0}
    counts = {"actor": 0, "writer": 0}
    for sample in samples:
        current, reference = backend.logprobs(sample.completion)
        old = torch.tensor(sample.completion.old_logprobs, device=current.device, dtype=current.dtype)
        loss = token_objective(
            current, old, reference, sample.advantage, clip_low=clip_low, clip_high=clip_high, kl_coef=kl_coef
        )
        (loss.sum() / total_tokens).backward()
        losses[sample.role] += loss.detach().sum().item()
        counts[sample.role] += loss.numel()
    norm = torch.nn.utils.clip_grad_norm_(backend.model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss_per_token": {r: losses[r] / counts[r] for r in counts},
        "tokens": counts,
        "gradient_norm": float(norm),
    }
