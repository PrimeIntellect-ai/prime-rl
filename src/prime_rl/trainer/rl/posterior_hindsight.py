"""Optional on-policy objectives for a verified two-goal hindsight pair."""

import torch
from torch import Tensor

from prime_rl.trainer.rl.supervised import summed_weighted_nll


def _probabilities(posterior: Tensor, reference: Tensor, dtype: torch.dtype) -> Tensor:
    weights = posterior.detach().to(device=reference.device, dtype=dtype)
    torch._assert_async(
        (torch.isfinite(weights) & (weights >= 0) & (weights <= 1)).all(),
        "Posterior probabilities must be finite and in [0, 1]",
    )
    return weights


def weighted_hindsight_loss(achieved_logp: Tensor, posterior: Tensor, episode_count: int) -> Tensor:
    """Return the zero-baseline loss from fresh summed path scores of eligible episodes.

    The denominator counts all original episodes, including those with zero achieved-goal prior.
    """
    if achieved_logp.ndim != 1 or posterior.shape != achieved_logp.shape:
        raise ValueError("Expected one posterior for each achieved-goal path score")
    if episode_count < 1 or episode_count < achieved_logp.numel():
        raise ValueError("The denominator must include all original episodes")
    dtype = torch.promote_types(torch.float32, torch.promote_types(achieved_logp.dtype, posterior.dtype))
    weight = _probabilities(posterior, achieved_logp, dtype)
    return summed_weighted_nll(achieved_logp.to(dtype), None, weight) / episode_count


def centered_pair_loss(
    achieved_logp: Tensor,
    foil_logp: Tensor,
    posterior: Tensor,
    baseline: Tensor,
    *,
    foil_posterior: Tensor | None = None,
) -> Tensor:
    """Average the centered negative policy score over every original episode at the collecting policy.

    Both scores are summed, differentiable log probabilities conditional on their respective goals.
    Use a zero placeholder for an absent role. The pre-action baseline is shared by the two goals.
    When available, pass the independently normalized foil probability to preserve small tail weights.
    """
    if achieved_logp.ndim != 1 or not achieved_logp.numel():
        raise ValueError("Expected one path-score row for each original episode")
    if foil_logp.shape != achieved_logp.shape or posterior.shape != achieved_logp.shape:
        raise ValueError("Achieved scores, foil scores and posterior must align")
    if baseline.ndim and baseline.shape != achieved_logp.shape:
        raise ValueError("The baseline must be scalar or have one value per original episode")
    dtype = torch.float32
    for value in (achieved_logp, foil_logp, posterior, baseline):
        dtype = torch.promote_types(dtype, value.dtype)
    if foil_posterior is not None:
        if foil_posterior.shape != posterior.shape:
            raise ValueError("The two posterior probabilities must align")
        dtype = torch.promote_types(dtype, foil_posterior.dtype)
    weight = _probabilities(posterior, achieved_logp, dtype)
    if foil_posterior is None:
        foil_weight = 1 - weight
    else:
        foil_weight = _probabilities(foil_posterior, achieved_logp, dtype)
        tolerance = max(
            (torch.finfo(t.dtype).eps for t in (posterior, foil_posterior) if t.is_floating_point()),
            default=torch.finfo(dtype).eps,
        )
        torch._assert_async(((weight + foil_weight - 1).abs() <= tolerance).all(), "Posterior roles must sum to one")
    center = baseline.detach().to(device=achieved_logp.device, dtype=dtype)
    torch._assert_async(torch.isfinite(center).all(), "Baseline must be finite")
    return -(weight * (1 - center) * achieved_logp.to(dtype) - foil_weight * center * foil_logp.to(dtype)).mean()
