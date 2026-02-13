"""TRPL (Trust Region Policy Layer) loss for prime-rl.

From-scratch implementation using sparse union of top-k indices.
Projects the current policy onto a KL-constrained trust region around the
old/inference policy, then computes:
  L = -advantages * exp(projected_logprobs - old_logprobs) + alpha * KL(current || projected.detach())

The projection finds eta such that:
  projected(a) = softmax((log_current(a) + eta * log_old(a)) / (eta + 1))
  KL(projected || old) <= kl_bound

All heavy operations are performed on the union of top-k indices from both
policies (~2K entries per token), never on the full vocabulary.

Current policy data comes from the fused LM head (FusedOutputLinear) which
computes top-K indices/values and log-probs at old positions during its
chunked vocab loop — no full [seq, vocab] tensor is ever materialized.
"""

import logging
import time

import torch
from torch import Tensor

from prime_rl.trainer.rl.config import TrplLossConfig
from prime_rl.trainer.rl.loss import LossFn, LossInputs, LossOutputs

logger = logging.getLogger(__name__)
_trpl_call_count = 0


def _build_union_indices(
    current_top_idx: Tensor,
    old_top_idx: Tensor,
    token_ids: Tensor,
    vocab_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build sorted, deduplicated union of current top-K, old top-K, and actual tokens.

    Args:
        current_top_idx: [N, K] top-k indices from current policy
        old_top_idx: [N, K] top-k indices from old policy
        token_ids: [N] actual generated token ids
        vocab_size: vocabulary size (used as sentinel for dedup)

    Returns:
        union_idx: [N, K_u] sorted unique indices, padded with 0 where invalid
        union_mask: [N, K_u] bool mask of valid entries
        token_pos: [N] position of actual token within union
    """
    # Concatenate all candidate indices: [N, 2K+1]
    all_idx = torch.cat([current_top_idx, old_top_idx, token_ids.unsqueeze(1)], dim=1)

    # Sort per row
    sorted_idx, _ = all_idx.sort(dim=1)

    # Mark duplicates (consecutive equal values)
    is_dup = torch.zeros_like(sorted_idx, dtype=torch.bool)
    is_dup[:, 1:] = sorted_idx[:, 1:] == sorted_idx[:, :-1]

    # Replace duplicates with sentinel (vocab_size), then re-sort to push them to end
    sorted_idx = sorted_idx.clone()
    sorted_idx[is_dup] = vocab_size
    sorted_idx, _ = sorted_idx.sort(dim=1)

    # Trim to max union size and build mask
    union_sizes = (sorted_idx < vocab_size).sum(dim=1)  # [N]
    K_u = union_sizes.max().item()
    union_idx = sorted_idx[:, :K_u]
    union_mask = union_idx < vocab_size
    union_idx = union_idx.clamp(max=vocab_size - 1)  # safe indexing for masked positions

    # Find token position within union
    token_expanded = token_ids.unsqueeze(1).expand_as(union_idx)
    token_match = (union_idx == token_expanded) & union_mask
    token_pos = token_match.long().argmax(dim=1)  # [N]

    return union_idx, union_mask, token_pos


def _gather_current_at_union(
    union_idx: Tensor,
    union_mask: Tensor,
    current_top_idx: Tensor,
    current_top_lp: Tensor,
    old_top_idx: Tensor,
    old_idx_current_lp: Tensor,
    token_ids: Tensor,
    token_lp: Tensor,
) -> Tensor:
    """Gather current policy log-probs at union positions from sparse sources.

    Uses three sources: current top-K, old top-K positions, and actual token.
    Every valid union position must have a match in at least one source
    (since the union is built from exactly these sources).

    Args:
        union_idx: [N, K_u] union indices
        union_mask: [N, K_u] valid mask
        current_top_idx: [N, K] current top-K indices
        current_top_lp: [N, K] current log-probs at current top-K (with grad)
        old_top_idx: [N, K_old] old top-K indices
        old_idx_current_lp: [N, K_old] current log-probs at old top-K positions (with grad)
        token_ids: [N] actual token ids
        token_lp: [N] current log-prob at actual token (with grad)

    Returns:
        current_lp: [N, K_u] current log-probs at union positions (with grad)
    """
    # Combine all sources into one lookup
    all_src_idx = torch.cat([current_top_idx, old_top_idx, token_ids.unsqueeze(1)], dim=1)  # [N, S]
    all_src_lp = torch.cat([current_top_lp, old_idx_current_lp, token_lp.unsqueeze(1)], dim=1)  # [N, S]

    # Match union positions to source positions: [N, K_u, S]
    match = union_idx.unsqueeze(2) == all_src_idx.unsqueeze(1)
    # For each union position, find first matching source index
    match_idx = match.float().argmax(dim=2)  # [N, K_u]

    # Gather from combined sources
    current_lp = all_src_lp.gather(1, match_idx)  # [N, K_u]

    return current_lp


def _gather_old_at_union(
    union_idx: Tensor,
    union_mask: Tensor,
    old_top_idx: Tensor,
    old_top_val: Tensor,
    default_log_prob: float,
) -> Tensor:
    """Gather old log-probs at union positions. Missing entries get default_log_prob.

    Args:
        union_idx: [N, K_u] union indices
        union_mask: [N, K_u] valid mask
        old_top_idx: [N, K] old top-k indices
        old_top_val: [N, K] old top-k log-prob values
        default_log_prob: fill value for positions not in old top-k

    Returns:
        old_union_lp: [N, K_u] old log-probs at union positions
    """
    # Compare union positions against old top-k: [N, K_u, 1] vs [N, 1, K]
    match = union_idx.unsqueeze(2) == old_top_idx.unsqueeze(1)  # [N, K_u, K]
    has_match = match.any(dim=2)  # [N, K_u]
    match_idx = match.float().argmax(dim=2)  # [N, K_u] index into old K dim

    # Gather matched values
    matched_values = old_top_val.gather(1, match_idx)  # [N, K_u]

    # Default where no match or invalid
    old_union_lp = torch.where(has_match & union_mask, matched_values, default_log_prob)
    return old_union_lp


def _tail_logsumexp(lp_union: Tensor, union_mask: Tensor, tail_count: Tensor, default_lp: Tensor) -> Tensor:
    """Compute logsumexp over full distribution from union values + tail.

    Args:
        lp_union: [N, K_u] log-probs at union positions
        union_mask: [N, K_u] valid mask
        tail_count: [N] number of positions not in union
        default_lp: [N] or scalar, log-prob for each tail position

    Returns:
        lse: [N] logsumexp values
    """
    # Mask invalid union positions to -inf for logsumexp
    masked_lp = lp_union.masked_fill(~union_mask, -float("inf"))

    # Max over union
    max_union = masked_lp.max(dim=-1).values  # [N]
    if default_lp.dim() == 0:
        default_lp = default_lp.expand(max_union.shape[0])
    max_val = torch.maximum(max_union, default_lp)  # [N]

    # Sum exp over union
    sum_exp_union = (masked_lp - max_val.unsqueeze(1)).exp().sum(dim=-1)  # [N]

    # Tail contribution
    sum_exp_tail = tail_count.float() * (default_lp - max_val).exp()  # [N]

    return max_val + (sum_exp_union + sum_exp_tail).log()  # [N]


def _renormalize_old_sparse(
    old_union_lp: Tensor,
    union_mask: Tensor,
    vocab_size: int,
    default_log_prob: float,
) -> tuple[Tensor, Tensor]:
    """Renormalize old log-probs so the full distribution sums to 1.

    Returns:
        old_union_lp: [N, K_u] renormalized old log-probs at union positions
        old_default_lp: [N] renormalized default log-prob for tail positions
    """
    tail_count = vocab_size - union_mask.sum(dim=-1)  # [N]
    default_lp = torch.tensor(default_log_prob, device=old_union_lp.device, dtype=old_union_lp.dtype)
    lse = _tail_logsumexp(old_union_lp, union_mask, tail_count, default_lp)  # [N]

    old_union_lp = old_union_lp - lse.unsqueeze(1)
    old_default_lp = default_lp - lse  # [N]
    return old_union_lp, old_default_lp


def _kl_sparse(
    p_union: Tensor,
    q_union: Tensor,
    p_default: Tensor,
    q_default: Tensor,
    union_mask: Tensor,
    tail_count: Tensor,
) -> Tensor:
    """Compute KL(p || q) over full distribution using union + tail.

    KL = sum_{i in union} p(i) * (log p(i) - log q(i))
       + tail_count * p_default_prob * (p_default - q_default)

    Args:
        p_union: [N, K_u] p log-probs at union positions
        q_union: [N, K_u] q log-probs at union positions
        p_default: [N] p default log-prob for tail
        q_default: [N] q default log-prob for tail
        union_mask: [N, K_u] valid mask
        tail_count: [N] number of tail positions

    Returns:
        kl: [N] KL divergence per token
    """
    # Union contribution
    p_probs = p_union.exp()
    kl_union = (p_probs * (p_union - q_union)).masked_fill(~union_mask, 0.0).sum(dim=-1)  # [N]

    # Tail contribution
    p_default_prob = p_default.exp()  # [N]
    kl_tail = tail_count.float() * p_default_prob * (p_default - q_default)  # [N]

    return kl_union + kl_tail


@torch.no_grad()
def _bisect_eta_sparse(
    current_lp: Tensor,
    old_lp: Tensor,
    current_default: Tensor,
    old_default: Tensor,
    union_mask: Tensor,
    tail_count: Tensor,
    vocab_size: int,
    bound: float,
    num_steps: int = 24,
    log_eta_lo: float = -8.0,
    log_eta_hi: float = 10.0,
) -> Tensor:
    """Bisect for optimal dual variable eta on sparse union representation.

    Args:
        current_lp: [K, K_u] current log-probs at union positions
        old_lp: [K, K_u] old log-probs at union positions
        current_default: [K] current default log-prob for tail
        old_default: [K] old default log-prob for tail
        union_mask: [K, K_u] valid mask
        tail_count: [K] number of tail positions
        vocab_size: full vocabulary size
        bound: target KL bound

    Returns:
        log_eta: [K, 1] optimal log(eta) values
    """
    K = current_lp.shape[0]
    device, dtype = current_lp.device, current_lp.dtype

    lo = torch.full((K, 1), log_eta_lo, device=device, dtype=dtype)
    hi = torch.full((K, 1), log_eta_hi, device=device, dtype=dtype)

    for _ in range(num_steps):
        mid = (lo + hi) * 0.5
        eta = mid.exp()  # [K, 1]

        # Projection on union: inner = (current + eta * old) / (eta + 1)
        inner_union = (current_lp + eta * old_lp) / (eta + 1)  # [K, K_u]
        inner_default = (current_default.unsqueeze(1) + eta * old_default.unsqueeze(1)) / (eta + 1)  # [K, 1]

        # Normalize with tail-aware logsumexp
        lse = _tail_logsumexp(inner_union, union_mask, tail_count, inner_default.squeeze(1))  # [K]
        proj_lp = inner_union - lse.unsqueeze(1)  # [K, K_u]
        proj_default = inner_default.squeeze(1) - lse  # [K]

        # KL(projected || old) on union + tail
        kl = _kl_sparse(proj_lp, old_lp, proj_default, old_default, union_mask, tail_count)  # [K]

        # Bisect: KL > bound means need more eta
        too_high = (kl > bound).unsqueeze(1)
        lo = torch.where(too_high, mid, lo)
        hi = torch.where(too_high, hi, mid)

    return (lo + hi) * 0.5  # [K, 1]


def trpl_loss_fn(
    inputs: LossInputs,
    alpha: float = 1.0,
    kl_bound: float = 0.05,
    top_k: int = 64,
    default_log_prob: float = -27.6,
) -> LossOutputs:
    """Compute TRPL loss for a single packed sequence.

    All heavy operations work on the union of top-k indices (~2K per token),
    never on the full vocabulary, making this memory-efficient for large vocabs.

    Accepts sparse current policy data from the fused LM head:
    - current_top_k_indices/values: current policy's top-K (with gradient)
    - old_indices_current_lp: current log-probs at old top-K positions (with gradient)

    Args:
        inputs: LossInputs with TRPL extensions populated
        alpha: weight of projection loss KL(current || projected)
        kl_bound: trust region KL bound
        top_k: number of top-k entries from current policy (for reference only)
        default_log_prob: default log-prob for tokens outside top-k support

    Returns:
        LossOutputs with loss and metrics.
    """
    assert inputs.current_top_k_indices is not None, "TRPL loss requires current_top_k_indices"
    assert inputs.current_top_k_values is not None, "TRPL loss requires current_top_k_values"
    assert inputs.old_indices_current_lp is not None, "TRPL loss requires old_indices_current_lp"
    assert inputs.old_top_indices is not None, "TRPL loss requires old_top_indices"
    assert inputs.old_top_values is not None, "TRPL loss requires old_top_values"
    assert inputs.token_ids is not None, "TRPL loss requires token_ids"
    assert inputs.vocab_size is not None, "TRPL loss requires vocab_size"

    loss_mask = inputs.loss_mask
    advantages = inputs.advantages
    token_ids = inputs.token_ids  # [seq]
    V = inputs.vocab_size

    # Early exit if no tokens to train on
    device = inputs.current_top_k_values.device
    if loss_mask.sum() == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return LossOutputs(
            loss=zero,
            metrics={"pg_loss": zero.detach(), "projection_loss": zero.detach()},
        )

    global _trpl_call_count
    _trpl_call_count += 1
    _do_profile = _trpl_call_count <= 5  # profile first 5 calls

    if _do_profile:
        torch.cuda.synchronize()
        _t0 = time.perf_counter()

    # Apply loss mask to get only completion tokens
    current_top_idx = inputs.current_top_k_indices[loss_mask]  # [N, K]
    current_top_lp = inputs.current_top_k_values[loss_mask]  # [N, K] — has grad
    old_idx_current_lp = inputs.old_indices_current_lp[loss_mask]  # [N, K_old] — has grad
    old_top_idx = inputs.old_top_indices[loss_mask]  # [N, K_old]
    old_top_val = inputs.old_top_values[loss_mask]  # [N, K_old]
    tokens = token_ids[loss_mask]  # [N]
    advs = advantages[loss_mask]  # [N]
    token_lp = inputs.trainer_logprobs[loss_mask]  # [N] — has grad
    N = tokens.shape[0]

    # Step 1: Build union of indices (current_top_K ∪ old_top_K ∪ {actual_token})
    union_idx, union_mask, token_pos = _build_union_indices(
        current_top_idx, old_top_idx, tokens, V
    )  # [N, K_u], [N, K_u], [N]
    K_u = union_idx.shape[1]
    tail_count = V - union_mask.sum(dim=-1)  # [N]

    if _do_profile:
        torch.cuda.synchronize()
        _t1 = time.perf_counter()

    # Step 2: Gather current log-probs at union positions from sparse sources
    current_lp = _gather_current_at_union(
        union_idx,
        union_mask,
        current_top_idx,
        current_top_lp,
        old_top_idx,
        old_idx_current_lp,
        tokens,
        token_lp,
    )  # [N, K_u] — has grad

    if _do_profile:
        torch.cuda.synchronize()
        _t2 = time.perf_counter()

    # Step 3: Compute current default log-prob from tail mass (differentiable).
    # Gradient through tail mass is essential: without it, the projection loss
    # KL gradient only covers union positions, and through the log-softmax Jacobian
    # gradient descent systematically pushes mass from union to tail, causing
    # entropy explosion over many steps.
    current_union_mass = (current_lp.exp() * union_mask.float()).sum(dim=-1)  # [N]
    current_tail_mass = (1.0 - current_union_mass).clamp(min=1e-20)  # [N]
    current_default_lp = (current_tail_mass / tail_count.float().clamp(min=1)).log()  # [N]

    # Step 4: Gather old log-probs at union positions + renormalize
    old_lp = _gather_old_at_union(union_idx, union_mask, old_top_idx, old_top_val, default_log_prob)
    old_lp, old_default_lp = _renormalize_old_sparse(old_lp, union_mask, V, default_log_prob)

    if _do_profile:
        torch.cuda.synchronize()
        _t3 = time.perf_counter()

    # Step 5: Compute initial KL(current || old) to find which positions need projection
    with torch.no_grad():
        initial_kl = _kl_sparse(current_lp.detach(), old_lp, current_default_lp, old_default_lp, union_mask, tail_count)
        needs_proj = initial_kl > kl_bound
        num_projected = needs_proj.sum().item()

    if _do_profile:
        torch.cuda.synchronize()
        _t4 = time.perf_counter()

    # Step 6: Find optimal eta via bisection (no gradient through eta)
    eta = torch.zeros(N, 1, device=current_lp.device, dtype=current_lp.dtype)
    if num_projected > 0:
        log_eta = _bisect_eta_sparse(
            current_lp[needs_proj].detach(),
            old_lp[needs_proj],
            current_default_lp[needs_proj],
            old_default_lp[needs_proj],
            union_mask[needs_proj],
            tail_count[needs_proj],
            V,
            kl_bound,
        )
        eta[needs_proj] = log_eta.exp()

    if _do_profile:
        torch.cuda.synchronize()
        _t5 = time.perf_counter()

    # Step 7: REPS update on union: projected = softmax((current + eta*old) / (eta+1))
    inner_union = (current_lp + eta * old_lp) / (eta + 1)  # [N, K_u]
    inner_default = (current_default_lp.unsqueeze(1) + eta * old_default_lp.unsqueeze(1)) / (eta + 1)  # [N, 1]
    lse = _tail_logsumexp(inner_union, union_mask, tail_count, inner_default.squeeze(1))  # [N]
    projected_lp = inner_union - lse.unsqueeze(1)  # [N, K_u]
    projected_default_lp = inner_default.squeeze(1) - lse  # [N]

    # Step 8: Importance ratio at actual token
    proj_token_lp = projected_lp.gather(1, token_pos.unsqueeze(1)).squeeze(1)  # [N]
    old_token_lp = old_lp.gather(1, token_pos.unsqueeze(1)).squeeze(1)  # [N]
    logratio = (proj_token_lp - old_token_lp).clamp(-20.0, 20.0)
    importance_ratio = torch.exp(logratio)

    # PG loss: REINFORCE-style with detached ratio.
    # In the reference, the SdtrplLayer backward zeros out PG gradient for non-projected
    # tokens. We can't replicate that without implicit differentiation, so we use
    # REINFORCE (gradient through token_lp only) which provides learning signal for ALL
    # tokens regardless of projection status.
    # Returns .sum() to match default_loss_fn convention; normalization happens in
    # train.py via loss_scale = total_response_tokens.
    pg_loss = -(advs * importance_ratio.detach() * token_lp).sum()

    # Step 9: Projection loss KL(current || projected.detach()) — ONLY on projected tokens.
    # The reference SdtrplLayer backward zeros out gradient for non-projected tokens
    # (where eta=0). For non-projected tokens, KL(current || projected.detach()) has
    # forward value ~0, but the gradient is non-zero due to incomplete log-softmax
    # cancellation in the sparse representation, causing spurious entropy increase.
    # Restricting to projected tokens eliminates this and matches reference behavior.
    # Also uses .sum() for consistent normalization.
    if num_projected > 0:
        projected_lp_det = projected_lp[needs_proj].detach()
        projected_default_det = projected_default_lp[needs_proj].detach()
        projection_loss = _kl_sparse(
            current_lp[needs_proj],
            projected_lp_det,
            current_default_lp[needs_proj],
            projected_default_det,
            union_mask[needs_proj],
            tail_count[needs_proj],
        ).sum()
    else:
        projection_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Total loss
    total_loss = pg_loss + alpha * projection_loss

    # Metrics
    with torch.no_grad():
        final_kl = _kl_sparse(
            projected_lp,
            old_lp,
            projected_default_lp,
            old_default_lp,
            union_mask,
            tail_count,
        )

        # mismatch_kl: KL divergence approx between trainer and inference
        # (same metric as default_loss_fn, using per-token scalar logprobs)
        log_ir = inputs.trainer_logprobs[loss_mask] - inputs.inference_logprobs[loss_mask]
        token_mismatch_kl = torch.exp(log_ir) - log_ir - 1

        metrics = {
            "pg_loss": pg_loss.detach() / max(N, 1),
            "projection_loss": projection_loss.detach() / max(num_projected, 1),
            "importance_ratio": importance_ratio.detach().mean(),
            "initial_kl": initial_kl.mean(),
            "final_kl": final_kl.mean(),
            "max_vio": torch.clamp(final_kl - kl_bound, min=0).max(),
            "projected_frac": torch.tensor(num_projected / max(N, 1), device=current_lp.device),
            "mismatch_kl": token_mismatch_kl.mean(),
        }

    if _do_profile:
        torch.cuda.synchronize()
        _t6 = time.perf_counter()
        logger.info(
            f"trpl_loss_fn profile (N={N}, K_u={K_u}, projected={num_projected}/{N}): "
            f"build_union={_t1 - _t0:.4f}s, gather_current={_t2 - _t1:.4f}s, "
            f"old+renorm={_t3 - _t2:.4f}s, initial_kl={_t4 - _t3:.4f}s, "
            f"bisect_eta={_t5 - _t4:.4f}s, project+loss={_t6 - _t5:.4f}s, "
            f"total={_t6 - _t0:.4f}s"
        )

    return LossOutputs(loss=total_loss, metrics=metrics)


def setup_trpl_loss_fn(loss_config: TrplLossConfig) -> LossFn:
    """Create a TRPL loss function from config."""

    def loss_fn(inputs: LossInputs) -> LossOutputs:
        return trpl_loss_fn(
            inputs,
            alpha=loss_config.alpha,
            kl_bound=loss_config.kl_bound,
            top_k=loss_config.top_k,
            default_log_prob=loss_config.default_log_prob,
        )

    return loss_fn
