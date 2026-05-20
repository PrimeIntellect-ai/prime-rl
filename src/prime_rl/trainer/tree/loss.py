import torch
import torch.nn.functional as F


def _tree_nll_weights(
    prev_map: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    if prev_map.shape != loss_mask.shape or prev_map.shape != loss_weights.shape:
        raise ValueError("prev_map, loss_mask, and loss_weights must have matching shapes")

    valid = prev_map >= 0
    dtype = loss_weights.dtype if loss_weights.is_floating_point() else torch.float32
    return loss_weights.to(dtype) * loss_mask.to(dtype) * valid.to(dtype)


def tree_nll_weighted_token_count(
    prev_map: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the denominator matching ``tree_nll_loss``'s g_t / K weighting."""
    return _tree_nll_weights(prev_map, loss_mask, loss_weights).sum()


def branch_group_nll_loss(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    branch_loss_weights: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [branches, seq, vocab], got {tuple(logits.shape)}")
    if target_ids.shape != loss_mask.shape or target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids and loss_mask must match logits' [branches, seq] shape")
    if branch_loss_weights.shape != (logits.shape[0],):
        raise ValueError("branch_loss_weights must have shape [branches]")

    token_loss = F.cross_entropy(
        logits.flatten(0, 1),
        target_ids.flatten(0, 1),
        reduction="none",
    ).view_as(target_ids)
    weights = branch_loss_weights.to(token_loss.dtype).unsqueeze(1) * loss_mask.to(token_loss.dtype)
    return (token_loss * weights).sum()


def branch_group_weighted_token_count(
    loss_mask: torch.BoolTensor,
    branch_loss_weights: torch.Tensor,
) -> torch.Tensor:
    if loss_mask.ndim != 2:
        raise ValueError(f"loss_mask must have shape [branches, seq], got {tuple(loss_mask.shape)}")
    if branch_loss_weights.shape != (loss_mask.shape[0],):
        raise ValueError("branch_loss_weights must have shape [branches]")
    return (loss_mask.to(branch_loss_weights.dtype) * branch_loss_weights.unsqueeze(1)).sum()


def tree_nll_loss(
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    prev_map: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if input_ids.shape != prev_map.shape or input_ids.shape != loss_mask.shape:
        raise ValueError("input_ids, prev_map, and loss_mask must have matching shapes")
    if input_ids.shape != loss_weights.shape:
        raise ValueError("input_ids and loss_weights must have matching shapes")

    gather_idx = prev_map.clamp(min=0).unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    predictor_logits = logits.gather(dim=1, index=gather_idx)
    token_loss = F.cross_entropy(
        predictor_logits.flatten(0, 1),
        input_ids.flatten(0, 1),
        reduction="none",
    ).view_as(input_ids)
    weights = _tree_nll_weights(prev_map, loss_mask, loss_weights).to(token_loss.dtype)
    return (token_loss * weights).sum()
