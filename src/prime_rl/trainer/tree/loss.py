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
