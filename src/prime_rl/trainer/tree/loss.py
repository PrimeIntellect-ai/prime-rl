import torch
import torch.nn.functional as F


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

    valid = prev_map >= 0
    gather_idx = prev_map.clamp(min=0).unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    predictor_logits = logits.gather(dim=1, index=gather_idx)
    token_loss = F.cross_entropy(
        predictor_logits.flatten(0, 1),
        input_ids.flatten(0, 1),
        reduction="none",
    ).view_as(input_ids)
    weights = loss_weights.to(token_loss.dtype) * loss_mask.to(token_loss.dtype) * valid.to(token_loss.dtype)
    return (token_loss * weights).sum()
