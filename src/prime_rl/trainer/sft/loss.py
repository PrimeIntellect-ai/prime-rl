import torch
from torch import Tensor


def masked_token_loss_stats(token_losses: Tensor, loss_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    loss_mask = loss_mask.to(dtype=torch.bool)
    loss_sum = token_losses.masked_select(loss_mask).sum()
    token_count = loss_mask.sum(dtype=torch.int64)
    loss_mean = loss_sum / torch.clamp_min(token_count, 1).to(dtype=loss_sum.dtype)
    return loss_sum, token_count, loss_mean


def gradient_scale_factor(global_token_count: Tensor, gradient_divide_factor: int) -> Tensor:
    denom = torch.clamp_min(global_token_count, 1).to(dtype=torch.float32)
    return torch.tensor(float(gradient_divide_factor), device=denom.device, dtype=torch.float32) / denom


def token_weighted_mean_loss(global_loss_sum: Tensor, global_token_count: Tensor) -> Tensor:
    denom = torch.clamp_min(global_token_count, 1).to(dtype=global_loss_sum.dtype)
    return global_loss_sum / denom
