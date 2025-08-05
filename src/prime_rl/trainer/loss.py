from dataclasses import dataclass

import torch
import torch.distributed as dist
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.nn import functional as F

from prime_rl.trainer.config import LossConfig
from prime_rl.trainer.model import Model, forward


@dataclass
class RatioInfo:
    ratio_sum: Float[Tensor, "1"]
    clipped_token_count: Float[Tensor, "1"]

    raw_ratio_sum: Float[Tensor, "1"]
    raw_ratio_max: Float[Tensor, "1"]
    raw_ratio_min: Float[Tensor, "1"]

    raw_ratio_abs_sum: Float[Tensor, "1"]


@jaxtyped(typechecker=typechecker)
def grpo_loss(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    loss_config: LossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    if loss_config.type == "clip":
        return grpo_loss_clip(
            shifted_logits=shifted_logits,
            input_ids=input_ids,
            advantages=advantages,
            original_logprobs=original_logprobs,
            loss_mask=loss_mask,
            temperature=temperature,
            epsilon_low=loss_config.epsilon_low,
            epsilon_high=loss_config.epsilon_high,
            clip_ratio=loss_config.clip_ratio,
        )
    elif loss_config.type == "ratio":
        return grpo_loss_ratio(
            shifted_logits=shifted_logits,
            input_ids=input_ids,
            advantages=advantages,
            original_logprobs=original_logprobs,
            loss_mask=loss_mask,
            temperature=temperature,
            clip_ratio=loss_config.clip_ratio,
        )


@jaxtyped(typechecker=typechecker)
def grpo_loss_clip(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    clip_ratio: float,
) -> tuple[Tensor, RatioInfo]:
    assert shifted_logits.dtype == torch.float32, "shifted_logits must be float32"
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    raw_ratio = torch.exp(per_token_logps - original_logprobs)

    coef_1 = torch.clamp(raw_ratio, 0, clip_ratio)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clipped_token_count = _masked_sum(is_clipped, loss_mask)

    loss = _masked_sum(per_token_loss, loss_mask)

    raw_ratio = (raw_ratio.detach() - 1) * loss_mask
    ratio = (coef_2.detach() - 1) * loss_mask

    return loss, RatioInfo(
        ratio_sum=ratio.sum().float(),
        clipped_token_count=clipped_token_count,
        raw_ratio_sum=raw_ratio.sum().float(),
        raw_ratio_max=raw_ratio.max().float() + 1,
        raw_ratio_min=raw_ratio.min().float() + 1,
        raw_ratio_abs_sum=raw_ratio.abs().sum().float(),
    )


@jaxtyped(typechecker=typechecker)
def grpo_loss_ratio(
    shifted_logits: Float[Tensor, "B L V"],
    input_ids: Int[Tensor, "B L"],
    original_logprobs: Float[Tensor, "B L"],
    advantages: Float[Tensor, "B L"],
    loss_mask: Int[Tensor, "B L"],
    clip_ratio: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    # Compute the logprobs
    logprobs = selective_log_softmax(shifted_logits, input_ids)

    # Compute the per-token loss
    logprob_ratio = torch.exp(logprobs - original_logprobs) * loss_mask  # (B, L)
    clipped_logprob_ratio = torch.clamp(logprob_ratio, 0, clip_ratio)  # (B, L)
    loss = -clipped_logprob_ratio * advantages  # (B, L)
    with torch.no_grad():
        is_clipped = (logprob_ratio > clip_ratio).float()  # (B, L)
        entropy = compute_entropy(shifted_logits, loss_mask)

    # Sum-reduce the loss for all unmasked tokens
    reduced_loss = _masked_sum(loss, loss_mask)

    return reduced_loss, {
        "loss": loss.detach().cpu(),
        "logprob_ratio": logprob_ratio.detach().cpu(),
        "clipped_logprob_ratio": clipped_logprob_ratio.detach().cpu(),
        "is_clipped": is_clipped.detach().cpu(),
        "entropy": entropy.detach().cpu(),
    }


@jaxtyped(typechecker=typechecker)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    """
    credits to https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/utils.py#L1659

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


@jaxtyped(typechecker=typechecker)
def compute_logprobs(
    model: Model,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    temperature: float,
) -> Float[Tensor, "batch seq"]:
    logits = forward(model, input_ids, position_ids).contiguous()
    shifted_logits = shift_logits(logits)
    shifted_logits = shifted_logits / temperature
    logprobs = selective_log_softmax(shifted_logits, input_ids)
    del logits, shifted_logits
    return logprobs


@jaxtyped(typechecker=typechecker)
def compute_entropy(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    loss_mask: Int[Tensor, "batch seq"],
) -> Tensor:
    pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
    entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)

    return _masked_sum(entropy, loss_mask)


def _masked_sum(tensor: Tensor, mask: Tensor) -> Tensor:
    """Sums over the unmasked tensor values"""
    return (tensor * mask).sum()


@jaxtyped(typechecker=typechecker)
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    B, _, V = logits.shape
    logits = logits[:, :-1, :]  # (B, L-1, V)
    zeros = torch.zeros(B, 1, V, device=logits.device, dtype=logits.dtype)  # (B, 1, V)
    logits = torch.cat([zeros, logits], dim=1)  # (B, L, V)
    return logits


class ImportanceRatioMetrics:
    """
    This class is used to compute the importance ratio metrics

    The importance ratio metrics are computed as follows:
    - error_sum: sum of the importance ratio error. Error is above or below 1
    - raw_error_sum: sum of the raw importance ratio error
    - max: max of the raw importance ratio
    - min: min of the raw importance ratio
    - clipped: clipped percentage of the importance ratio. This is the percentage of tokens that were clipped
    - ratio: ratio of the importance ratio. This is the ratio after clipping
    - raw_ratio: raw ratio of the importance ratio. This is the ratio before clipping
    """

    def __init__(self):
        self.error_sum = torch.tensor(0.0).to("cuda")
        self.raw_error_sum = torch.tensor(0.0).to("cuda")
        self.max = torch.tensor(0.0).to("cuda")
        self.min = torch.tensor(float("inf")).to("cuda")
        self.clipped = torch.tensor(0.0).to("cuda")
        self.ratio = torch.tensor(0.0).to("cuda")
        self.raw_ratio = torch.tensor(0.0).to("cuda")

        self.raw_abs_error_sum = torch.tensor(0.0).to("cuda")

    def update(self, ratio_info: RatioInfo):
        self.error_sum += ratio_info.ratio_sum.detach().float()
        self.raw_error_sum += ratio_info.raw_ratio_sum.detach().float()
        self.raw_abs_error_sum += ratio_info.raw_ratio_abs_sum.detach().float()
        self.max = torch.max(self.max, ratio_info.raw_ratio_max.detach().float())
        self.min = torch.min(self.min, ratio_info.raw_ratio_min.detach().float())
        self.clipped += ratio_info.clipped_token_count.detach().float()

    def sync(self, total_non_masked_tokens: Tensor, loss_scale: float):
        """
        Sync the importance ratio metrics across all ranks.
        """
        self.clipped = self.clipped / loss_scale
        dist.all_reduce(self.clipped, op=dist.ReduceOp.AVG)
        dist.all_reduce(self.max, op=dist.ReduceOp.MAX)
        dist.all_reduce(self.min, op=dist.ReduceOp.MIN)
        dist.all_reduce(self.error_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.raw_error_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.raw_abs_error_sum, op=dist.ReduceOp.SUM)

        self.ratio = (total_non_masked_tokens + self.error_sum) / total_non_masked_tokens
        self.raw_ratio = (total_non_masked_tokens + self.raw_error_sum) / total_non_masked_tokens

    def to_dict(self) -> dict[str, float]:
        """
        return a dict of float values (could be used to log to wandb)
        """
        return {
            "importance_ratio/error_sum": self.error_sum.item(),
            "importance_ratio/raw_error_sum": self.raw_error_sum.item(),
            "importance_ratio/max": self.max.item(),
            "importance_ratio/min": self.min.item() if self.min != float("inf") else 0.0,
            "importance_ratio/clipped": self.clipped.item(),
            "importance_ratio/ratio": self.ratio.item(),
            "importance_ratio/raw_ratio": self.raw_ratio.item(),
            "importance_ratio/raw_abs_error_sum": self.raw_abs_error_sum.item(),
        }
