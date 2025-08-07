import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.nn import functional as F

from prime_rl.trainer.config import LossConfig


@jaxtyped(typechecker=typechecker)
def compute_loss(
    logprobs: Float[Tensor, "batch seq"],
    old_logprobs: Float[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    loss_config: LossConfig,
) -> tuple[Float[Tensor, "batch seq"], dict[str, Float[Tensor, "batch seq"]]]:
    if loss_config.type == "clip":
        return grpo_loss_clip(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            epsilon_low=loss_config.epsilon_low,
            epsilon_high=loss_config.epsilon_high,
            clip_ratio=loss_config.clip_ratio,
        )
    elif loss_config.type == "ratio":
        return grpo_loss_ratio(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_ratio=loss_config.clip_ratio,
        )


@jaxtyped(typechecker=typechecker)
def grpo_loss_clip(
    logprobs: Float[Tensor, "batch seq"],
    old_logprobs: Float[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    epsilon_low: float,
    epsilon_high: float,
    clip_ratio: float,
) -> tuple[Float[Tensor, "batch seq"], dict[str, Float[Tensor, "batch seq"]]]:
    assert logprobs.dtype == torch.float32, "logprobs must be float32"
    assert old_logprobs.dtype == torch.float32, "old_logprobs must be float32"
    assert advantages.dtype == torch.float32, "advantages must be float32"

    # Compute the per-token loss
    importance_ratio = torch.exp(logprobs - old_logprobs)
    coef_1 = torch.clamp(importance_ratio, 0, clip_ratio)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    loss_1 = -coef_1 * advantages
    loss_2 = -coef_2 * advantages
    loss = torch.max(loss_1, loss_2)
    is_clipped = (loss_1 < loss_2).float()

    return loss, {
        "loss": loss,
        "coef_1": coef_1,
        "coef_2": coef_2,
        "is_clipped": is_clipped,
    }


@jaxtyped(typechecker=typechecker)
def grpo_loss_ratio(
    logprobs: Float[Tensor, "batch seq"],
    old_logprobs: Float[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    clip_ratio: float,
) -> tuple[Float[Tensor, "batch seq"], dict[str, Float[Tensor, "batch seq"]]]:
    assert logprobs.dtype == torch.float32, "logprobs must be float32"
    assert old_logprobs.dtype == torch.float32, "old_logprobs must be float32"
    assert advantages.dtype == torch.float32, "advantages must be float32"

    # Compute the per-token loss
    importance_ratio = torch.exp(logprobs - old_logprobs)  # (B, L)
    clipped_importance_ratio = torch.clamp(importance_ratio, 0, clip_ratio)  # (B, L)
    loss = -clipped_importance_ratio * advantages  # (B, L)
    is_clipped = (importance_ratio > clip_ratio).float()  # (B, L)

    return loss, {
        "loss": loss,
        "importance_ratio": importance_ratio,
        "clipped_importance_ratio": clipped_importance_ratio,
        "is_clipped": is_clipped,
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
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
    entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)

    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    zeros = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([zeros, logits], dim=1)  # (batch, seq, vocab)
    return logits
