import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.nn import functional as F

from prime_rl.trainer.config import LossConfig
from prime_rl.trainer.model import Model, forward


@jaxtyped(typechecker=typechecker)
def grpo_loss(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    loss_config: LossConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    
    if loss_config.type == "clip-grpo":
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
    elif loss_config.type == "ratio-grpo":
        return grpo_loss_ratio(
            shifted_logits=shifted_logits,
            input_ids=input_ids,
            advantages=advantages,
            original_logprobs=original_logprobs,
            loss_mask=loss_mask,
            temperature=temperature,
            clip_ratio=loss_config.clip_ratio,
        )
    elif loss_config.type == "clip-gspo":
        return gspo_loss_clip(
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
    elif loss_config.type == "ratio-gspo":  
        return gspo_loss_ratio(
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
) -> tuple[Tensor, Tensor, Tensor]:
    
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    coef_1 = torch.clamp(torch.exp(per_token_logps - original_logprobs), 0, clip_ratio)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clipped_count = _masked_sum(is_clipped, loss_mask)

    loss = _masked_sum(per_token_loss, loss_mask)
    ratio = _masked_sum(coef_1, loss_mask)
    return loss, ratio, clipped_count


@jaxtyped(typechecker=typechecker)
def grpo_loss_ratio(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    clip_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    raw_ratio = torch.exp(per_token_logps - original_logprobs)

    is_clipped = (raw_ratio > clip_ratio).float()
    clipped_count = _masked_sum(is_clipped, loss_mask)

    ratio = torch.clamp(raw_ratio, 0, clip_ratio)
    loss = -ratio * advantages

    loss = _masked_sum(loss, loss_mask)
    ratio = _masked_sum(raw_ratio, loss_mask)

    return loss, ratio, clipped_count

@jaxtyped(typechecker=typechecker)
def gspo_loss_clip(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    clip_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    # compute sequence-level importance ratio by averaging over the tokens
    coef_1 = ((per_token_logps - original_logprobs) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)

    coef_1 = per_token_logps - per_token_logps.detach() + coef_1.detach().unsqueeze(-1)
    coef_1 = torch.clamp(coef_1.clamp(max=10.0).exp(), 0, clip_ratio)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clipped_count = _masked_sum(is_clipped, loss_mask)

    loss = _masked_sum(per_token_loss, loss_mask) / loss_mask.sum()
    ratio = _masked_sum(coef_1, loss_mask)
    return loss, ratio, clipped_count


@jaxtyped(typechecker=typechecker)
def gspo_loss_ratio(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    clip_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    # compute sequence-level importance ratio by averaging over the tokens
    raw_seq_ratio = ((per_token_logps - original_logprobs) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)

    raw_ratio = per_token_logps - per_token_logps.detach() + raw_seq_ratio.detach().unsqueeze(-1)
    raw_ratio = raw_ratio.clamp(max=10.0).exp()

    is_clipped = (raw_ratio > clip_ratio).float()
    clipped_count = _masked_sum(is_clipped, loss_mask)

    ratio = torch.clamp(raw_ratio, 0, clip_ratio)
    loss = -ratio * advantages

    loss = _masked_sum(loss, loss_mask) / loss_mask.sum()
    ratio = _masked_sum(raw_ratio, loss_mask)

    return loss, ratio, clipped_count

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
    temperature: float,
) -> Tensor:
    shifted_logits = shifted_logits / temperature
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
