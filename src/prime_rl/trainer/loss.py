import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.nn import functional as F

from prime_rl.trainer.config import ClippingConfig, GRPOVariantsConfig, GSPOConfig, RatioConfig
from prime_rl.trainer.model import Model, forward


@jaxtyped(typechecker=typechecker)
def grpo_loss(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    grpo_loss_config: GRPOVariantsConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(grpo_loss_config, ClippingConfig):
        return grpo_loss_clip(
            shifted_logits=shifted_logits,
            input_ids=input_ids,
            advantages=advantages,
            original_logprobs=original_logprobs,
            loss_mask=loss_mask,
            temperature=temperature,
            epsilon_low=grpo_loss_config.epsilon_low,
            epsilon_high=grpo_loss_config.epsilon_high,
            clip_ratio=grpo_loss_config.clip_ratio,
            highest_entropy_percentage=grpo_loss_config.highest_entropy_ratio_loss,
        )
    elif isinstance(grpo_loss_config, RatioConfig):
        return grpo_loss_ratio(
            shifted_logits=shifted_logits,
            input_ids=input_ids,
            advantages=advantages,
            original_logprobs=original_logprobs,
            loss_mask=loss_mask,
            temperature=temperature,
            clip_ratio=grpo_loss_config.clip_ratio,
            highest_entropy_percentage=grpo_loss_config.highest_entropy_ratio_loss,
        )
    elif isinstance(grpo_loss_config, GSPOConfig):
        return gspo_loss(
            shifted_logits=shifted_logits,
            input_ids=input_ids,
            advantages=advantages,
            original_logprobs=original_logprobs,
            loss_mask=loss_mask,
            temperature=temperature,
            epsilon=grpo_loss_config.epsilon,
        )
    else:
        raise ValueError(f"Invalid grpo_loss_type: {grpo_loss_config.type}")


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
    highest_entropy_percentage: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    DeepSeek Math Loss: https://arxiv.org/abs/2402.03300

    Args:
        policy_logprobs: Log probabilities from the policy model
        ref_logprobs: Log probabilities from the reference model
        advantages: Advantages for each token
        beta: KL penalty coefficient
        epsilon: Clipping parameter for PPO
        ignore_index: Specifies a target value that is ignored and does not contribute to the loss
    """
    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    coef_1 = torch.clamp(torch.exp(per_token_logps - original_logprobs), 0, clip_ratio)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clipped_token_count = _masked_sum(is_clipped, loss_mask)

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(shifted_logits, loss_mask, highest_entropy_percentage)

    loss = _masked_sum(per_token_loss, loss_mask)
    ratio = _masked_sum(coef_2, loss_mask)
    return loss, ratio, clipped_token_count


@jaxtyped(typechecker=typechecker)
def grpo_loss_ratio(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    clip_ratio: float,
    highest_entropy_percentage: float,
) -> tuple[Tensor, Tensor, Tensor]:
    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    raw_ratio = torch.exp(per_token_logps - original_logprobs)

    is_clipped = (raw_ratio > clip_ratio).float()
    clipped_token_count = _masked_sum(is_clipped, loss_mask)

    ratio = torch.clamp(raw_ratio, 0, clip_ratio)
    loss = -ratio * advantages

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(shifted_logits, loss_mask, highest_entropy_percentage)

    loss = _masked_sum(loss, loss_mask)
    ratio = _masked_sum(ratio, loss_mask)

    return loss, ratio, clipped_token_count


@jaxtyped(typechecker=typechecker)
def gspo_loss(
    shifted_logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    shifted_logits = shifted_logits / temperature
    per_token_logps = selective_log_softmax(shifted_logits, input_ids)

    sequence_level_log_likelihood = (per_token_logps * loss_mask).sum(-1)
    original_sequence_level_log_likelihood = (original_logprobs * loss_mask).sum(-1)
    seq_len = loss_mask.sum(-1).clamp(min=1)

    sequence_level_importance_ratio = torch.exp(
        (sequence_level_log_likelihood - original_sequence_level_log_likelihood) / seq_len
    )
    sequence_level_importance_ratio_clipped = torch.clamp(sequence_level_importance_ratio, 1 - epsilon, 1 + epsilon)

    is_clipped = (sequence_level_importance_ratio > sequence_level_importance_ratio_clipped).float()
    clipped_token_count = _masked_sum(is_clipped, loss_mask)

    loss = -sequence_level_importance_ratio_clipped * advantages

    loss = _masked_sum(loss, loss_mask)
    ratio = _masked_sum(sequence_level_importance_ratio_clipped, loss_mask)

    return loss, ratio, clipped_token_count


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
def highest_entropy_mask(
    logits: Float[Tensor, "batch seq vocab"],
    loss_mask: Int[Tensor, "batch seq"],
    percent: float,
) -> Tensor:
    """
    Returns a mask (batch, seq) where the top `percent` of masked tokens (loss_mask==1)
    with the highest entropy are 1, others 0.
    Args:
        logits: Tensor of shape (batch, seq, vocab)
        loss_mask: Tensor of shape (batch, seq), 1 for valid tokens, 0 for padding
        percent: float in (0, 1), e.g., 0.2 for top 20%
        temperature: float, temperature for softmax (default 1.0)
    Returns:
        mask: Tensor of shape (batch, seq), dtype=torch.bool
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)  # (batch, seq)

    valid_entropy = entropy[loss_mask.bool()]
    k = int(percent * valid_entropy.numel())
    if k < 1:
        k = 1
    if k == valid_entropy.numel():
        threshold = valid_entropy.min() - 1  # all True
    else:
        threshold = torch.kthvalue(valid_entropy, valid_entropy.numel() - k + 1).values

    mask = (entropy >= threshold) & (loss_mask.bool())
    return mask


@jaxtyped(typechecker=typechecker)
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    B, _, V = logits.shape
    logits = logits[:, :-1, :]  # (B, L-1, V)
    zeros = torch.zeros(B, 1, V, device=logits.device, dtype=logits.dtype)  # (B, 1, V)
    logits = torch.cat([zeros, logits], dim=1)  # (B, L, V)
    return logits
