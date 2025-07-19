import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from prime_rl.trainer.config import ClippingConfig, GRPOVariantsConfig, RatioConfig
from prime_rl.trainer.model import selective_log_softmax


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


def _masked_sum(tensor: Tensor, mask: Tensor) -> Tensor:
    """Sums over the unmasked tensor values"""
    return (tensor * mask).sum()


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
