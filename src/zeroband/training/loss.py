import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from zeroband.training.config import ClippingConfig, GRPOVariantsConfig, RatioConfig
from zeroband.training.model import Model


@jaxtyped(typechecker=typechecker)
def grpo_loss(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    max_tokens: int,
    grpo_loss_config: GRPOVariantsConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    if isinstance(grpo_loss_config, ClippingConfig):
        loss, clip_ratio = grpo_loss_clip(
            logits,
            input_ids,
            advantages,
            original_logprobs,
            loss_mask,
            temperature,
            grpo_loss_config.epsilon_low,
            grpo_loss_config.epsilon_high,
            grpo_loss_config.clip_ratio,
            max_tokens,
            grpo_loss_config.highest_entropy_ratio_loss,
        )
        return loss, {"clip_ratio": clip_ratio}
    elif isinstance(grpo_loss_config, RatioConfig):
        loss, average_ratio = grpo_loss_ratio(
            logits,
            input_ids,
            advantages,
            original_logprobs,
            loss_mask,
            temperature,
            max_tokens,
            grpo_loss_config.clip_ratio,
            grpo_loss_config.highest_entropy_ratio_loss,
        )
        return loss, {"mean_ratio": average_ratio}
    else:
        raise ValueError(f"Invalid grpo_loss_type: {grpo_loss_config.type}")


@jaxtyped(typechecker=typechecker)
def grpo_loss_clip(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    clip_ratio: float,
    max_tokens: int,
    highest_entropy_percentage: float,
) -> tuple[Tensor, Tensor]:
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
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = input_ids[:, 1:]
    advantages = advantages[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    coef_1 = torch.clamp(torch.exp(per_token_logps - original_logprobs), 0, clip_ratio)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clip_ratio = _apply_mask(is_clipped, loss_mask, max_tokens)

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(logits, loss_mask, highest_entropy_percentage)

    loss = _apply_mask(per_token_loss, loss_mask, max_tokens)

    return loss, clip_ratio


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss_ratio(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    max_tokens: int,
    clip_ratio: float,
    highest_entropy_percentage: float,
) -> tuple[Tensor, Tensor]:
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = input_ids[:, 1:]
    advantages = advantages[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    ratio = torch.clamp(torch.exp(per_token_logps - original_logprobs), 0, clip_ratio)

    per_token_loss = -ratio * advantages

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(logits, loss_mask, highest_entropy_percentage)

    loss = _apply_mask(per_token_loss, loss_mask, max_tokens)

    with torch.no_grad():
        ratio_avg = _apply_mask(ratio, loss_mask, max_tokens)

    return loss, ratio_avg


def selective_log_softmax(logits, index):
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


def compute_logprobs(
    model: Model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(
        input_ids=input_ids, position_ids=position_ids
    ).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


@jaxtyped(typechecker=typechecker)
def entropy_loss(
    logits: Float[Tensor, "batch seq vocab"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    max_tokens: int,
) -> Tensor:
    logits = logits[:, :-1, :]
    logits = logits / temperature

    loss_mask = loss_mask[:, 1:]
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

    return _apply_mask(entropy, loss_mask, max_tokens)


def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor, max_tokens: int) -> torch.Tensor:
    return (tensor * mask).sum() / max_tokens


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
