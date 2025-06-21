import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from zeroband.training.config import ClippingConfig, GRPOVariantsConfig, KlCovConfig, RatioConfig


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
) -> tuple[Tensor, Tensor | None]:
    if isinstance(grpo_loss_config, ClippingConfig):
        return grpo_loss_clip(
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
    elif isinstance(grpo_loss_config, RatioConfig):
        return grpo_loss_ratio(
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

    elif isinstance(grpo_loss_config, KlCovConfig):
        return grpo_loss_kl_cov(
            logits,
            input_ids,
            advantages,
            original_logprobs,
            loss_mask,
            temperature,
            max_tokens,
            grpo_loss_config.kl_coef,
            grpo_loss_config.k_percent,
            grpo_loss_config.highest_entropy_ratio_loss,
        )
    else:
        raise ValueError(f"Invalid grpo_loss_type: {grpo_loss_config.type}")


# @jaxtyped(typechecker=typechecker)
@torch.compile
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


# @jaxtyped(typechecker=typechecker)
@torch.compile
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
) -> tuple[Tensor, Tensor | None]:
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


# @jaxtyped(typechecker=typechecker)
@torch.compile
def grpo_loss_kl_cov(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    max_tokens: int,
    kl_coef_cov: float,
    k_percent: float,
    highest_entropy_percentage: float,
) -> tuple[Tensor, Tensor | None]:
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

    negative_approx_kl = per_token_logps - original_logprobs

    abs_kl = negative_approx_kl.abs()

    ratio = torch.exp(negative_approx_kl)

    ppo_kl_abs = (abs_kl * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    pg_losses1 = -advantages * ratio

    pg_losses_kl = -advantages * ratio + kl_coef_cov * abs_kl

    pg_losses = pg_losses1

    all_valid = loss_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = per_token_logps[all_valid].detach().reshape(-1).cpu()

    k = min(k_percent, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * k / 100))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(logits, loss_mask, highest_entropy_percentage)

    pg_loss = _apply_mask(pg_losses, loss_mask, max_tokens)

    return pg_loss, ppo_kl_abs


@torch.compile
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


# @jaxtyped(typechecker=typechecker)
@torch.compile
def entropy_loss(
    logits: Float[Tensor, "batch seq vocab"], loss_mask: Int[Tensor, "batch seq"], temperature: float, max_tokens: int
) -> Tensor:
    logits = logits[:, :-1, :]
    logits = logits / temperature

    loss_mask = loss_mask[:, 1:]
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

    return _apply_mask(entropy, loss_mask, max_tokens)


# @jaxtyped(typechecker=typechecker)
@torch.compile
def kl_penalty(
    logprob: Float[Tensor, "batch seq_minus_1"],
    ref_logprob: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    max_tokens: int,
) -> Float[Tensor, ""]:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py#L351

    Args:
        logprob:
        ref_logprob:

    Returns:

    """

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    loss_mask = loss_mask[:, 1:]

    kl = ref_logprob - logprob
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    kl = torch.clamp(kld, min=-10, max=10)
    return _apply_mask(kl, loss_mask, max_tokens)


def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor, max_tokens: int) -> torch.Tensor:
    return (tensor * mask).sum() / max_tokens


@torch.compile
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
