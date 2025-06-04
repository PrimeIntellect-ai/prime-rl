import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    clamp_log_prob_coef: float,
    max_tokens: int,
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
    # original_logprobs = original_logprobs[:, 1:] # no need to do it now
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    coef_1 = torch.clamp(torch.exp(per_token_logps - original_logprobs), 0, clamp_log_prob_coef)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    loss = _apply_mask(per_token_loss, loss_mask, max_tokens)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clip_ratio = _apply_mask(is_clipped, loss_mask, max_tokens)
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
) -> tuple[Tensor, Tensor | None]:
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = input_ids[:, 1:]
    advantages = advantages[:, 1:]
    # original_logprobs = original_logprobs[:, 1:] # no need to do it now
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    ratio = torch.exp(per_token_logps - original_logprobs)

    per_token_loss = -ratio * advantages

    loss = _apply_mask(per_token_loss, loss_mask, max_tokens)

    return loss, None


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


@jaxtyped(typechecker=typechecker)
def entropy_loss(
    logits: Float[Tensor, "batch seq vocab"], loss_mask: Int[Tensor, "batch seq"], temperature: float, max_tokens: int
) -> Tensor:
    return _compile_entropy_loss(logits=logits, loss_mask=loss_mask, temperature=temperature, max_tokens=max_tokens)


# @torch.compile
def _compile_entropy_loss(logits: torch.Tensor, loss_mask: torch.Tensor, temperature: float, max_tokens: int):
    logits = logits[:, :-1, :]
    logits = logits / temperature

    loss_mask = loss_mask[:, 1:]
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

    return _apply_mask(entropy, loss_mask, max_tokens)


@jaxtyped(typechecker=typechecker)
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
