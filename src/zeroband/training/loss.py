import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
import torch.nn.functional as F

from zeroband.training.verl_utils import logprobs_from_logits, masked_mean


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    epsilon: float,
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
    log_probs = grpo_logprobs(logits, input_ids, temperature, epsilon)

    advantages = advantages[:, 1:]
    loss_mask = loss_mask[:, 1:]

    pg_loss, pg_clipfrac, _ = compute_policy_loss(
        old_log_prob=original_logprobs, log_prob=log_probs, advantages=advantages, eos_mask=loss_mask, cliprange=epsilon
    )
    return pg_loss, pg_clipfrac

@jaxtyped(typechecker=typechecker)
def grpo_logprobs(logits: Float[Tensor, "batch seq vocab"], input_ids: Int[Tensor, "batch seq"], temperature: float, epsilon: float) -> Float[Tensor, "batch seq_minus_1"]:
    """
    Compute the log probabilities of the actions taken by the policy.
    """
    input_ids = input_ids[:, 1:]
    logits.div_(temperature)
    response_length = logits.shape[1]
    logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
    return logprobs_from_logits(logits, input_ids)
    

def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


@jaxtyped(typechecker=typechecker)
def entropy_loss(logits: Float[Tensor, "batch seq vocab"], loss_mask: Int[Tensor, "batch seq"], temperature: float) -> Tensor:
    return _compile_entropy_loss(logits=logits, loss_mask=loss_mask, temperature=temperature)


@torch.compile
def _compile_entropy_loss(logits: torch.Tensor, loss_mask: torch.Tensor, temperature: float):
    logits = logits[:, :-1, :]
    logits = logits / temperature

    loss_mask = loss_mask[:, 1:]
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    masked_entropy = entropy * loss_mask

    return masked_entropy.sum() / loss_mask.sum()