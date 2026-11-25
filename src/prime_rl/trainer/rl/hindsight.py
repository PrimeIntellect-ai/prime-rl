"""Goal-conditioned supervised learning on achieved-goal action sequences."""

from torch import Tensor

from prime_rl.trainer.rl.supervised import gather_action_logprobs, summed_weighted_nll


def hindsight_loss(legal_logits: Tensor, action_columns: Tensor, episode_count: int) -> Tensor:
    """Sum legal-action NLL over eligible rows and divide by all original episodes.

    Logits have shape [eligible episode, step, legal action], with temperature already applied.
    Action columns index the environment's legal token ordering. Each eligible row is a fresh
    student forward under its achieved goal. Divide by the fixed horizon for a per-decision mean.
    """
    if legal_logits.ndim != 3 or min(legal_logits.shape[1:]) < 1 or legal_logits.shape[:2] != action_columns.shape:
        raise ValueError("Expected aligned [episode, step, action] logits and [episode, step] actions")
    if episode_count < 1 or episode_count < legal_logits.shape[0]:
        raise ValueError("The original episode count must be positive and include every eligible episode")
    logprobs = gather_action_logprobs(legal_logits.float(), action_columns)
    return summed_weighted_nll(logprobs, None, None) / episode_count
