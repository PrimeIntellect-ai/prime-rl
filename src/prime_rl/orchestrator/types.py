from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class TensorizedRollout:
    """Represents a the tensorized version of a single rollout."""

    input_ids: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    advantages: Float[Tensor, "seq"]
    logprobs: Float[Tensor, "seq"]
    loss_mask: Int[Tensor, "seq"]


@dataclass
class Batch:
    """Represent a training-ready (micro) batch of rollouts."""

    # Token level
    input_ids: Int[Tensor, "bs seq"]
    position_ids: Int[Tensor, "bs seq"]
    advantages: Float[Tensor, "bs seq"]
    logprobs: Float[Tensor, "bs seq"]
    loss_mask: Int[Tensor, "bs seq"]

    # Batch level
    temperature: float
    total_tokens: int


@dataclass
class Rollout:
    """Represents the raw output of a single rollout."""

    problem_id: int
    prompt_tokens: list[int]
    prompt_mask: list[int]
    completion_tokens: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    reward: float
    advantage: float

    @staticmethod
    def to_tensor(self) -> TensorizedRollout:
        # Prepare prompt tokens
        prompt_token_ids = torch.tensor(self.prompt_tokens).long()
        prompt_token_mask = torch.tensor(self.prompt_mask).long()

        # Prepare completion tokens
        completion_token_ids = torch.tensor(self.completion_tokens).long()
        completion_token_mask = torch.tensor(self.completion_mask).long()
        completion_logprobs = torch.tensor(self.completion_logprobs).float()

        # Prepare input_ids, loss_mask, position_ids, logprobs, and advantages
        input_ids = torch.cat([prompt_token_ids, completion_token_ids])
        assert input_ids.dtype == prompt_token_ids.dtype == completion_token_ids.dtype
        position_ids = torch.arange(len(input_ids)).long()
        loss_mask = torch.cat([prompt_token_mask, completion_token_mask]).long()
        assert loss_mask.dtype == prompt_token_mask.dtype == completion_token_mask.dtype
        logprobs = torch.cat([torch.zeros(len(prompt_token_ids), dtype=completion_logprobs.dtype), completion_logprobs])
        assert logprobs.dtype == completion_logprobs.dtype
        advantages = torch.tensor(self.advantage).repeat(len(input_ids)).float()

        return TensorizedRollout(
            input_ids=input_ids,
            position_ids=position_ids,
            advantages=advantages,
            logprobs=logprobs,
            loss_mask=loss_mask,
        )
