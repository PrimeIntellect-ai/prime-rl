from typing import TypedDict

from jaxtyping import Bool, Float, Int
from torch import Tensor


class TrainingExample(TypedDict):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    advantage: float | None


class TensorTrainingExample(TypedDict):
    """A single training example as tensors."""

    input_ids: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]
    advantages: Float[Tensor, "seq"]
    inference_logprobs: Float[Tensor, "seq"]
