from dataclasses import dataclass
from typing import Callable

from jaxtyping import Bool, Float
from torch import Tensor


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample."""

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    teacher_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""
