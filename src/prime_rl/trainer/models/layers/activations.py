from typing import Literal, Protocol

import torch
import torch.nn.functional as F


class Activation(Protocol):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor: ...


class Relu2(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        if gate is not None:
            return F.relu(gate).square() * up
        return F.relu(up).square()


class Silu(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        if gate is None:
            return F.silu(up)
        return F.silu(gate) * up


ActivationType = Literal["silu", "relu2"]

ActivationDispatch: dict[ActivationType, type[Activation]] = {
    "silu": Silu,
    "relu2": Relu2,
}
