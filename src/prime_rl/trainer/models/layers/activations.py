from typing import Literal, Protocol

import torch
import torch.nn.functional as F


class Activation(Protocol):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor: ...


class ClampedSwiglu(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        gate = gate.clamp(max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        return (up + 1) * gate * torch.sigmoid(gate * 1.702)


class GlmClampedSwiglu(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        gate = gate.clamp(max=10.0)
        up = up.clamp(min=-10.0, max=10.0)
        return gate * torch.sigmoid(gate) * up


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


ActivationType = Literal["silu", "relu2", "clamped_swiglu", "glm_clamped_swiglu"]

ActivationDispatch: dict[ActivationType, type[Activation]] = {
    "clamped_swiglu": ClampedSwiglu,
    "glm_clamped_swiglu": GlmClampedSwiglu,
    "silu": Silu,
    "relu2": Relu2,
}
