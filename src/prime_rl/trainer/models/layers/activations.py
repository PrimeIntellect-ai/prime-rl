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


class ClampedSilu(Activation):
    """SwiGLU with the gate clamped to `max=limit` and up to `[-limit, limit]` (DeepSeek V4)."""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def apply(self, gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up


class SoftcappedSwiglu(Activation):
    """SwiGLU whose sigmoid sees the raw gate while softcaps bound the linear gate factor and ``up``
    (Inkling 2: ``tanh(g/4)*4 * sigmoid(g) * tanh(u/25)*25``, computed in fp32)."""

    gate_softcap: float = 4.0
    up_softcap: float = 25.0

    @classmethod
    def apply(cls, gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        dtype = gate.dtype
        gate = gate.float()
        up = up.float()
        out = torch.tanh(gate / cls.gate_softcap) * cls.gate_softcap * torch.sigmoid(gate)
        out = out * (torch.tanh(up / cls.up_softcap) * cls.up_softcap)
        return out.to(dtype)


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


ActivationType = Literal["silu", "relu2", "clamped_swiglu", "softcapped_swiglu"]

ActivationDispatch: dict[ActivationType, type[Activation]] = {
    "clamped_swiglu": ClampedSwiglu,
    "silu": Silu,
    "relu2": Relu2,
    "softcapped_swiglu": SoftcappedSwiglu,
}
