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
    """SwiGLU whose gate and up branches are soft-capped by `cap * tanh(x / cap)` (Inkling 2); a cap of 0
    is no cap. The sigmoid reads the uncapped gate. Computed in fp32 and rounded once."""

    def __init__(self, gate_cap: float, up_cap: float) -> None:
        self.gate_cap = gate_cap
        self.up_cap = up_cap

    @staticmethod
    def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
        return cap * torch.tanh(x / cap) if cap > 0 else x

    def apply(self, gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        gate_fp32, up_fp32 = gate.float(), up.float()
        hidden = self.softcap(gate_fp32, self.gate_cap) * torch.sigmoid(gate_fp32)
        return (hidden * self.softcap(up_fp32, self.up_cap)).to(up.dtype)


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


ActivationType = Literal["silu", "relu2", "clamped_swiglu"]

ActivationDispatch: dict[ActivationType, type[Activation]] = {
    "clamped_swiglu": ClampedSwiglu,
    "silu": Silu,
    "relu2": Relu2,
}
