import torch

from prime_rl.trainer.models.layers.activations import Activation


def differentiable_expert_ffn(
    hidden: torch.Tensor,
    offs: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation: type[Activation],
    gate_proj: torch.Tensor | None = None,
) -> torch.Tensor:
    up = torch._grouped_mm(hidden, up_proj.transpose(-2, -1), offs=offs)
    gate = torch._grouped_mm(hidden, gate_proj.transpose(-2, -1), offs=offs) if gate_proj is not None else None
    act = activation.apply(gate, up).to(hidden.dtype)
    return torch._grouped_mm(act, down_proj.transpose(-2, -1), offs=offs)
