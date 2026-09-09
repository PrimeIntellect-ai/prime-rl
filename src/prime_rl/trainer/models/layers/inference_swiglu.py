"""vLLM's BF16 SwiGLU forward with the eager two-stage PyTorch derivative."""

import torch
import torch.nn.functional as F
from torch import Tensor


class _InferenceSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        ctx.save_for_backward(gate, up)
        output = torch.empty_like(gate, memory_format=torch.contiguous_format)
        torch.ops._C.silu_and_mul(output, torch.cat((gate, up), dim=-1))
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        gate, up = ctx.saved_tensors
        return torch.ops.aten.silu_backward(grad_output * up, gate), grad_output * F.silu(gate)


class InferenceSilu:
    @staticmethod
    @torch.compiler.disable
    def apply(gate: Tensor | None, up: Tensor) -> Tensor:
        # A graph break preserves the inference kernel's intermediate rounding.
        if gate is None:
            return F.silu(up)
        return _InferenceSwiGLU.apply(gate, up)


def enable_inference_swiglu(model: torch.nn.Module) -> int:
    from vllm.platforms import current_platform

    from prime_rl.trainer.models.layers.activations import Silu
    from prime_rl.trainer.models.layers.mlp import FeedForward

    current_platform.import_kernels()
    count = 0
    for module in model.modules():
        if isinstance(module, FeedForward) and module.gate_proj is not None and module.activation is Silu:
            module.activation = InferenceSilu
            count += 1
    if not count:
        raise ValueError("inference_swiglu requires unfused dense SiLU FeedForward modules")
    return count
