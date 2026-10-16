"""Activation-gradient clamp of Step 5 Preview (StepFun, 2026): rescale the few massive per-token gradient vectors
of the residual stream at each decoder layer's output in the backward pass."""

import torch
from torch import Tensor, nn


class GradVectorClamp(torch.autograd.Function):
    """Identity in the forward pass. In the backward pass, every token's gradient vector whose L2 norm exceeds
    ``factor`` times the median norm over the tokens is scaled down to that norm. Records the largest norm over
    the median (max over calls) and the number of clamped vectors (sum) in ``stats``."""

    @staticmethod
    def forward(ctx, hidden: Tensor, factor: float, stats: Tensor) -> Tensor:  # type: ignore[override]
        ctx.factor, ctx.stats = factor, stats
        return hidden.view_as(hidden)

    @staticmethod
    def backward(ctx, grad: Tensor):  # type: ignore[override]
        norms = grad.float().norm(dim=-1)
        median = norms.median()
        limit = ctx.factor * median
        ctx.stats[0] = torch.maximum(ctx.stats[0], norms.amax() / median)
        ctx.stats[1] += (norms > limit).sum()
        if ctx.factor != float("inf"):
            grad = grad * (limit / norms).clamp(max=1.0).unsqueeze(-1).to(grad.dtype)
        return grad, None, None


def install_grad_clamp(layers: nn.ModuleList, factor: float) -> Tensor:
    """Clamp the output gradient of every layer. Returns the ``[layers, 2]`` statistics (max norm over median,
    clamped vectors) that accumulate until the caller resets them."""
    stats = torch.zeros(len(layers), 2, device="cuda")
    for index, layer in enumerate(layers):
        # Outside the compiled layer: the backward then runs eagerly as its own autograd node.
        @torch.compiler.disable
        def clamp(module, args, output, row=stats[index]):
            return GradVectorClamp.apply(output, factor, row)

        layer.register_forward_hook(clamp)
    return stats
