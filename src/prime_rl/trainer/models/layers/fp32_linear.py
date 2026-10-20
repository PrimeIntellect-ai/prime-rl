import torch
import torch.nn.functional as F


class _FP32OutputMatmul(torch.autograd.Function):
    """Low-precision operands with an FP32 output and FP32 gradient accumulation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        # PyTorch's mm(out_dtype=...) does not implement autograd.
        return torch.mm(x, weight.T, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x, weight = ctx.saved_tensors
        with torch.autocast(device_type=x.device.type, enabled=False):
            grad_output = grad_output.float()
            grad_x = (grad_output @ weight.float()).to(x.dtype) if ctx.needs_input_grad[0] else None
            grad_weight = (grad_output.T @ x.float()).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return grad_x, grad_weight


def fp32_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Cast inputs to the weight dtype and return FP32 output without an intermediate downcast."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.to(weight.dtype)
        if x.is_cuda and weight.dtype in (torch.bfloat16, torch.float16):
            output = _FP32OutputMatmul.apply(x.reshape(-1, x.shape[-1]), weight)
            output = output.reshape(*x.shape[:-1], weight.shape[0])
            return output if bias is None else output + bias.float()
        return F.linear(x.float(), weight.float(), bias.float() if bias is not None else None)


class FP32Linear(torch.nn.Linear):
    """A linear layer whose output stays in FP32 regardless of its parameter dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp32_linear(x, self.weight, self.bias)
