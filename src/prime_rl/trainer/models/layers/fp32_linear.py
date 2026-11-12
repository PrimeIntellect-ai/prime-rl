import torch
import torch.nn.functional as F


class _FP32OutputMatmul(torch.autograd.Function):
    """BF16 operands with an FP32 output and FP32 gradient accumulation."""

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


def fp32_output_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for 2D operands. CUDA BF16/FP16 operands get an FP32 output from the GEMM instead of one rounded
    to their dtype; others a plain matmul."""
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        return torch.mm(x, weight.t(), out_dtype=torch.float32)
    return x @ weight.t()


class FP32OutputMatmulBF16Backward(torch.autograd.Function):
    """``fp32_output_mm`` with autograd. Unlike ``_FP32OutputMatmul``, the backward casts the output gradient to the
    operands' dtype and runs BF16 GEMMs, which large outputs such as LM-head logits need."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        return fp32_output_mm(x, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x, weight = ctx.saved_tensors
        grad_output = grad_output.to(x.dtype)
        grad_x = grad_output @ weight if ctx.needs_input_grad[0] else None
        grad_weight = grad_output.t() @ x if ctx.needs_input_grad[1] else None
        return grad_x, grad_weight


class FP32Linear(torch.nn.Linear):
    """A linear layer whose output stays in FP32 regardless of its parameter dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(self.weight.dtype)
            if x.is_cuda and self.weight.dtype == torch.bfloat16:
                output = _FP32OutputMatmul.apply(x.reshape(-1, x.shape[-1]), self.weight)
                output = output.reshape(*x.shape[:-1], self.weight.shape[0])
                return output if self.bias is None else output + self.bias.float()
            return F.linear(x.float(), self.weight.float(), self.bias.float() if self.bias is not None else None)
