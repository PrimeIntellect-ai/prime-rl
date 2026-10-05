import torch

from prime_rl.experimental.quant_ckpt.quant_cache_tensor import QuantCacheTensor

_calls: dict[str, int] = {}


@torch.library.custom_op("proto::fake_quantize", mutates_args=())
def fake_quantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    _calls["fake_quantize"] = _calls.get("fake_quantize", 0) + 1
    return (x * scale).to(torch.float16)


@fake_quantize.register_fake
def _(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=torch.float16)  # meta only — must NOT touch _calls


def _setup_ctx(ctx, inputs, output):
    ctx.input_dtype = inputs[0].dtype


def _backward(ctx, grad_output):
    return grad_output.to(ctx.input_dtype), None  # straight-through estimator


fake_quantize.register_autograd(_backward, setup_context=_setup_ctx)
QuantCacheTensor.register_cacheable_op(torch.ops.proto.fake_quantize.default, key_fn=lambda args, kwargs: args[1])


# Realistic case: an unfused SwiGLU MLP, w1 (gate_proj) and w3 (up_proj) both consuming
# the same input activation h. In fp8_linear.py's Float8BlockwiseLinear, x is quantized
# once per GEMM layout it's needed in: a row-major cast for the forward matmul (Y=X@W),
# and a genuinely *different* op — a transposed-layout cast — for the weight-gradient
# matmul (dW=dY^T@X). So w1 and w3 each create two independent dedup opportunities: their
# forwards both want the row-major cast of h, and their backwards (wgrad) both want the
# transposed-layout cast of h. These are two separate cache entries (different op key),
# not one merged count.


@torch.library.custom_op("proto::fake_quantize_fwd", mutates_args=())
def fake_quantize_fwd(x: torch.Tensor, scale: float) -> torch.Tensor:
    _calls["fake_quantize_fwd"] = _calls.get("fake_quantize_fwd", 0) + 1
    return (x * scale).to(torch.float16)


@fake_quantize_fwd.register_fake
def _(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=torch.float16)


@torch.library.custom_op("proto::fake_quantize_wgrad", mutates_args=())
def fake_quantize_wgrad(x: torch.Tensor, scale: float) -> torch.Tensor:
    _calls["fake_quantize_wgrad"] = _calls.get("fake_quantize_wgrad", 0) + 1
    return (x * scale).to(torch.float16)


@fake_quantize_wgrad.register_fake
def _(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=torch.float16)


QuantCacheTensor.register_cacheable_op(torch.ops.proto.fake_quantize_fwd.default, key_fn=lambda args, kwargs: args[1])
QuantCacheTensor.register_cacheable_op(
    torch.ops.proto.fake_quantize_wgrad.default, key_fn=lambda args, kwargs: args[1]
)


class QuantLinear(torch.autograd.Function):
    # Stand-in for Float8BlockwiseLinear, invoked once per branch (w1, w3).
    @staticmethod
    def forward(ctx, h, weight):
        ctx.save_for_backward(h, weight)
        h_fwd_q = torch.ops.proto.fake_quantize_fwd(h, 2.0)
        return h_fwd_q.float() @ weight

    @staticmethod
    def backward(ctx, grad_output):
        h, weight = ctx.saved_tensors
        h_wgrad_q = torch.ops.proto.fake_quantize_wgrad(h, 2.0)
        grad_h = grad_output @ weight.T
        grad_weight = h_wgrad_q.float().T @ grad_output
        return grad_h, grad_weight
