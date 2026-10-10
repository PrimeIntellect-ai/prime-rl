from __future__ import annotations

import re
from collections.abc import Callable

import torch
from torch import nn

from prime_rl.trainer.models.kernels.clamped_swiglu_fp8 import clamped_swiglu_fp8, clamped_swiglu_fp8_backward
from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    per_token_cast_to_fp8_triton,
    stacked_per_block_cast_to_fp8_triton,
    ue8m0_for_device,
)
from prime_rl.utils.logger import get_logger

# Module key -> callable returning the fp32 buffer its weight gradient accumulates into
# (`Float8BlockwiseLinear.accumulate_wgrad_fp32`).
_wgrad_accumulators: dict[int, Callable[[], torch.Tensor]] = {}


@torch.library.custom_op("prime_rl::fp8_blockwise_mm", mutates_args=())
def _fp8_blockwise_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    wgrad_key: int = 0,
    x_fp8: torch.Tensor | None = None,
    x_sf: torch.Tensor | None = None,
) -> torch.Tensor:
    """`x @ weight.T` in FP8. `x_fp8` / `x_sf`, when given, are `x`'s per-token cast as its producer stored it (scales
    `[K / 128, >= tokens]`, see `clamped_swiglu_fp8`); the backward still reads the bf16 `x`."""
    import deep_gemm

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    use_ue8m0 = ue8m0_for_device(x.device)
    if x_fp8 is None:
        x_fp8 = per_token_cast_to_fp8_triton(x_2d, use_ue8m0, block_size)
    else:
        x_fp8 = (x_fp8.reshape(x_2d.shape), x_sf[:, : x_2d.size(0)].T)
    weight_fp8 = per_block_cast_to_fp8_triton(weight, use_ue8m0, block_size)

    out = torch.empty((x_2d.size(0), weight.size(0)), device=x.device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(x_fp8, weight_fp8, out)
    return out.reshape(*x.shape[:-1], out.size(-1))


@_fp8_blockwise_mm.register_fake
def _fp8_blockwise_mm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    wgrad_key: int = 0,
    x_fp8: torch.Tensor | None = None,
    x_sf: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::fp8_blockwise_mm_backward", mutates_args=())
def _fp8_blockwise_mm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    needs_grad_x: bool,
    needs_grad_weight: bool,
    wgrad_key: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """With an fp32 accumulator registered for `wgrad_key`, the weight gradient is added straight into it and the
    returned weight gradient is empty."""
    import deep_gemm

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    use_ue8m0 = ue8m0_for_device(grad_output.device)
    grad_x = x.new_empty(x.shape)
    grad_weight = weight.new_empty(weight.shape)

    if needs_grad_x:
        grad_output_fp8 = per_token_cast_to_fp8_triton(grad_output_2d, use_ue8m0, block_size)
        weight_dx_fp8 = per_block_cast_to_fp8_tp_triton(weight, use_ue8m0, block_size)
        grad_x_2d = torch.empty_like(x_2d)
        deep_gemm.fp8_gemm_nt(grad_output_fp8, weight_dx_fp8, grad_x_2d)
        grad_x = grad_x_2d.reshape(x.shape)

    if needs_grad_weight:
        # The transposed casts zero-pad the token dimension, as DeepGEMM's (1, 1, 128) recipe requires.
        grad_output_t_fp8 = per_token_cast_to_fp8_tp_triton(grad_output_2d, use_ue8m0, block_size)
        x_t_fp8 = per_token_cast_to_fp8_tp_triton(x_2d, use_ue8m0, block_size)
        accumulator = _wgrad_accumulators.get(wgrad_key)
        grad_weight_fp32 = (
            accumulator()
            if accumulator is not None
            else torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
        )
        deep_gemm.fp8_gemm_nt(
            grad_output_t_fp8,
            x_t_fp8,
            grad_weight_fp32,
            c=grad_weight_fp32,
            recipe=(1, 1, 128),
        )
        grad_weight = weight.new_empty(0) if accumulator is not None else grad_weight_fp32.to(weight.dtype)

    return grad_x, grad_weight


@_fp8_blockwise_mm_backward.register_fake
def _fp8_blockwise_mm_backward_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    needs_grad_x: bool,
    needs_grad_weight: bool,
    wgrad_key: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    accumulates = needs_grad_weight and wgrad_key in _wgrad_accumulators
    return x.new_empty(x.shape), weight.new_empty(0) if accumulates else weight.new_empty(weight.shape)


def _fp8_blockwise_mm_setup_context(ctx, inputs, output) -> None:
    x, weight, block_size, wgrad_key, *_ = inputs
    ctx.save_for_backward(x, weight)
    ctx.block_size = block_size
    ctx.wgrad_key = wgrad_key


def _fp8_blockwise_mm_autograd_backward(ctx, grad_output: torch.Tensor):
    x, weight = ctx.saved_tensors
    needs_grad_x, needs_grad_weight, *_ = ctx.needs_input_grad
    grad_x, grad_weight = _fp8_blockwise_mm_backward(
        grad_output,
        x.detach(),
        weight.detach(),
        ctx.block_size,
        needs_grad_x,
        needs_grad_weight,
        ctx.wgrad_key,
    )
    # A weight gradient accumulated in fp32 is not returned.
    keep_grad_weight = needs_grad_weight and ctx.wgrad_key not in _wgrad_accumulators
    return grad_x if needs_grad_x else None, grad_weight if keep_grad_weight else None, None, None, None, None


_fp8_blockwise_mm.register_autograd(
    _fp8_blockwise_mm_autograd_backward,
    setup_context=_fp8_blockwise_mm_setup_context,
)


@torch.library.custom_op("prime_rl::fp8_gate_up_mm", mutates_args=())
def _fp8_gate_up_mm(
    x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    """`[x @ gate_weight.T | x @ up_weight.T]` (2-D `x`) as one FP8 GEMM: `x` is cast once, the weights blockwise in
    place, so each half equals `_fp8_blockwise_mm` of its weight."""
    import deep_gemm

    use_ue8m0 = ue8m0_for_device(x.device)
    x_fp8 = per_token_cast_to_fp8_triton(x, use_ue8m0, block_size)
    weight_fp8 = stacked_per_block_cast_to_fp8_triton((gate_weight, up_weight), use_ue8m0, block_size)
    out = torch.empty((x.size(0), gate_weight.size(0) + up_weight.size(0)), device=x.device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(x_fp8, weight_fp8, out)
    return out


@_fp8_gate_up_mm.register_fake
def _fp8_gate_up_mm_fake(
    x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    return x.new_empty((x.size(0), gate_weight.size(0) + up_weight.size(0)), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::fp8_clamped_swiglu", mutates_args=())
def _fp8_clamped_swiglu(gate_up: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return clamped_swiglu_fp8(gate_up, limit, ue8m0_for_device(gate_up.device))


@_fp8_clamped_swiglu.register_fake
def _fp8_clamped_swiglu_fake(gate_up: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, I = gate_up.size(0), gate_up.size(1) // 2
    return (
        gate_up.new_empty((rows, I)),
        gate_up.new_empty((rows, I), dtype=torch.float8_e4m3fn),
        gate_up.new_empty((I // 128, (rows + 3) // 4 * 4), dtype=torch.float32),
    )


@torch.library.custom_op("prime_rl::fp8_gate_up_clamped_swiglu_backward", mutates_args=())
def _fp8_gate_up_clamped_swiglu_backward(
    grad_h: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_up: torch.Tensor,
    limit: float,
    block_size: int,
    needs_grad_x: bool,
    needs_grad_weight: bool,
    gate_wgrad_key: int = 0,
    up_wgrad_key: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gate and up projections' backward from `d[gate | up]`'s fused casts: two data-gradient GEMMs summed in
    bf16 (as autograd sums the two projections' gradients of `x`), one transposed cast of `x`, two weight-gradient
    GEMMs. A weight with an fp32 accumulator registered for its key gets its gradient added there (and an empty one
    returned)."""
    import deep_gemm

    use_ue8m0 = ue8m0_for_device(x.device)
    rows = x.size(0)
    q, sf, qt, sft = clamped_swiglu_fp8_backward(grad_h.contiguous(), gate_up, limit, use_ue8m0)
    weights = (gate_weight, up_weight)
    grad_x = x.new_empty(x.shape)
    grad_weights = [w.new_empty(w.shape) for w in weights]

    if needs_grad_x:
        grad_x_halves = []
        for half, weight in enumerate(weights):
            weight_dx_fp8 = per_block_cast_to_fp8_tp_triton(weight, use_ue8m0, block_size)
            grad_x_half = torch.empty_like(x)
            deep_gemm.fp8_gemm_nt((q[half], sf[half][:, :rows].T), weight_dx_fp8, grad_x_half)
            grad_x_halves.append(grad_x_half)
        grad_x = grad_x_halves[0] + grad_x_halves[1]

    if needs_grad_weight:
        x_t_fp8 = per_token_cast_to_fp8_tp_triton(x, use_ue8m0, block_size)
        for half, (weight, key) in enumerate(zip(weights, (gate_wgrad_key, up_wgrad_key))):
            accumulator = _wgrad_accumulators.get(key)
            grad_weight_fp32 = (
                accumulator()
                if accumulator is not None
                else torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
            )
            deep_gemm.fp8_gemm_nt(
                (qt[half], sft[half].T), x_t_fp8, grad_weight_fp32, c=grad_weight_fp32, recipe=(1, 1, 128)
            )
            grad_weights[half] = weight.new_empty(0) if accumulator is not None else grad_weight_fp32.to(weight.dtype)

    return grad_x, grad_weights[0], grad_weights[1]


@_fp8_gate_up_clamped_swiglu_backward.register_fake
def _fp8_gate_up_clamped_swiglu_backward_fake(
    grad_h,
    x,
    gate_weight,
    up_weight,
    gate_up,
    limit,
    block_size,
    needs_grad_x,
    needs_grad_weight,
    gate_wgrad_key=0,
    up_wgrad_key=0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_weights = [
        w.new_empty(0) if needs_grad_weight and key in _wgrad_accumulators else w.new_empty(w.shape)
        for w, key in ((gate_weight, gate_wgrad_key), (up_weight, up_wgrad_key))
    ]
    return x.new_empty(x.shape), *grad_weights


class _FP8GateUpClampedSwiglu(torch.autograd.Function):
    """`h = silu(clamp(x @ gate.T, max=limit)) * clamp(x @ up.T, -limit, limit)` with `h`'s per-token FP8 cast.

    One autograd node, so the backward's fused kernel hands the cast `d[gate | up]` to the projections' GEMMs.
    Activation checkpointing sees the GEMM (`fp8_gate_up_mm`, saved where `fp8_blockwise_mm` is) and the SwiGLU
    (`fp8_clamped_swiglu`, recomputed), as it saw the two projections and the elementwise SwiGLU."""

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, limit, block_size, gate_wgrad_key, up_wgrad_key):
        gate_up = torch.ops.prime_rl.fp8_gate_up_mm(x, gate_weight, up_weight, block_size)
        h, h_fp8, h_sf = torch.ops.prime_rl.fp8_clamped_swiglu(gate_up, limit)
        ctx.save_for_backward(x, gate_weight, up_weight, gate_up)
        ctx.limit, ctx.block_size = limit, block_size
        ctx.wgrad_keys = (gate_wgrad_key, up_wgrad_key)
        ctx.mark_non_differentiable(h_fp8, h_sf)
        return h, h_fp8, h_sf

    @staticmethod
    def backward(ctx, grad_h, _grad_h_fp8, _grad_h_sf):
        x, gate_weight, up_weight, gate_up = ctx.saved_tensors
        needs_grad_x, needs_grad_gate, needs_grad_up = ctx.needs_input_grad[:3]
        grad_x, grad_gate, grad_up = torch.ops.prime_rl.fp8_gate_up_clamped_swiglu_backward(
            grad_h,
            x,
            gate_weight,
            up_weight,
            gate_up,
            ctx.limit,
            ctx.block_size,
            needs_grad_x,
            needs_grad_gate or needs_grad_up,
            *ctx.wgrad_keys,
        )
        # A weight gradient accumulated in fp32 is not returned.
        keep_gate, keep_up = (key not in _wgrad_accumulators for key in ctx.wgrad_keys)
        return (
            grad_x if needs_grad_x else None,
            grad_gate if needs_grad_gate and keep_gate else None,
            grad_up if needs_grad_up and keep_up else None,
            None,
            None,
            None,
            None,
        )


def fp8_clamped_swiglu_mlp(
    x: torch.Tensor,
    gate_proj: Float8BlockwiseLinear,
    up_proj: Float8BlockwiseLinear,
    down_proj: Float8BlockwiseLinear,
    limit: float,
) -> torch.Tensor:
    """`down_proj(silu(clamp(gate_proj(x), max=limit)) * clamp(up_proj(x), -limit, limit))`, bitwise, with one
    gate/up GEMM, one fused SwiGLU + FP8 cast kernel each way and the down projection reading that cast."""
    assert gate_proj.block_size == up_proj.block_size == down_proj.block_size
    x_2d = x.reshape(-1, x.shape[-1])
    h, h_fp8, h_sf = _FP8GateUpClampedSwiglu.apply(
        x_2d, gate_proj.weight, up_proj.weight, limit, gate_proj.block_size, gate_proj.wgrad_key, up_proj.wgrad_key
    )
    out = _fp8_blockwise_mm(h, down_proj.weight, down_proj.block_size, down_proj.wgrad_key, h_fp8, h_sf)
    return out.reshape(*x.shape[:-1], out.size(-1))


class Float8BlockwiseLinear(nn.Linear):
    """nn.Linear replacement that uses FP8 blockwise matmul via DeepGEMM.

    Requires:
    - SM90 (Hopper) or SM100 (Blackwell) GPU
    - bfloat16 inputs/weights
    - No bias
    - in_features and out_features divisible by 128
    """

    def __init__(self, *args, block_size: int = 128, dtype=torch.bfloat16, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.wgrad_key = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fp8_blockwise_mm(x, self.weight, self.block_size, self.wgrad_key)

    def accumulate_wgrad_fp32(self, accumulator: Callable[[], torch.Tensor]) -> None:
        """Add the weight-gradient GEMM straight into `accumulator()` (fp32) instead of returning a bf16 gradient."""
        self.wgrad_key = id(self)
        _wgrad_accumulators[self.wgrad_key] = accumulator

    @classmethod
    def from_linear(cls, mod: nn.Linear) -> "Float8BlockwiseLinear":
        """Convert an existing nn.Linear to Float8BlockwiseLinear."""
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def replace_linear_with_fp8_blockwise_linear(model: nn.Module, ignore_modules: list[str]) -> None:
    """Replace nn.Linear in `model` with Float8BlockwiseLinear, skipping any
    module whose qualified name matches an ignore pattern (substring or regex).

    The default ignore list covers layers that should never be quantized:
    - lm_head
    - MoE routers and gates (router, mlp.gate., shared_expert.output_gate)
    - sparse-MLA scalar projection (weights_proj)
    - GLM-5.1 MTP head (eh_proj)
    - hybrid-Mamba projections (in_proj_a, in_proj_b)

    Independently of the name-based ignore list, we also skip any nn.Linear
    whose in_features or out_features is not a multiple of 128. Float8BlockwiseLinear
    documents that requirement and DeepGEMM's fp8_gemm_nt crashes at runtime
    on unaligned dims — better to keep them in BF16 with a clear log line than
    silently break in the kernel.

    Conv1d, layer norms, and embedding tables are not nn.Linear and are
    skipped automatically by the type check; we don't need to list them.
    """
    logger = get_logger()
    logger.info(f"Replacing linear layers with FP8 blockwise linear layers (ignore={ignore_modules})")
    replaced_modules = []
    skipped_modules = []
    skipped_unaligned: list[str] = []
    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        # Subclasses of nn.Linear with their own forward (e.g. block-diagonal projections) compute
        # something else than `x @ weight.T`, so only plain linears are swapped.
        if type(module) is not nn.Linear:
            continue
        if any(re.search(pattern, name) for pattern in ignore_modules):
            skipped_modules.append(name)
            continue
        if module.in_features % 128 != 0 or module.out_features % 128 != 0:
            skipped_unaligned.append(f"{name}({module.in_features}->{module.out_features})")
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, Float8BlockwiseLinear.from_linear(module))
        replaced_modules.append(name)

    logger.info(
        f"Replaced {len(replaced_modules)} linear layers with FP8 blockwise linear "
        f"(skipped {len(skipped_modules)} by name, "
        f"{len(skipped_unaligned)} by 128-divisibility); "
        f"first replaced={replaced_modules[:3]}, "
        f"first skipped(name)={skipped_modules[:3]}, "
        f"first skipped(unaligned)={skipped_unaligned[:3]}"
    )
