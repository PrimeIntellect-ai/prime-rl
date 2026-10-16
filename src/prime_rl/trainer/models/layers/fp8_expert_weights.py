"""FP8 all-gather of the routed experts' weights under FSDP2.

With ``deepgemm_fp8`` expert compute, FSDP casts each expert weight's shard to 128 x 128-block FP8 once per
all-gather and all-gathers the FP8 blocks with their scales. The grouped GEMMs use them directly, so the forward, the
AC recompute and every micro-step no longer re-quantize the weight, and the all-gather moves half the bytes. The cast
is the one the GEMM did on the bf16 weight, and the input-gradient GEMM's blocks are its transpose, so the numerics
are unchanged.
"""

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    grouped_per_block_cast_to_fp8_triton,
    ue8m0_for_device,
)
from prime_rl.trainer.models.layers.fp8_grouped_gemm import _backward, _gemm, _grouped_layout


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_prequantized", mutates_args=())
def _prequantized_gemm(
    x: torch.Tensor, weight_data: torch.Tensor, weight_scale: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    grouped_layout, block_to_group, _, starts_tensor, actual_ms_tensor, block_starts_tensor = _grouped_layout(x, offs)
    return _gemm(
        x,
        (weight_data, weight_scale),
        weight_data.size(1),
        grouped_layout,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    )


@_prequantized_gemm.register_fake
def _prequantized_gemm_fake(
    x: torch.Tensor, weight_data: torch.Tensor, weight_scale: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight_data.shape[1]))


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_prequantized_backward", mutates_args=())
def _prequantized_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight_data: torch.Tensor,
    weight_scale: torch.Tensor,
    offs: torch.Tensor,
    needs_grad_x: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rows, cols = weight_data.shape
    return _backward(
        grad_output,
        x,
        lambda: (weight_data.transpose(1, 2).contiguous(), weight_scale.transpose(1, 2).contiguous()),
        torch.Size((groups, cols, rows)),
        torch.bfloat16,
        offs,
        needs_grad_x,
        needs_grad_weight,
        True,
    )


@_prequantized_backward.register_fake
def _prequantized_backward_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight_data: torch.Tensor,
    weight_scale: torch.Tensor,
    offs: torch.Tensor,
    needs_grad_x: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rows, cols = weight_data.shape
    return x.new_empty(x.shape), x.new_empty((groups, rows, cols), dtype=torch.bfloat16).transpose(1, 2)


class Fp8ExpertWeight(torch.Tensor):
    """An all-gathered expert weight ``[experts, out, in]`` as FP8 128 x 128 blocks and their scales, standing in for
    the bf16 weight (its shape and dtype). ``transposed`` marks the ``[experts, in, out]`` view the grouped GEMM takes."""

    @staticmethod
    def __new__(cls, data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, transposed: bool = False):
        groups, rows, cols = data.shape
        shape, strides = ((groups, cols, rows), (rows * cols, 1, cols)) if transposed else (data.shape, data.stride())
        return torch.Tensor._make_wrapper_subclass(cls, shape, strides=strides, dtype=dtype, device=data.device)

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, transposed: bool = False):
        self._data = data
        self._scale = scale
        self._transposed = transposed

    def _like(self, transposed: bool) -> "Fp8ExpertWeight":
        return Fp8ExpertWeight(self._data, self._scale, self.dtype, transposed)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        aten = torch.ops.aten
        if func in (aten.detach.default, aten.alias.default):
            return args[0]._like(args[0]._transposed)
        if func in (aten.view.default, aten.as_strided.default, aten._unsafe_view.default) and tuple(args[1]) == tuple(
            args[0].shape
        ):
            return args[0]._like(args[0]._transposed)
        if func == aten.transpose.int and {d % 3 for d in args[1:3]} == {1, 2}:
            return args[0]._like(not args[0]._transposed)
        if func in (aten.zeros_like.default, aten.empty_like.default):
            return func(torch.empty(args[0].shape, dtype=args[0].dtype, device=args[0].device), **kwargs)
        raise NotImplementedError(f"{func} is not supported on an FP8 all-gathered expert weight")

    def __tensor_flatten__(self):
        return ["_data", "_scale"], (self.dtype, self._transposed)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        dtype, transposed = metadata
        return Fp8ExpertWeight(inner_tensors["_data"], inner_tensors["_scale"], dtype, transposed)

    def __repr__(self):
        return f"Fp8ExpertWeight(shape={tuple(self.shape)}, transposed={self._transposed})"


class _PrequantizedGroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: Fp8ExpertWeight, offs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, offs)
        ctx.weight = weight
        return torch.ops.prime_rl.grouped_fp8_gemm_prequantized(x, weight._data, weight._scale, offs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, offs = ctx.saved_tensors
        needs_grad_x, needs_grad_weight, _ = ctx.needs_input_grad
        grad_x, grad_weight = torch.ops.prime_rl.grouped_fp8_gemm_prequantized_backward(
            grad_output, x, ctx.weight._data, ctx.weight._scale, offs, needs_grad_x, needs_grad_weight
        )
        return grad_x if needs_grad_x else None, grad_weight if needs_grad_weight else None, None


def prequantized_grouped_fp8_gemm(x: torch.Tensor, weight: Fp8ExpertWeight, offs: torch.Tensor) -> torch.Tensor:
    """``grouped_fp8_gemm`` on an all-gathered FP8 weight, given as its ``[experts, in, out]`` view."""
    assert weight._transposed
    return _PrequantizedGroupedGemm.apply(x, weight, offs)


# FSDP pads and copies the local shard when it shards the parameter; these keep the master subclass.
_OPS_KEEPING_MASTER = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


class Fp8ExpertMaster(torch.Tensor):
    """An expert weight's local master shard. Every op runs on the wrapped tensor; FSDP all-gathers it as FP8 blocks
    (``fsdp_pre_all_gather``) into an ``Fp8ExpertWeight`` (``fsdp_post_all_gather``)."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return Fp8ExpertMaster(args[0]._tensor)
        args, kwargs = pytree.tree_map_only(Fp8ExpertMaster, lambda t: t._tensor, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _OPS_KEEPING_MASTER:
            return out
        return pytree.tree_map_only(torch.Tensor, Fp8ExpertMaster, out)

    def __tensor_flatten__(self):
        return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return Fp8ExpertMaster(inner_tensors["_tensor"])

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        data, scale = grouped_per_block_cast_to_fp8_triton(
            self._tensor.to(mp_policy.param_dtype), ue8m0_for_device(self._tensor.device), GROUP_ALIGNMENT
        )
        if mesh.size() == 1:  # FSDP copies a single input instead of all-gathering: pack both into one byte buffer
            packed = torch.cat((data.view(torch.uint8).flatten(), scale.view(torch.uint8).flatten()))
            return (packed,), (data.shape, scale.shape)
        return (data, scale), None

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        if out is not None:  # FSDP refilled the same buffers that the weight views
            return
        if len(all_gather_outputs) == 1:
            (packed,) = all_gather_outputs
            data_shape, scale_shape = metadata
            data = packed[: data_shape.numel()].view(torch.float8_e4m3fn).view(data_shape)
            scale = packed[data_shape.numel() :].view(torch.float32).view(scale_shape)
        else:
            data, scale = all_gather_outputs
        return Fp8ExpertWeight(data, scale, param_dtype), ()

    def __repr__(self):
        return f"Fp8ExpertMaster({self._tensor})"


# Checkpoints hold the master shards as this subclass.
torch.serialization.add_safe_globals([Fp8ExpertMaster])


def all_gather_expert_weights_in_fp8(experts: nn.Module) -> None:
    """Make FSDP all-gather ``experts``' 3-D weights as FP8 blocks. Call before ``fully_shard``."""
    for name, param in list(experts.named_parameters(recurse=False)):
        if param.ndim != 3:
            continue
        if isinstance(param, DTensor):
            local = Fp8ExpertMaster(param.to_local())
            wrapped = DTensor.from_local(
                local, param.device_mesh, param.placements, run_check=False, shape=param.shape, stride=param.stride()
            )
        else:
            wrapped = Fp8ExpertMaster(param.data)
        experts.register_parameter(name, nn.Parameter(wrapped, requires_grad=param.requires_grad))
