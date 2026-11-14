"""NVFP4 all-gather of the routed experts' weights under FSDP2 (the NVFP4 counterpart of ``fp8_expert_weights``).

With ``nvfp4`` expert compute and ``quantized_all_gather``, FSDP quantizes each expert weight's shard once per
all-gather into the GEMMs' NVFP4 forward operand and all-gathers that instead of BF16 (~0.28x the bytes): the
forward, the activation-checkpoint recompute and the backward (which dequantizes it) of every micro-step reuse one
quantization done on 1/shards of the weight. FSDP shards the experts on the expert dimension and every
quantized tensor leads with it (per-expert scales, per-row blocks), so the gathered bytes are exactly the ones the
GEMM would make from the gathered BF16 weight.

The quantizer is passed in (``prime_kernels.nvfp4_moe``'s, which needs Blackwell), so the FSDP plumbing here runs
with any per-expert quantizer.
"""

from collections.abc import Callable

import torch

from prime_rl.trainer.models.layers.fp8_expert_weights import Fp8ExpertMaster

Quantize = Callable[[torch.Tensor], tuple[torch.Tensor, ...]]
"""``[experts, out, in]`` weights to quantized tensors, each ``[experts, ...]``."""


class NVFP4ExpertWeight(torch.Tensor):
    """An all-gathered expert weight ``[experts, out, in]`` as its quantized tensors, standing in for the BF16 weight
    (its shape and dtype). ``transposed`` marks the ``[experts, in, out]`` view the grouped GEMM takes."""

    @staticmethod
    def __new__(cls, quantized: tuple[torch.Tensor, ...], shape: torch.Size, dtype: torch.dtype, transposed=False):
        groups, rows, cols = shape
        if transposed:
            shape, strides = (groups, cols, rows), (rows * cols, 1, cols)
        else:
            strides = (rows * cols, cols, 1)
        device = quantized[0].device
        return torch.Tensor._make_wrapper_subclass(cls, shape, strides=strides, dtype=dtype, device=device)

    def __init__(self, quantized: tuple[torch.Tensor, ...], shape: torch.Size, dtype: torch.dtype, transposed=False):
        self.quantized = quantized
        self._weight_shape = torch.Size(shape)
        self._transposed = transposed

    def _like(self, transposed: bool) -> "NVFP4ExpertWeight":
        return NVFP4ExpertWeight(self.quantized, self._weight_shape, self.dtype, transposed)

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
        raise NotImplementedError(f"{func} is not supported on an NVFP4 all-gathered expert weight")

    def __tensor_flatten__(self):
        names = [f"_q{i}" for i in range(len(self.quantized))]
        for name, tensor in zip(names, self.quantized):
            setattr(self, name, tensor)
        return names, (self._weight_shape, self.dtype, self._transposed)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        shape, dtype, transposed = metadata
        quantized = tuple(inner_tensors[f"_q{i}"] for i in range(len(inner_tensors)))
        return NVFP4ExpertWeight(quantized, shape, dtype, transposed)

    def __repr__(self):
        return f"NVFP4ExpertWeight(shape={tuple(self.shape)}, transposed={self._transposed})"


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """``[experts, ...]`` as contiguous ``[experts, bytes]`` uint8: one dtype for FSDP, concatenated on experts."""
    return tensor.contiguous().view(torch.uint8).reshape(tensor.shape[0], -1)


_ALIGN = 16


def _aligned(size: int) -> int:
    return (size + _ALIGN - 1) // _ALIGN * _ALIGN


class NVFP4ExpertMaster(Fp8ExpertMaster):
    """An expert weight's local master shard that FSDP all-gathers quantized (``quantize``) into an
    ``NVFP4ExpertWeight``. Every other op runs on the wrapped tensor, as for ``Fp8ExpertMaster``."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, quantize: Quantize):
        return Fp8ExpertMaster.__new__(cls, tensor)

    def __init__(self, tensor: torch.Tensor, quantize: Quantize):
        super().__init__(tensor)
        self._quantize = quantize

    def _rewrap(self, tensor: torch.Tensor) -> "NVFP4ExpertMaster":
        return NVFP4ExpertMaster(tensor, self._quantize)

    def __tensor_flatten__(self):
        return ["_tensor"], self._quantize

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return NVFP4ExpertMaster(inner_tensors["_tensor"], metadata)

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        quantized = self._quantize(self._tensor.to(mp_policy.param_dtype))
        layouts = tuple((t.dtype, t.shape[1:]) for t in quantized)
        inputs = tuple(_as_bytes(t) for t in quantized)
        if mesh.size() > 1:
            # One input per tensor: each gathers into its own contiguous buffer, which FSDP refills in place
            return inputs, (layouts, None)
        # FSDP copies a single input instead of all-gathering: pack them into one buffer at aligned offsets
        offsets, size = [], 0
        for tensor in inputs:
            offsets.append(size)
            size = _aligned(size + tensor.numel())
        packed = torch.zeros(size, dtype=torch.uint8, device=self._tensor.device)
        for offset, tensor in zip(offsets, inputs):
            packed[offset : offset + tensor.numel()] = tensor.flatten()
        return (packed,), (layouts, (tuple(offsets), self._tensor.shape[0]))

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        if out is not None:  # FSDP refilled the same buffers that the weight's tensors view
            return
        layouts, packing = metadata
        if packing is None:
            experts = all_gather_outputs[0].shape[0]
            parts = all_gather_outputs
        else:
            (packed,) = all_gather_outputs
            offsets, experts = packing
            parts = []
            for offset, (dtype, shape) in zip(offsets, layouts):
                numel = experts * torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
                parts.append(packed[offset : offset + numel])
        quantized = tuple(
            part.view(dtype).view(experts, *shape) for part, (dtype, shape) in zip(parts, layouts, strict=True)
        )
        weight_shape = torch.Size((experts, *self._tensor.shape[1:]))
        return NVFP4ExpertWeight(quantized, weight_shape, param_dtype), ()


def all_gather_expert_weights_in_nvfp4(experts: torch.nn.Module, shard_size: int, quantize: Quantize) -> None:
    """Make FSDP all-gather ``experts``' 3-D weights quantized by ``quantize``. Call before ``fully_shard``."""
    from prime_rl.trainer.models.layers.fp8_expert_weights import all_gather_expert_weights_in_fp8

    all_gather_expert_weights_in_fp8(
        experts,
        shard_size,
        master=lambda tensor: NVFP4ExpertMaster(tensor, quantize),
        option="model.moe.compute.quantized_all_gather",
    )
