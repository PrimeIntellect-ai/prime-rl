"""Slot allocation for trainer→inference weight transfer.

A :class:`Slot` is a trainer-side destination buffer mirroring the inference
side's shape and dtype for one logical parameter, plus an optional paired
scale buffer for quantized conversions. The slot owns the bound conversion
function from the registry; calling :meth:`Slot.materialize` writes a fused
source tensor (concatenated from the spec's ``sources``) into the buffers.

Allocation takes only the shapes of trainer-side source tensors — no actual
tensor data — so a publisher can size its registered buffers without holding
references to the live state dict.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_spec import ConversionSpec
from prime_rl.trainer.models.conversions import ConversionEntry, resolve
from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div

Shape = Sequence[int]


@contextmanager
def _alloc_context(device: torch.device | str) -> Iterator[None]:
    """Allocate inside the classic-cudaMalloc pool when on CUDA.

    Required for NIXL-registered buffers under
    ``PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`` (the default in
    prime-rl trainer launches): VMM-backed allocations span multiple
    ``cuMemCreate`` handles which ``nvidia_peermem`` cannot pin.
    """
    if torch.device(device).type != "cuda":
        with nullcontext():
            yield
        return
    from prime_rl.transport.classic_cuda_pool import classic_cuda_alloc

    with classic_cuda_alloc():
        yield


@dataclass
class Slot:
    spec: ConversionSpec
    full_name: str  # e.g. "model.layers.0.self_attn.qkv_proj.weight"
    scale_name: str | None  # full destination name for scale, or None
    weight: Tensor
    scale: Tensor | None
    conversion: ConversionEntry

    def materialize(self, srcs: Sequence[Tensor]) -> None:
        """Concatenate ``srcs`` along ``spec.cat_dim`` and write into the slot."""
        src = srcs[0] if len(srcs) == 1 else torch.cat(list(srcs), dim=self.spec.cat_dim)
        self.conversion.fn(src, self.weight, self.scale)


def _dst_shape(spec: ConversionSpec, src_shapes: Sequence[Shape]) -> tuple[int, ...]:
    if len(src_shapes) == 1:
        return tuple(src_shapes[0])
    base = list(src_shapes[0])
    base[spec.cat_dim] = sum(s[spec.cat_dim] for s in src_shapes)
    return tuple(base)


def allocate_slot(
    spec: ConversionSpec,
    *,
    prefix: str,
    src_shapes: Sequence[Shape],
    default_conversion: str,
    base_dtype: torch.dtype,
    device: torch.device | str = "cuda",
) -> Slot:
    """Allocate the destination buffers for one spec.

    For quantized conversions, ``weight`` is :data:`torch.float8_e4m3fn`,
    ``scale`` is :data:`torch.float32` with shape
    ``(*leading, ceil_div(rows, 128), ceil_div(cols, 128))``, and
    ``scale_name`` follows :meth:`ConversionSpec.scale_name`. For
    non-quantized conversions, ``weight`` uses ``base_dtype`` and ``scale``
    / ``scale_name`` are ``None``.
    """
    entry = resolve(spec.conversion.conversion_type, default_conversion)
    dst_shape = _dst_shape(spec, src_shapes)

    with _alloc_context(device):
        if entry.requires_scale:
            rows, cols = dst_shape[-2], dst_shape[-1]
            leading = dst_shape[:-2]
            scale_shape = (*leading, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE))
            weight = torch.empty(dst_shape, dtype=torch.float8_e4m3fn, device=device)
            scale = torch.empty(scale_shape, dtype=torch.float32, device=device)
            scale_name = spec.scale_name(prefix)
        else:
            weight = torch.empty(dst_shape, dtype=base_dtype, device=device)
            scale = None
            scale_name = None

    return Slot(
        spec=spec,
        full_name=spec.full_name(prefix),
        scale_name=scale_name,
        weight=weight,
        scale=scale,
        conversion=entry,
    )


def allocate_slots(
    state_shapes: Mapping[str, Shape],
    *,
    layer_specs_fn: Callable[[int, bool], tuple[ConversionSpec, ...]],
    non_layer_specs: tuple[ConversionSpec, ...],
    is_dense_fn: Callable[[int], bool],
    num_layers: int,
    default_conversion: str,
    base_dtype: torch.dtype,
    device: torch.device | str = "cuda",
) -> list[Slot]:
    """Allocate one slot per spec for every transformer layer plus the non-layer specs.

    ``state_shapes`` maps trainer-side source tensor names to their shapes.
    The slot allocator never holds references to the live state dict.
    """
    slots: list[Slot] = []

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        for spec in layer_specs_fn(i, is_dense_fn(i)):
            src_shapes = [state_shapes[f"{prefix}.{s}"] for s in spec.sources]
            slots.append(
                allocate_slot(
                    spec,
                    prefix=prefix,
                    src_shapes=src_shapes,
                    default_conversion=default_conversion,
                    base_dtype=base_dtype,
                    device=device,
                )
            )

    for spec in non_layer_specs:
        src_shapes = [state_shapes[s] for s in spec.sources]
        slots.append(
            allocate_slot(
                spec,
                prefix="",
                src_shapes=src_shapes,
                default_conversion=default_conversion,
                base_dtype=base_dtype,
                device=device,
            )
        )

    return slots
