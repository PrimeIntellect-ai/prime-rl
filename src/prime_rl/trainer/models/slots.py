"""Slot allocation for trainer→inference weight transfer.

A :class:`Slot` is a trainer-side destination buffer mirroring the inference
side's shape and dtype for one logical parameter, plus an optional paired
scale buffer for quantized conversions. The slot owns the bound conversion
function from the registry; calling :meth:`Slot.materialize` writes a fused
source tensor (concatenated from the spec's ``sources``) into the buffers.

This module is model-agnostic. The caller supplies a layer-iterator
(``layer_specs_fn``, ``is_dense_fn``, ``num_layers``) so that per-model
tables (e.g. :func:`prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe.conversion_specs`)
plug in cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_spec import ConversionSpec
from prime_rl.trainer.models.conversions import ConversionEntry, resolve
from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div


@dataclass
class Slot:
    spec: ConversionSpec
    full_name: str  # e.g. "model.layers.0.self_attn.qkv_proj.weight"
    weight: Tensor
    scale: Tensor | None
    conversion: ConversionEntry

    def materialize(self, srcs: Sequence[Tensor]) -> None:
        """Concatenate ``srcs`` along ``spec.cat_dim`` and write into the slot."""
        src = srcs[0] if len(srcs) == 1 else torch.cat(list(srcs), dim=self.spec.cat_dim)
        self.conversion.fn(src, self.weight, self.scale)


def _dst_shape(spec: ConversionSpec, src_shapes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    if len(src_shapes) == 1:
        return tuple(src_shapes[0])
    base = list(src_shapes[0])
    base[spec.cat_dim] = sum(s[spec.cat_dim] for s in src_shapes)
    return tuple(base)


def allocate_slot(
    spec: ConversionSpec,
    *,
    full_name: str,
    src_shapes: Sequence[tuple[int, ...]],
    default_conversion: str,
    base_dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> Slot:
    """Allocate the destination buffers for one spec.

    For quantized conversions, ``weight`` is :data:`torch.float8_e4m3fn`
    and a paired ``scale`` buffer is allocated with shape
    ``(*leading, ceil_div(rows, 128), ceil_div(cols, 128))`` in
    :data:`torch.float32`. For non-quantized conversions, ``weight`` uses
    ``base_dtype`` (typically the inference model's ``torch_dtype``) and
    ``scale`` is ``None``.
    """
    entry = resolve(spec.conversion.conversion_type, default_conversion)
    dst_shape = _dst_shape(spec, src_shapes)

    if entry.requires_scale:
        rows, cols = dst_shape[-2], dst_shape[-1]
        leading = dst_shape[:-2]
        scale_shape = (*leading, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE))
        weight = torch.empty(dst_shape, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty(scale_shape, dtype=torch.float32, device=device)
    else:
        weight = torch.empty(dst_shape, dtype=base_dtype, device=device)
        scale = None

    return Slot(spec=spec, full_name=full_name, weight=weight, scale=scale, conversion=entry)


def allocate_slots(
    src_state_dict: dict[str, Tensor],
    *,
    layer_specs_fn: Callable[[int, bool], tuple[ConversionSpec, ...]],
    non_layer_specs: tuple[ConversionSpec, ...],
    is_dense_fn: Callable[[int], bool],
    num_layers: int,
    default_conversion: str,
    base_dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> list[Slot]:
    """Allocate one slot per spec for every transformer layer plus the non-layer specs.

    ``default_conversion`` and ``base_dtype`` come from the inference target
    (use :func:`select_default_conversion` and ``AutoConfig.torch_dtype`` on
    the caller side). Source shapes are read from ``src_state_dict``.
    """
    slots: list[Slot] = []

    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        for spec in layer_specs_fn(i, is_dense_fn(i)):
            src_shapes = [src_state_dict[f"{prefix}{s}"].shape for s in spec.sources]
            slots.append(
                allocate_slot(
                    spec,
                    full_name=f"{prefix}{spec.dst}",
                    src_shapes=src_shapes,
                    default_conversion=default_conversion,
                    base_dtype=base_dtype,
                    device=device,
                )
            )

    for spec in non_layer_specs:
        src_shapes = [src_state_dict[s].shape for s in spec.sources]
        slots.append(
            allocate_slot(
                spec,
                full_name=spec.dst,
                src_shapes=src_shapes,
                default_conversion=default_conversion,
                base_dtype=base_dtype,
                device=device,
            )
        )

    return slots
