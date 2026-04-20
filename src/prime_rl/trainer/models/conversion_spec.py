"""Conversion specs describe how a trainer-side source tensor is transformed
into a vLLM-kernel-side destination tensor.

Two dataclasses:

* :class:`QuantizationSpec` — the *transformation* for one destination slot.
  Covers both the plain-precision case (bf16 / fp32 dtype cast via
  ``out.copy_(src)``) and the FP8 block-quantized case (2D or 3D dispatched
  from ``src.ndim``).
* :class:`ConversionSpec` — the *routing* for one logical parameter: which
  source tensors fuse into which vLLM destination, along which axis, using
  which :class:`QuantizationSpec`.

This module is model-agnostic. Per-model spec tables live next to the
model's converter and reuse the primitives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import fp8_block_quantize, grouped_fp8_block_quantize


@dataclass(frozen=True)
class QuantizationSpec:
    """How a source tensor is written into a destination slot.

    For non-quantized specs (``scale_suffix=""``) this is a plain
    ``out.copy_(src)`` — PyTorch auto-casts from ``src.dtype`` to
    ``destination_dtype``. For FP8 specs (non-empty ``scale_suffix``),
    :meth:`apply` runs ``fp8_block_quantize`` (2D input) or
    ``grouped_fp8_block_quantize`` (3D stacked-expert input), writing into
    both the slot and the paired scale buffer.

    Attributes:
        destination_dtype: Dtype of the allocated destination slot.
        scale_suffix: Suffix that replaces ``.weight`` or ``_weight`` in
            a :class:`ConversionSpec`'s ``dst`` to form the paired scale
            buffer's name. Empty means no scale buffer → plain copy-cast.
    """

    destination_dtype: torch.dtype
    scale_suffix: str = ""

    @property
    def requires_scale(self) -> bool:
        """True iff this spec produces a paired FP8 scale buffer."""
        return bool(self.scale_suffix)

    def apply(self, src: Tensor, out: Tensor, scale_out: Optional[Tensor]) -> None:
        """Write ``src`` into ``out`` (and ``scale_out`` for FP8 specs)."""
        if self.requires_scale:
            assert scale_out is not None, "quantized spec requires a scale_out buffer"
            if src.ndim == 3:
                grouped_fp8_block_quantize(src, out=out, sf=scale_out)
            else:
                fp8_block_quantize(src, out=out, sf=scale_out)
        else:
            assert scale_out is None, "non-quantized spec was given a scale_out buffer"
            out.copy_(src)


@dataclass(frozen=True)
class ConversionSpec:
    """How one trainer-side logical parameter converts to its vLLM destination.

    Attributes:
        dst: Destination suffix after ``model.layers.{i}.``. E.g.
            ``"self_attn.fused_qkv_a_proj.weight"``.
        sources: One or more source suffixes (after ``model.layers.{i}.``)
            that fuse into ``dst``. Fused along ``cat_dim``.
        cat_dim: Axis along which multiple ``sources`` are concatenated.
        quantization: Transformation to apply from source to slot. Defaults
            to a plain bf16 copy; override for fp32 layernorm params or
            FP8-quantized projections.
    """

    dst: str
    sources: tuple[str, ...]
    cat_dim: int = 0
    quantization: QuantizationSpec = field(default_factory=lambda: QuantizationSpec(torch.bfloat16))

    @property
    def quantized(self) -> bool:
        """True iff this spec produces a paired scale buffer."""
        return self.quantization.requires_scale

    @property
    def slot_dtype(self) -> torch.dtype:
        """Dtype of the main destination slot."""
        return self.quantization.destination_dtype

    def scale_name(self, prefix: str) -> str:
        """Full scale buffer name for this spec's destination at ``prefix``.

        E.g. with ``dst="self_attn.o_proj.weight"``, ``scale_suffix=".weight_scale_inv"``,
        and ``prefix="model.layers.0"`` → ``"model.layers.0.self_attn.o_proj.weight_scale_inv"``.
        """
        assert self.quantized, f"spec {self.dst!r} is not quantized"
        suffix = self.quantization.scale_suffix
        strip = ".weight" if suffix.startswith(".") else "_weight"
        return f"{prefix}.{self.dst}".removesuffix(strip) + suffix

    def per_source_scale_key(self, slot_key: str) -> str:
        """Scale buffer name for a per-source slot of a quantized spec.

        Non-expert quantized specs fuse N sources into one vLLM destination
        but keep a trainer-side scale buffer per source to match the NIXL
        write layout. Same suffix substitution as :meth:`scale_name`,
        keyed by source slot rather than destination.
        """
        assert self.quantized, f"spec {self.dst!r} is not quantized"
        suffix = self.quantization.scale_suffix
        strip = ".weight" if suffix.startswith(".") else "_weight"
        return slot_key.removesuffix(strip) + suffix
