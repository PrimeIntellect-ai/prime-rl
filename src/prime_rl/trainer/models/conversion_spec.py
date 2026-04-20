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
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import BLOCK_SIZE, fp8_block_quantize, grouped_fp8_block_quantize

if TYPE_CHECKING:
    from prime_rl.trainer.models.slots import Slot
    from prime_rl.trainer.parallel_dims import ParallelDims


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

    @property
    def is_expert_spec(self) -> bool:
        """True iff this spec produces a fused stacked-expert slot."""
        return self.dst.startswith("mlp.experts.")

    def get_handler_class(
        self, src: Optional[Tensor] = None, parallel_dims: Optional["ParallelDims"] = None
    ) -> type["Slot"]:
        """Which :class:`~prime_rl.trainer.models.slots.Slot` subclass handles this spec.

        Expert specs route to :class:`ExpertSlot` regardless of runtime
        context, so ``src`` and ``parallel_dims`` may be omitted.
        Non-expert specs require both to pick :class:`ShardedSlot` vs
        :class:`GatheredSlot` (shape divisibility, FP8 block alignment,
        :data:`~prime_rl.trainer.models.slots.SMALL_NON_EXPERT_BYTES`).
        """
        from prime_rl.trainer.models.slots import (
            SMALL_NON_EXPERT_BYTES,
            ExpertSlot,
            GatheredSlot,
            ShardedSlot,
        )

        if self.is_expert_spec:
            return ExpertSlot
        assert src is not None and parallel_dims is not None, (
            f"non-expert spec {self.dst!r} needs src + parallel_dims for dispatch"
        )
        fsdp_total = parallel_dims.dp_shard * parallel_dims.cp
        src_rows = src.shape[0]
        per_shard = (
            src_rows % fsdp_total == 0
            and (not self.quantized or (src_rows // fsdp_total) % BLOCK_SIZE == 0)
            and src.numel() * src.element_size() >= SMALL_NON_EXPERT_BYTES
        )
        return ShardedSlot if per_shard else GatheredSlot

    def build_slots(
        self, prefix: str, state_dict: dict[str, Tensor], parallel_dims: "ParallelDims"
    ) -> list["Slot"]:
        """Instantiate every slot this spec produces at ``prefix``.

        Dispatches through :meth:`get_handler_class` for both expert and
        non-expert cases. Expert specs yield one :class:`ExpertSlot`;
        non-expert specs yield one :class:`ShardedSlot` or
        :class:`GatheredSlot` per source (fused specs may have one source
        shardable and another not).
        """
        if self.is_expert_spec:
            cls = self.get_handler_class()
            return [cls.from_spec(self, prefix, state_dict, parallel_dims)]
        slots: list["Slot"] = []
        row_off = 0
        scale_row_off = 0
        for src_name in self.sources:
            src = state_dict[f"{prefix}.{src_name}"]
            cls = self.get_handler_class(src, parallel_dims)
            slots.append(
                cls.from_spec(
                    spec=self,
                    prefix=prefix,
                    src_name=src_name,
                    src=src,
                    parallel_dims=parallel_dims,
                    offset_rows=row_off,
                    scale_offset_rows=scale_row_off,
                )
            )
            row_off += src.shape[0]
            if self.quantized:
                scale_row_off += (src.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        return slots
