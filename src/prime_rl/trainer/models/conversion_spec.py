"""Conversion specs describe how a trainer-side source tensor is transformed
into a vLLM-kernel-side destination tensor.

* :class:`MaybeQuantize` — the *transformation* selector for one destination
  slot. Carries an opaque ``conversion_type`` string (or ``None`` to let
  :func:`prime_rl.trainer.models.conversions.resolve` pick the default
  based on the destination dtype). All conversion-specific data — block
  size, scale layout, kernel dispatch — lives in the conversion registry,
  not on the spec.
* :class:`ConversionSpec` — the *routing* for one logical parameter:
  which source tensors fuse into which vLLM destination, along which axis,
  using which :class:`MaybeQuantize`.

This module is model-agnostic. Per-model spec tables live next to the
model's converter and reuse the primitives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MaybeQuantize:
    """Selects a conversion. ``None`` lets the registry pick the default
    based on the destination dtype.
    """

    conversion_type: str | None = None


@dataclass(frozen=True)
class ConversionSpec:
    """How one trainer-side logical parameter converts to its vLLM destination.

    Attributes:
        dst: Destination suffix after ``model.layers.{i}.``. E.g.
            ``"self_attn.qkv_proj.weight"``.
        sources: One or more source suffixes (after ``model.layers.{i}.``)
            that fuse into ``dst``. Fused along ``cat_dim``.
        cat_dim: Axis along which multiple ``sources`` are concatenated.
        conversion: Conversion selector. Default leaves the choice to the
            registry; override to pin e.g. ``MaybeQuantize("passthrough")``
            for tensors that must never be quantized regardless of the
            inference variant.
    """

    dst: str
    sources: tuple[str, ...]
    cat_dim: int = 0
    conversion: MaybeQuantize = field(default_factory=MaybeQuantize)

    @property
    def is_expert_spec(self) -> bool:
        """True iff this spec produces a fused stacked-expert slot."""
        return self.dst.startswith("mlp.experts.")

    def full_name(self, prefix: str = "") -> str:
        """Full destination name for the weight buffer at ``prefix``."""
        return f"{prefix}.{self.dst}" if prefix else self.dst

    def scale_name(self, prefix: str = "") -> str:
        """Full destination name for the paired scale buffer at ``prefix``.

        Mirrors vLLM's FP8 naming: ``.weight`` → ``.weight_scale_inv`` for
        2D linears, ``_weight`` → ``_weight_scale_inv`` for 3D stacked-expert
        buffers. Only meaningful when the resolved conversion entry has
        ``requires_scale=True``.
        """
        full = self.full_name(prefix)
        if full.endswith(".weight"):
            return full.removesuffix(".weight") + ".weight_scale_inv"
        if full.endswith("_weight"):
            return full.removesuffix("_weight") + "_weight_scale_inv"
        raise ValueError(f"cannot derive scale name from {full!r}")
