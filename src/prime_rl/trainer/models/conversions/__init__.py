"""Registry of named conversion kernels for trainer→inference weight transfer.

A conversion is a function that writes one source tensor into one destination
tensor, optionally producing a paired scale buffer. Each conversion is
registered under a string name (e.g. ``"fp8_128x128"``).

Resolution flow at startup:

1. The trainer reads the inference model's HF ``config.json`` and calls
   :func:`select_default_conversion` to pick one conversion name to use as
   the default for every spec that doesn't pin its own. The choice is
   driven entirely by ``config.quantization_config`` (or its absence).
2. For each :class:`~prime_rl.trainer.models.conversion_spec.ConversionSpec`,
   :func:`resolve` returns the registry entry — explicit ``conversion_type``
   on the spec wins, otherwise the startup-chosen default applies.

The registry never inspects the destination buffer's dtype; dtype is the
slot allocator's concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from transformers import AutoConfig

ConversionFn = Callable[[Tensor, Tensor, "Tensor | None"], None]


@dataclass(frozen=True)
class ConversionEntry:
    fn: ConversionFn
    requires_scale: bool
    dst_dtype: torch.dtype | None = None
    preserve_source_dtype: bool = False


_REGISTRY: dict[str, ConversionEntry] = {}


def register(
    name: str,
    fn: ConversionFn,
    *,
    requires_scale: bool,
    dst_dtype: torch.dtype | None = None,
    preserve_source_dtype: bool = False,
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"conversion {name!r} is already registered")
    _REGISTRY[name] = ConversionEntry(
        fn=fn,
        requires_scale=requires_scale,
        dst_dtype=dst_dtype,
        preserve_source_dtype=preserve_source_dtype,
    )


def get(name: str) -> ConversionEntry:
    if name not in _REGISTRY:
        raise KeyError(f"unknown conversion {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def select_default_conversion(inference_model_name: str) -> str:
    """Pick the default conversion name for the given inference model.

    Loads the HF config and inspects ``quantization_config``:

    * absent → ``"passthrough"`` (no quantization; trainer→inference is a
      plain dtype cast).
    * ``quant_method == "fp8"`` with ``weight_block_size == [128, 128]`` →
      ``"fp8_128x128"``.
    * anything else → :class:`NotImplementedError`.
    """
    config = AutoConfig.from_pretrained(inference_model_name)
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        return "passthrough"
    if hasattr(quant, "to_dict"):
        quant = quant.to_dict()
    method = quant["quant_method"]
    block_size = tuple(quant.get("weight_block_size") or ())
    if method == "fp8" and block_size == (128, 128):
        return "fp8_128x128"
    raise NotImplementedError(
        f"unsupported inference quantization: quant_method={method!r}, weight_block_size={block_size}"
    )


def resolve(conversion_type: str | None, default: str) -> ConversionEntry:
    """Return the registry entry for a spec. Explicit name wins; otherwise ``default``."""
    return get(conversion_type or default)


from prime_rl.trainer.models.conversions import bf16_cast as _bf16_cast  # noqa: E402, F401
from prime_rl.trainer.models.conversions import fp8_blockwise as _fp8_blockwise  # noqa: E402, F401

__all__ = [
    "ConversionEntry",
    "ConversionFn",
    "register",
    "get",
    "resolve",
    "select_default_conversion",
]
