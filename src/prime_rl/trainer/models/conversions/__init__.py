"""Registry of named conversion kernels for trainer→inference weight transfer.

A conversion is a function that writes one source tensor into one destination
tensor, optionally producing a paired scale buffer. Each conversion is
registered under a string name (e.g. ``"fp8_128x128"``) and carries a
``compile_target`` tag plus a ``compile_metadata`` dict that downstream MX
clients (Phase 3a on ``ai-dynamo/modelexpress:kavink/post-2389-phase3-4``)
use to advertise the bytes' layout to receivers.

Resolution flow at startup:

1. The trainer reads the inference model's HF ``config.json`` and calls
   :func:`select_default_conversion` to pick one conversion name to use as
   the default for every spec that doesn't pin its own. The choice is
   driven entirely by ``config.quantization_config`` (or its absence).
   The resolver is **table-driven** (see ``_DEFAULT_RULES``) so adding a
   new kernel = adding one row, not editing if/else chains.
2. For each :class:`~prime_rl.trainer.models.conversion_spec.ConversionSpec`,
   :func:`resolve` returns the registry entry — explicit ``conversion_type``
   on the spec wins, otherwise the startup-chosen default applies.

The registry never inspects destination buffer dtype; slot allocation is
owned by the transfer slot builder.

When the Phase 2 graduation of ``MxRendezvous`` onto
``MxV2TrainingPublisher`` lands (see
``KavinKrishnan/prime-rl:kavink/post-2389-phase2-rendezvous-fixes``), the
publisher reads each tensor's resolved ``ConversionEntry.compile_target``
and ``compile_metadata`` and tags ``TensorDescriptorV2`` accordingly.
Receivers filter via ``MxV2RefitReceiver.discover_v2_sources(
compile_target_filter=…, required_compile_metadata=…)``. Until graduation
lands, the fields are populated but unused — callers can read them via
``ConversionEntry.compile_target`` to plumb manually if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from torch import Tensor

ConversionFn = Callable[[Tensor, Tensor, "Tensor | None"], None]


# Canonical compile-target strings. Mirror the constants in
# ``modelexpress.shape_descriptors`` (Phase 3a, kavink/post-2389-phase3-4)
# so the two repos use exactly the same vocabulary without a hard import
# dependency in either direction.
COMPILE_TARGET_HF_RAW = "hf_raw"
COMPILE_TARGET_DEEPGEMM_FP8 = "deep_gemm_fp8"
COMPILE_TARGET_CUTLASS_FP8 = "cutlass_fp8"
COMPILE_TARGET_VLLM_FUSED = "vllm_fused"
COMPILE_TARGET_TRTLLM = "trtllm"


@dataclass(frozen=True)
class ConversionEntry:
    """Registry record for one trainer→inference conversion kernel.

    Fields:
        fn: The actual conversion function. Signature
            ``(src, out, scale_out_or_None) -> None``.
        requires_scale: True if ``fn`` writes a scale buffer; the slot
            builder must allocate one.
        compile_target: One of the ``COMPILE_TARGET_*`` strings. Identifies
            the layout family the output bytes belong to. Receivers filter
            on this via the v2 MX client. Default ``"hf_raw"`` means "no
            kernel-specific layout, plain HF state-dict".
        compile_metadata: Free-form key/value blob describing the specific
            compile invocation (e.g. ``{"block_size": 128,
            "scale_layout": "K-major"}``). Receivers should treat a
            mismatch on any byte-affecting field as a hard reject even
            if ``compile_target`` matches.
    """

    fn: ConversionFn
    requires_scale: bool
    compile_target: str = COMPILE_TARGET_HF_RAW
    compile_metadata: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ConversionEntry] = {}


def register(
    name: str,
    fn: ConversionFn,
    *,
    requires_scale: bool,
    compile_target: str = COMPILE_TARGET_HF_RAW,
    compile_metadata: dict[str, Any] | None = None,
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"conversion {name!r} is already registered")
    _REGISTRY[name] = ConversionEntry(
        fn=fn,
        requires_scale=requires_scale,
        compile_target=compile_target,
        compile_metadata=dict(compile_metadata) if compile_metadata else {},
    )


def get(name: str) -> ConversionEntry:
    if name not in _REGISTRY:
        raise KeyError(f"unknown conversion {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_names() -> list[str]:
    """Snapshot the currently-registered conversion names. Used in tests + diagnostics."""
    return sorted(_REGISTRY)


# Table-driven default selection. Each row is a predicate on the parsed
# HF ``quantization_config`` plus the conversion name to return when it
# matches. Walked in order; first match wins. Extending support for a new
# kernel = appending one row (or registering a row from the kernel's
# module on import — see how cutlass_fp8.py does this).
_QuantPredicate = Callable[[dict[str, Any]], bool]
_DEFAULT_RULES: list[tuple[_QuantPredicate, str]] = []


def register_default_rule(
    predicate: _QuantPredicate,
    name: str,
    *,
    insert_first: bool = False,
) -> None:
    """Add a rule to the default-conversion resolver.

    Args:
        predicate: callable taking the dict form of the HF
            ``quantization_config`` (always non-None — the resolver
            short-circuits to ``"bf16_cast"`` when no quantization_config
            is present). Return True to claim this config.
        name: registered conversion name to return on match. Must already
            be in ``_REGISTRY`` (or be registered before
            ``select_default_conversion`` is called).
        insert_first: if True, prepend the rule so it beats earlier-
            registered rules. Use sparingly — preferred is to append and
            let earlier rules with stricter predicates win.
    """
    pair = (predicate, name)
    if insert_first:
        _DEFAULT_RULES.insert(0, pair)
    else:
        _DEFAULT_RULES.append(pair)


def select_default_conversion(inference_model_name: str) -> str:
    """Pick the default conversion name for the given inference model.

    Loads the HF config and inspects ``quantization_config``. When no
    quantization_config is present we short-circuit to ``"bf16_cast"`` so
    test environments without a real HF download can still exercise the
    default path. When present, we walk the ``_DEFAULT_RULES`` table in
    order and return the first matching name. If nothing matches the
    function raises :class:`NotImplementedError` with the full set of
    registered conversions in the message — extend support by adding a
    row to ``_DEFAULT_RULES`` (see :func:`register_default_rule`) from
    the kernel's own module.
    """
    # Deferred import: ``transformers`` is a heavy dep we don't want to
    # pay at registry-load time (the registry is imported by tests and
    # tooling that have no HF download capability). The function is the
    # only place that needs it.
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(inference_model_name)
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        return "bf16_cast"
    if hasattr(quant, "to_dict"):
        quant = quant.to_dict()
    for predicate, name in _DEFAULT_RULES:
        try:
            if predicate(quant):
                return name
        except Exception:
            # A predicate that raises on an unexpected config shape should
            # not crash the resolver — treat it as "doesn't match" and
            # move on. This keeps registry hooks robust to model-name
            # weirdness without forcing every predicate to be defensive.
            continue
    raise NotImplementedError(
        f"unsupported inference quantization: {quant!r}; "
        f"registered conversions: {sorted(_REGISTRY)}; "
        f"register a new rule via prime_rl.trainer.models.conversions.register_default_rule"
    )


def resolve(conversion_type: str | None, default: str) -> ConversionEntry:
    """Return the registry entry for a spec. Explicit name wins; otherwise ``default``."""
    return get(conversion_type or default)


from prime_rl.trainer.models.conversions import bf16_cast as _bf16_cast  # noqa: E402, F401
from prime_rl.trainer.models.conversions import fp8_blockwise as _fp8_blockwise  # noqa: E402, F401
from prime_rl.trainer.models.conversions import cutlass_fp8 as _cutlass_fp8  # noqa: E402, F401

__all__ = [
    "COMPILE_TARGET_CUTLASS_FP8",
    "COMPILE_TARGET_DEEPGEMM_FP8",
    "COMPILE_TARGET_HF_RAW",
    "COMPILE_TARGET_TRTLLM",
    "COMPILE_TARGET_VLLM_FUSED",
    "ConversionEntry",
    "ConversionFn",
    "get",
    "register",
    "register_default_rule",
    "registered_names",
    "resolve",
    "select_default_conversion",
]
