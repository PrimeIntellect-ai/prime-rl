"""Conversion-spec resolution for Qwen3 MoE — bf16 and FP8 variants."""

from __future__ import annotations

import pytest

from prime_rl.trainer.models.conversions import resolve, select_default_conversion
from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import (
    _BASE,
    _DENSE,
    _SPARSE,
    non_layer_conversion_specs,
)


@pytest.fixture(
    params=[
        pytest.param(("Qwen/Qwen3-235B-A22B-Thinking-2507", "passthrough"), id="bf16"),
        pytest.param(("Qwen/Qwen3-235B-A22B-Thinking-2507-FP8", "fp8_128x128"), id="fp8"),
    ]
)
def qwen3_variant(request) -> tuple[str, str]:
    return request.param


def test_select_default_conversion(qwen3_variant):
    model_name, expected = qwen3_variant
    assert select_default_conversion(model_name) == expected


def test_specs_resolve_correctly(qwen3_variant):
    _, default = qwen3_variant
    for spec in _BASE + _DENSE + _SPARSE + non_layer_conversion_specs():
        entry = resolve(spec.conversion.conversion_type, default)
        expected = spec.conversion.conversion_type or default
        assert entry.fn.__name__ == expected, f"{spec.dst} -> {entry.fn.__name__}"
