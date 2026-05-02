"""Slot allocation for tiny Qwen3 MoE configs — bf16 and FP8 inference."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3MoeConfig

from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import (
    conversion_specs,
    is_dense_layer,
    non_layer_conversion_specs,
)
from prime_rl.trainer.models.slots import allocate_slots

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="slot allocation lives on CUDA")


def _make_tiny_config(num_layers: int = 2) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        num_hidden_layers=num_layers,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        vocab_size=128,
        max_position_embeddings=128,
    )


def _make_tt_state_dict(config: Qwen3MoeConfig) -> dict[str, torch.Tensor]:
    """Hand-roll a TT-format trainer state dict for a Qwen3MoE config.

    Mirrors the layout produced by ``convert_hf_layer_to_tt`` (3D fused
    expert tensors ``mlp.experts.w{1,2,3}`` and ``mlp.router.gate.weight``).
    """
    h = config.hidden_size
    mh = config.moe_intermediate_size
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = h // n_q
    e = config.num_experts
    v = config.vocab_size

    sd: dict[str, torch.Tensor] = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.empty(h)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.empty(h)
        sd[f"{p}.self_attn.q_norm.weight"] = torch.empty(head_dim)
        sd[f"{p}.self_attn.k_norm.weight"] = torch.empty(head_dim)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.empty(n_q * head_dim, h)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.empty(n_kv * head_dim, h)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.empty(n_kv * head_dim, h)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.empty(h, n_q * head_dim)
        sd[f"{p}.mlp.router.gate.weight"] = torch.empty(e, h)
        sd[f"{p}.mlp.experts.w1"] = torch.empty(e, mh, h)
        sd[f"{p}.mlp.experts.w2"] = torch.empty(e, h, mh)
        sd[f"{p}.mlp.experts.w3"] = torch.empty(e, mh, h)
    sd["model.embed_tokens.weight"] = torch.empty(v, h)
    sd["model.norm.weight"] = torch.empty(h)
    sd["lm_head.weight"] = torch.empty(v, h)
    return sd


@pytest.fixture
def tiny_qwen3_state() -> tuple[Qwen3MoeConfig, dict[str, torch.Tensor]]:
    config = _make_tiny_config()
    return config, _make_tt_state_dict(config)


@pytest.fixture(
    params=[
        pytest.param(("passthrough", torch.bfloat16), id="bf16"),
        pytest.param(("fp8_128x128", torch.bfloat16), id="fp8"),
    ]
)
def inference_target(request) -> tuple[str, torch.dtype]:
    """(default_conversion, base_dtype) — both Qwen3-235B variants store
    non-quantized tensors in bf16; the FP8 variant adds blockwise FP8 weights.
    """
    return request.param


def _allocate(config, state_dict, default_conversion, base_dtype):
    state_shapes = {k: v.shape for k, v in state_dict.items()}
    return allocate_slots(
        state_shapes,
        layer_specs_fn=conversion_specs,
        non_layer_specs=non_layer_conversion_specs(),
        is_dense_fn=lambda i: is_dense_layer(config, i),
        num_layers=config.num_hidden_layers,
        default_conversion=default_conversion,
        base_dtype=base_dtype,
    )


def test_slot_count(tiny_qwen3_state, inference_target):
    config, sd = tiny_qwen3_state
    default, base = inference_target
    slots = _allocate(config, sd, default, base)
    # Per layer: _BASE (6) + _SPARSE (3); plus non-layer (3).
    expected = config.num_hidden_layers * 9 + 3
    assert len(slots) == expected


def test_slot_dtype_and_shape_per_target(tiny_qwen3_state, inference_target):
    config, sd = tiny_qwen3_state
    default, base = inference_target
    slots = _allocate(config, sd, default, base)
    by_name = {s.full_name: s for s in slots}

    # Every spec produced a slot at its full destination name.
    for s in slots:
        assert s.full_name in by_name

    # Pinned (passthrough) tensors keep the inference base dtype, no scale.
    pinned_full = {
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for name in pinned_full:
        s = by_name[name]
        assert s.weight.dtype == base, f"{name}: dtype={s.weight.dtype}"
        assert s.scale is None, f"{name}: should have no scale"

    # Default-resolution tensors take the inference target's storage choice.
    sample = by_name["model.layers.0.self_attn.qkv_proj.weight"]
    if default == "fp8_128x128":
        assert sample.weight.dtype == torch.float8_e4m3fn
        assert sample.scale is not None
        assert sample.scale.dtype == torch.float32
    else:
        assert sample.weight.dtype == base
        assert sample.scale is None


def test_qkv_fusion_shape(tiny_qwen3_state, inference_target):
    config, sd = tiny_qwen3_state
    default, base = inference_target
    slots = _allocate(config, sd, default, base)
    by_name = {s.full_name: s for s in slots}

    n_q, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = config.hidden_size // n_q
    expected_rows = (n_q + 2 * n_kv) * head_dim
    qkv = by_name["model.layers.0.self_attn.qkv_proj.weight"]
    assert qkv.weight.shape == (expected_rows, config.hidden_size)


def test_w13_fusion_shape(tiny_qwen3_state, inference_target):
    config, sd = tiny_qwen3_state
    default, base = inference_target
    slots = _allocate(config, sd, default, base)
    by_name = {s.full_name: s for s in slots}

    e = config.num_experts
    mh = config.moe_intermediate_size
    h = config.hidden_size
    w13 = by_name["model.layers.0.mlp.experts.w13_weight"]
    # Sources w1 (e, mh, h) and w3 (e, mh, h) concat along cat_dim=1 → (e, 2*mh, h).
    assert w13.weight.shape == (e, 2 * mh, h)


def test_fp8_scale_buffer_shape(tiny_qwen3_state):
    """FP8 specs allocate scales sized by 128x128 block tiling on the trailing
    two dims; leading dims (e.g. expert axis) are preserved.
    """
    config, sd = tiny_qwen3_state
    slots = _allocate(config, sd, "fp8_128x128", torch.bfloat16)
    by_name = {s.full_name: s for s in slots}

    qkv = by_name["model.layers.0.self_attn.qkv_proj.weight"]
    rows, cols = qkv.weight.shape
    assert qkv.scale.shape == (ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE))

    w13 = by_name["model.layers.0.mlp.experts.w13_weight"]
    e, rows, cols = w13.weight.shape
    assert w13.scale.shape == (e, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE))


def test_materialize_passthrough(tiny_qwen3_state):
    """End-to-end roundtrip: allocate slots, materialize from sources, check values."""
    config, sd = tiny_qwen3_state
    g = torch.Generator(device="cuda").manual_seed(0)
    for k, v in sd.items():
        sd[k] = torch.randn(v.shape, generator=g, dtype=torch.float32, device="cuda")

    slots = _allocate(config, sd, "passthrough", torch.bfloat16)
    by_name = {s.full_name: s for s in slots}

    qkv = by_name["model.layers.0.self_attn.qkv_proj.weight"]
    sources = [sd[f"model.layers.0.self_attn.{x}_proj.weight"] for x in ("q", "k", "v")]
    qkv.materialize(sources)
    expected = torch.cat(sources, dim=0).to(torch.bfloat16)
    torch.testing.assert_close(qkv.weight, expected)
