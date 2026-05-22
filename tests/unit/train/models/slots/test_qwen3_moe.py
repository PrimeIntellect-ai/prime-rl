"""Slot allocation for tiny Qwen3 MoE configs — bf16 and FP8 inference, single-rank GPU.

Verifies the dispatch (ShardedSlot vs GatheredSlot vs ExpertSlot), per-slot
buffer keys, layout payloads, write entries, and an end-to-end materialize
roundtrip on the qkv-projection slot.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3MoeConfig

from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import (
    BASE_LAYER_CONVERSION_SPEC,
    DENSE_LAYER_CONVERSION_SPEC,
    NON_LAYER_CONVERSION_SPEC,
    SPARSE_LAYER_CONVERSION_SPEC,
)
from prime_rl.trainer.models.slots import (
    SMALL_NON_EXPERT_BYTES,
    ExpertSlot,
    GatheredSlot,
    ShardedSlot,
    build_slots_for_conversion_spec,
)
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transport.wire import PeerInfo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="slot allocation lives on CUDA")


def _tiny_config() -> Qwen3MoeConfig:
    # Hidden = 4096 so qkv (4096 rows for q, 1024 for k, 1024 for v) is large
    # enough (~ 8MiB) to clear SMALL_NON_EXPERT_BYTES and land as ShardedSlot
    # under trivial parallelism (dp_shard=1).
    return Qwen3MoeConfig(
        num_hidden_layers=2,
        hidden_size=4096,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        vocab_size=128,
        max_position_embeddings=128,
    )


def _state_dict(config: Qwen3MoeConfig) -> dict[str, torch.Tensor]:
    h, mh = config.hidden_size, config.moe_intermediate_size
    n_q, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = h // n_q
    e, v = config.num_experts, config.vocab_size
    sd: dict[str, torch.Tensor] = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.empty(h, device="cuda")
        sd[f"{p}.post_attention_layernorm.weight"] = torch.empty(h, device="cuda")
        sd[f"{p}.self_attn.q_norm.weight"] = torch.empty(head_dim, device="cuda")
        sd[f"{p}.self_attn.k_norm.weight"] = torch.empty(head_dim, device="cuda")
        sd[f"{p}.self_attn.q_proj.weight"] = torch.empty(n_q * head_dim, h, device="cuda")
        sd[f"{p}.self_attn.k_proj.weight"] = torch.empty(n_kv * head_dim, h, device="cuda")
        sd[f"{p}.self_attn.v_proj.weight"] = torch.empty(n_kv * head_dim, h, device="cuda")
        sd[f"{p}.self_attn.o_proj.weight"] = torch.empty(h, n_q * head_dim, device="cuda")
        sd[f"{p}.mlp.router.gate.weight"] = torch.empty(e, h, device="cuda")
        sd[f"{p}.mlp.experts.w1"] = torch.empty(e, mh, h, device="cuda")
        sd[f"{p}.mlp.experts.w2"] = torch.empty(e, h, mh, device="cuda")
        sd[f"{p}.mlp.experts.w3"] = torch.empty(e, mh, h, device="cuda")
    sd["model.embed_tokens.weight"] = torch.empty(v, h, device="cuda")
    sd["model.norm.weight"] = torch.empty(h, device="cuda")
    sd["lm_head.weight"] = torch.empty(v, h, device="cuda")
    return sd


def _trivial_dims() -> ParallelDims:
    return ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)


@pytest.fixture
def tiny_state() -> tuple[Qwen3MoeConfig, dict[str, torch.Tensor]]:
    config = _tiny_config()
    return config, _state_dict(config)


@pytest.fixture(
    params=[
        pytest.param(("bf16_cast", torch.bfloat16), id="bf16"),
        pytest.param(("fp8_128x128", torch.bfloat16), id="fp8"),
    ]
)
def inference_target(request) -> tuple[str, torch.dtype]:
    return request.param


def _is_dense_layer(config, layer_idx: int) -> bool:
    if layer_idx in config.mlp_only_layers:
        return True
    if config.num_experts == 0:
        return True
    return (layer_idx + 1) % config.decoder_sparse_step != 0


def _build(config, sd, default, base):
    slots = []
    dims = _trivial_dims()
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        tail = DENSE_LAYER_CONVERSION_SPEC if _is_dense_layer(config, i) else SPARSE_LAYER_CONVERSION_SPEC
        for spec in BASE_LAYER_CONVERSION_SPEC + tail:
            slots.extend(
                build_slots_for_conversion_spec(
                    spec, prefix=prefix, state_dict=sd, parallel_dims=dims, default_conversion=default, base_dtype=base
                )
            )
    for spec in NON_LAYER_CONVERSION_SPEC:
        slots.extend(
            build_slots_for_conversion_spec(
                spec, prefix="", state_dict=sd, parallel_dims=dims, default_conversion=default, base_dtype=base
            )
        )
    return slots


def test_dispatch_picks_expected_slot_types(tiny_state, inference_target):
    """Layernorms / router gate land as GatheredSlot; large projections as
    ShardedSlot; expert specs as ExpertSlot. Independent of inference target.
    """
    config, sd = tiny_state
    default, base = inference_target
    slots = _build(config, sd, default, base)
    by_key = {s.slot_key: s for s in slots}

    # Layernorms are 1D and tiny → GatheredSlot.
    assert isinstance(by_key["model.layers.0.input_layernorm.weight"], GatheredSlot)
    assert isinstance(by_key["model.norm.weight"], GatheredSlot)

    # Large 2D projections clear SMALL_NON_EXPERT_BYTES and divide cleanly → ShardedSlot.
    q = by_key["model.layers.0.self_attn.q_proj.weight"]
    assert isinstance(q, ShardedSlot)
    assert q.weight.numel() * q.weight.element_size() >= SMALL_NON_EXPERT_BYTES

    # Stacked-expert specs are always ExpertSlot.
    assert isinstance(by_key["model.layers.0.mlp.experts.w13_weight"], ExpertSlot)
    assert isinstance(by_key["model.layers.0.mlp.experts.w2_weight"], ExpertSlot)


def test_qkv_three_sources_yield_three_slots(tiny_state, inference_target):
    """A fused qkv ConversionSpec produces three independent slots
    (one per source) with offset_rows accumulated along the fused dim.
    """
    config, sd = tiny_state
    default, base = inference_target
    slots = _build(config, sd, default, base)
    qkv_slots = sorted(
        (
            s
            for s in slots
            if s.slot_key.startswith("model.layers.0.self_attn.")
            and s.slot_key.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight"))
        ),
        key=lambda s: s.offset_rows,
    )
    q, k, v = qkv_slots
    assert q.source_name == "model.layers.0.self_attn.q_proj.weight"
    assert k.source_name == "model.layers.0.self_attn.k_proj.weight"
    assert v.source_name == "model.layers.0.self_attn.v_proj.weight"
    assert q.offset_rows == 0
    assert k.offset_rows == q.rows
    assert v.offset_rows == q.rows + k.rows
    # All three share the fused inference destination.
    assert q.inference_name == k.inference_name == v.inference_name == "model.layers.0.self_attn.qkv_proj.weight"


def test_fp8_only_quantized_slots_carry_scale(tiny_state):
    """Pinned bf16 specs never get scale buffers under FP8 inference."""
    config, sd = tiny_state
    slots = _build(config, sd, "fp8_128x128", torch.bfloat16)
    by_key = {s.slot_key: s for s in slots}

    # Pinned in conversion spec: layernorms, router gate, model.norm, lm_head.
    for key in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.router.gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]:
        assert by_key[key].scale is None, f"{key} must stay non-quantized"

    # Default-resolution: scale buffer present; weight is fp8.
    q = by_key["model.layers.0.self_attn.q_proj.weight"]
    assert q.weight.dtype == torch.float8_e4m3fn
    assert q.scale is not None and q.scale.dtype == torch.float32
    # Scale uses per-source naming on trainer side, fused on inference side.
    assert q.scale_key == "model.layers.0.self_attn.q_proj.weight_scale_inv"
    assert q.inference_scale_name == "model.layers.0.self_attn.qkv_proj.weight_scale_inv"


def test_expert_slot_buffer_layout_and_writes(tiny_state):
    config, sd = tiny_state
    slots = _build(config, sd, "fp8_128x128", torch.bfloat16)
    w13 = next(s for s in slots if s.slot_key == "model.layers.0.mlp.experts.w13_weight")
    assert isinstance(w13, ExpertSlot)
    e = config.num_experts
    mh = config.moe_intermediate_size
    h = config.hidden_size
    # w1 + w3 fused along cat_dim=1 → (e, 2*mh, h).
    assert tuple(w13.weight.shape) == (e, 2 * mh, h)
    assert w13.scale is not None
    assert tuple(w13.scale.shape) == (e, ceil_div(2 * mh, BLOCK_SIZE), ceil_div(h, BLOCK_SIZE))
    # All experts owned in single-rank EP=1 setup.
    assert w13.owned_global_experts == list(range(e))
    # buffers report num_chunks == num_local_experts for the 3D path.
    assert [(k, t.shape, n) for k, t, n in w13.buffers] == [
        ("model.layers.0.mlp.experts.w13_weight", w13.weight.shape, e),
        ("model.layers.0.mlp.experts.w13_weight_scale_inv", w13.scale.shape, e),
    ]
    # Expert layout uses peer.expert_map, so layout_payload is empty.
    assert w13.layout_payload() == []

    # Build writes against a fake peer that owns experts 1 and 2.
    peer = PeerInfo(
        agent_name="inference-test-r0",
        agent_metadata=b"",
        tensor_addrs={},
        expert_map={"model.layers.0.mlp.experts": [1, 2]},
    )
    writes = w13.build_writes([peer])
    # Two experts × (weight + scale) = 4 writes; peer owns global experts 1, 2 at chunks 0, 1.
    assert len(writes) == 4
    by_chunks = sorted((w.local_chunk_idx, w.remote_chunk_idx, w.local_buffer_key) for w in writes)
    assert by_chunks == [
        (1, 0, "model.layers.0.mlp.experts.w13_weight"),
        (1, 0, "model.layers.0.mlp.experts.w13_weight_scale_inv"),
        (2, 1, "model.layers.0.mlp.experts.w13_weight"),
        (2, 1, "model.layers.0.mlp.experts.w13_weight_scale_inv"),
    ]


def test_sharded_slot_writes_target_my_rank_chunk_on_every_peer(tiny_state):
    config, sd = tiny_state
    slots = _build(config, sd, "bf16_cast", torch.bfloat16)
    q = next(s for s in slots if s.slot_key == "model.layers.0.self_attn.q_proj.weight")
    assert isinstance(q, ShardedSlot)
    peers = [
        PeerInfo(agent_name=f"inference-test-r{r}", agent_metadata=b"", tensor_addrs={}, expert_map={})
        for r in range(3)
    ]
    writes = q.build_writes(peers)
    # One write per (peer, buffer); single-rank trainer → my_rank=0, no scale buffer.
    assert {(w.peer_name, w.remote_chunk_idx) for w in writes} == {
        ("inference-test-r0", 0),
        ("inference-test-r1", 0),
        ("inference-test-r2", 0),
    }


def test_gathered_slot_round_robin_writes_when_single_rank(tiny_state):
    """With trainer_ws=1, a single trainer rank owns every gathered write."""
    config, sd = tiny_state
    slots = _build(config, sd, "bf16_cast", torch.bfloat16)
    norm = next(s for s in slots if s.slot_key == "model.layers.0.input_layernorm.weight")
    assert isinstance(norm, GatheredSlot)
    peers = [PeerInfo(agent_name=f"inf-r{r}", agent_metadata=b"", tensor_addrs={}, expert_map={}) for r in range(4)]
    writes = norm.build_writes(peers)
    # One write per peer; remote_chunk_idx=0 (gathered → single chunk).
    assert {w.peer_name for w in writes} == {"inf-r0", "inf-r1", "inf-r2", "inf-r3"}
    assert all(w.remote_chunk_idx == 0 for w in writes)


def test_materialize_roundtrip_on_sharded_slot(tiny_state):
    config, _ = tiny_state
    sd = _state_dict(config)
    g = torch.Generator(device="cuda").manual_seed(0)
    for k, v in sd.items():
        sd[k] = torch.randn(v.shape, generator=g, dtype=torch.float32, device="cuda")

    slots = _build(config, sd, "bf16_cast", torch.bfloat16)
    q = next(s for s in slots if s.slot_key == "model.layers.0.self_attn.q_proj.weight")
    q.convert(sd)
    # Single-rank: ShardedSlot's weight equals the source cast to bf16.
    expected = sd["model.layers.0.self_attn.q_proj.weight"].to(torch.bfloat16)
    torch.testing.assert_close(q.weight, expected)
