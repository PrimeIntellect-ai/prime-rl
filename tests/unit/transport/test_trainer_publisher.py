"""Trainer publisher end-to-end against a live ME server.

Builds shapes for a tiny synthetic Qwen3-MoE configuration, hands them to
the publisher with an explicitly resolved conversion default, and verifies
a peer rendezvous fetches identical NIXL metadata + a tensor descriptor
list whose addresses match the registered slot buffers.

Kept to a single test case — the per-variant resolution of
``select_default_conversion`` is covered by the conversion-spec tests.
"""

from __future__ import annotations

import uuid

import pytest
import torch
from modelexpress.client import MxClient
from transformers import Qwen3MoeConfig

from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import (
    conversion_specs,
    is_dense_layer,
    non_layer_conversion_specs,
)
from prime_rl.transport.mx_rendezvous import MxRendezvous

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer publisher needs CUDA"),
]
pytest.importorskip("nixl_cu13")

from prime_rl.transport.trainer_publisher import TrainerPublisher  # noqa: E402


def _tiny_qwen3_shapes() -> tuple[Qwen3MoeConfig, dict[str, tuple[int, ...]]]:
    """Shapes for one tiny Qwen3 MoE checkpoint (2 layers, all sparse)."""
    config = Qwen3MoeConfig(
        num_hidden_layers=2,
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
    h, mh = config.hidden_size, config.moe_intermediate_size
    n_q, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = h // n_q
    e, v = config.num_experts, config.vocab_size
    shapes: dict[str, tuple[int, ...]] = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        shapes[f"{p}.input_layernorm.weight"] = (h,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (h,)
        shapes[f"{p}.self_attn.q_norm.weight"] = (head_dim,)
        shapes[f"{p}.self_attn.k_norm.weight"] = (head_dim,)
        shapes[f"{p}.self_attn.q_proj.weight"] = (n_q * head_dim, h)
        shapes[f"{p}.self_attn.k_proj.weight"] = (n_kv * head_dim, h)
        shapes[f"{p}.self_attn.v_proj.weight"] = (n_kv * head_dim, h)
        shapes[f"{p}.self_attn.o_proj.weight"] = (h, n_q * head_dim)
        shapes[f"{p}.mlp.router.gate.weight"] = (e, h)
        shapes[f"{p}.mlp.experts.w1"] = (e, mh, h)
        shapes[f"{p}.mlp.experts.w2"] = (e, h, mh)
        shapes[f"{p}.mlp.experts.w3"] = (e, mh, h)
    shapes["model.embed_tokens.weight"] = (v, h)
    shapes["model.norm.weight"] = (h,)
    shapes["lm_head.weight"] = (v, h)
    return config, shapes


def test_trainer_publisher_publishes_through_mx(mx_server):
    config, shapes = _tiny_qwen3_shapes()
    model_name = f"prime-rl-test/{uuid.uuid4().hex}"

    client = MxClient(server_url=mx_server)
    try:
        publisher = TrainerPublisher(
            client=client,
            rank=0,
            peer_world_size=1,
            inference_model_name=model_name,
            default_conversion="fp8_128x128",
            base_dtype=torch.bfloat16,
            layer_specs_fn=conversion_specs,
            non_layer_specs=non_layer_conversion_specs(),
            is_dense_fn=lambda i: is_dense_layer(config, i),
            num_layers=config.num_hidden_layers,
            state_shapes=shapes,
        )
        # Slot count: per-layer (BASE+SPARSE = 9) + non-layer (3).
        assert len(publisher.slots) == config.num_hidden_layers * 9 + 3
        # FP8 default → at least some scale buffers exist; pinned passthrough specs do not.
        assert any(s.scale is not None for s in publisher.slots)
        assert any(s.scale is None for s in publisher.slots)

        publisher.publish()

        peer = MxRendezvous(
            client=client,
            role="inference",
            rank=0,
            peer_world_size=1,
            model_name=model_name,
            quantization=publisher.rendezvous.quantization,
        )
        peer.publish(nixl_metadata=b"infer-stub", tensors=[])
        [trainer_ref] = peer.wait_for_peers(timeout=5)
        fetched = peer.fetch_peer(trainer_ref)

        assert fetched.nixl_metadata == publisher.agent.get_metadata()
        addrs = {td.name: td.addr for td in fetched.tensors}
        for slot in publisher.slots:
            assert addrs[slot.full_name] == slot.weight.data_ptr()
            if slot.scale is not None:
                assert addrs[slot.scale_name] == slot.scale.data_ptr()
    finally:
        client.close()
