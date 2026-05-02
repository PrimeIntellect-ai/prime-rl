"""Trainer publisher end-to-end against a live ME server.

Builds a tiny synthetic Qwen3-MoE state dict on CUDA, hands it to the
publisher with an explicitly resolved conversion default + trivial
parallel dims, and verifies a peer rendezvous fetches identical NIXL
metadata + tensor descriptors whose addresses match registered buffers.
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
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transport.mx_rendezvous import MxRendezvous

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer publisher needs CUDA")
pytest.importorskip("nixl_cu13")

from prime_rl.transport.trainer_publisher import TrainerPublisher  # noqa: E402


def _tiny_qwen3_state() -> tuple[Qwen3MoeConfig, dict[str, torch.Tensor]]:
    config = Qwen3MoeConfig(
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
    return config, sd


def test_trainer_publisher_publishes_through_mx(mx_server):
    config, sd = _tiny_qwen3_state()
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
            state_dict=sd,
            parallel_dims=ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1),
        )
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
            expert_parallel_size=publisher.rendezvous.expert_parallel_size,
        )
        peer.publish(nixl_metadata=b"infer-stub", tensors=[])
        [trainer_ref] = peer.wait_for_peers(timeout=5)
        fetched = peer.fetch_peer(trainer_ref)

        assert fetched.nixl_metadata == publisher.agent.get_metadata()
        # Every registered buffer (weight + optional scale per slot) shows up.
        addrs = {td.name: td.addr for td in fetched.tensors}
        for slot in publisher.slots:
            for buf_key, tensor, _ in slot.buffers:
                assert addrs[buf_key] == tensor.data_ptr()
    finally:
        client.close()
