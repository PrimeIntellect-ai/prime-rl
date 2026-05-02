"""Trainer publisher + inference receiver + transport-plan negotiation
against a live ME server.

Runs both halves in-process: the publisher registers trainer slots and
publishes the layout; the receiver registers vLLM-shaped destinations,
publishes its expert_map, and flips to ``READY``; the transport plan
fetches the peer's payload + tensor base addresses and assembles the
write table. Verifies addresses round-trip and the negotiated peer list
+ write table are non-empty.
"""

from __future__ import annotations

import uuid

import pytest
import torch
from modelexpress.client import MxClient
from transformers import Qwen3MoeConfig

from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import (
    conversion_specs,
    is_dense_layer,
    non_layer_conversion_specs,
)
from prime_rl.trainer.parallel_dims import ParallelDims

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer publisher needs CUDA")
pytest.importorskip("nixl_cu13")

from prime_rl.transport.inference_receiver import InferenceReceiver  # noqa: E402
from prime_rl.transport.trainer_publisher import TrainerPublisher  # noqa: E402
from prime_rl.transport.transport_plan import TransportPlan  # noqa: E402


def _tiny_config() -> Qwen3MoeConfig:
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


def _trainer_state(config: Qwen3MoeConfig) -> dict[str, torch.Tensor]:
    """TT-format trainer state dict: per-source q/k/v/o, per-expert w1/w2/w3."""
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


def _inference_state(
    config: Qwen3MoeConfig, default_conversion: str, base_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """vLLM-format inference state dict: fused qkv_proj, fused experts.w13/w2."""
    h, mh = config.hidden_size, config.moe_intermediate_size
    n_q, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = h // n_q
    e, v = config.num_experts, config.vocab_size
    qkv_rows = (n_q + 2 * n_kv) * head_dim
    fp8 = default_conversion == "fp8_128x128"
    quant_dtype = torch.float8_e4m3fn if fp8 else base_dtype

    sd: dict[str, torch.Tensor] = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        # Pinned passthrough specs — always base_dtype, no scale buffer.
        sd[f"{p}.input_layernorm.weight"] = torch.empty(h, dtype=base_dtype, device="cuda")
        sd[f"{p}.post_attention_layernorm.weight"] = torch.empty(h, dtype=base_dtype, device="cuda")
        sd[f"{p}.self_attn.q_norm.weight"] = torch.empty(head_dim, dtype=base_dtype, device="cuda")
        sd[f"{p}.self_attn.k_norm.weight"] = torch.empty(head_dim, dtype=base_dtype, device="cuda")
        sd[f"{p}.mlp.gate.weight"] = torch.empty(e, h, dtype=base_dtype, device="cuda")
        # Default-resolution weights — fp8 with scale_inv when FP8 inference.
        sd[f"{p}.self_attn.qkv_proj.weight"] = torch.empty(qkv_rows, h, dtype=quant_dtype, device="cuda")
        sd[f"{p}.self_attn.o_proj.weight"] = torch.empty(h, n_q * head_dim, dtype=quant_dtype, device="cuda")
        sd[f"{p}.mlp.experts.w13_weight"] = torch.empty(e, 2 * mh, h, dtype=quant_dtype, device="cuda")
        sd[f"{p}.mlp.experts.w2_weight"] = torch.empty(e, h, mh, dtype=quant_dtype, device="cuda")
        if fp8:
            sd[f"{p}.self_attn.qkv_proj.weight_scale_inv"] = torch.empty(
                ceil_div(qkv_rows, BLOCK_SIZE), ceil_div(h, BLOCK_SIZE), dtype=torch.float32, device="cuda"
            )
            sd[f"{p}.self_attn.o_proj.weight_scale_inv"] = torch.empty(
                ceil_div(h, BLOCK_SIZE), ceil_div(n_q * head_dim, BLOCK_SIZE), dtype=torch.float32, device="cuda"
            )
            sd[f"{p}.mlp.experts.w13_weight_scale_inv"] = torch.empty(
                e, ceil_div(2 * mh, BLOCK_SIZE), ceil_div(h, BLOCK_SIZE), dtype=torch.float32, device="cuda"
            )
            sd[f"{p}.mlp.experts.w2_weight_scale_inv"] = torch.empty(
                e, ceil_div(h, BLOCK_SIZE), ceil_div(mh, BLOCK_SIZE), dtype=torch.float32, device="cuda"
            )
    sd["model.embed_tokens.weight"] = torch.empty(v, h, dtype=quant_dtype, device="cuda")
    if fp8:
        sd["model.embed_tokens.weight_scale_inv"] = torch.empty(
            ceil_div(v, BLOCK_SIZE), ceil_div(h, BLOCK_SIZE), dtype=torch.float32, device="cuda"
        )
    sd["model.norm.weight"] = torch.empty(h, dtype=base_dtype, device="cuda")
    sd["lm_head.weight"] = torch.empty(v, h, dtype=base_dtype, device="cuda")
    return sd


def test_two_pass_rendezvous_over_mx(mx_server):
    config = _tiny_config()
    model_name = f"prime-rl-test/{uuid.uuid4().hex}"
    default_conversion = "fp8_128x128"
    base_dtype = torch.bfloat16

    client = MxClient(server_url=mx_server)
    try:
        publisher = TrainerPublisher(
            client=client,
            rank=0,
            peer_world_size=1,
            inference_model_name=model_name,
            default_conversion=default_conversion,
            base_dtype=base_dtype,
            layer_specs_fn=conversion_specs,
            non_layer_specs=non_layer_conversion_specs(),
            is_dense_fn=lambda i: is_dense_layer(config, i),
            num_layers=config.num_hidden_layers,
            state_dict=_trainer_state(config),
            parallel_dims=ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1),
        )
        # FP8 default → at least some scale buffers exist; pinned passthrough specs do not.
        assert any(s.scale is not None for s in publisher.slots)
        assert any(s.scale is None for s in publisher.slots)

        inference_params = _inference_state(config, default_conversion, base_dtype)
        # Single inference rank: owns every global expert at every MoE prefix.
        all_experts = list(range(config.num_experts))
        expert_map = {f"model.layers.{i}.mlp.experts": all_experts for i in range(config.num_hidden_layers)}
        receiver = InferenceReceiver(
            client=client,
            rank=0,
            peer_world_size=1,
            inference_model_name=model_name,
            live_tensors=inference_params,
            expert_map=expert_map,
        )

        publisher.publish()
        receiver.publish()
        receiver.mark_ready()

        plan = TransportPlan(publisher)
        plan.negotiate(timeout=5)

        assert len(plan.peers) == 1
        peer = plan.peers[0]
        assert peer.agent_name == receiver.agent.name
        assert peer.agent_metadata == receiver.agent.get_metadata()
        # Inference base addresses round-trip and match the registered live params.
        for name, tensor in inference_params.items():
            base, size, _ = peer.tensor_addrs[name]
            assert base == tensor.data_ptr()
            assert size == tensor.numel() * tensor.element_size()
        # Write table is non-empty (every peer × every slot's writes).
        assert plan.writes
    finally:
        client.close()
