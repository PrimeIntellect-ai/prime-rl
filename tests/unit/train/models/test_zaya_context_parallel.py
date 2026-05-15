import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbeddingConfig
from prime_rl.trainer.models.zaya.configuration_zaya import ZayaConfig
from prime_rl.trainer.models.zaya.modeling_zaya import (
    ZayaCCAProjection,
    ZayaFlashAttention,
    ZayaQKNorm,
    ZayaRotaryEmbedding,
    ZayaSPDAAttention,
)


def _tiny_zaya_config() -> ZayaConfig:
    return ZayaConfig(
        vocab_size=32,
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=1,
        num_experts=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_query_groups=2,
        head_dim=4,
        moe_intermediate_size=8,
        router_hidden_size=4,
        use_grouped_mm=False,
        _attn_implementation="sdpa",
    )


def test_zaya_attention_cp_attributes_propagate_to_children():
    config = _tiny_zaya_config()
    attention = ZayaFlashAttention(config, layer_idx=0)
    cp_group = object()

    attention.set_context_parallel_attributes(cp_group, cp_rank=1, cp_world_size=2)

    assert attention.cp_enabled
    assert attention._cp_group is cp_group
    assert attention._cp_rank == 1
    assert attention._cp_world_size == 2
    assert attention.qkv_proj._cp_group is cp_group
    assert attention.qkv_proj._cp_rank == 1
    assert attention.qkv_proj._cp_world_size == 2
    assert attention.qk_norm._cp_rank == 1
    assert attention.qk_norm._cp_world_size == 2


def test_zaya_cca_cp_channel_indices_preserve_q_then_k_order():
    config = _tiny_zaya_config()
    projection = ZayaCCAProjection(config, layer_idx=0)
    projection.set_context_parallel_attributes(object(), cp_rank=1, cp_world_size=2)

    channel_indices = projection._local_head_channel_indices()

    assert channel_indices.tolist() == [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23]


def test_zaya_qk_norm_slices_temperature_for_local_kv_heads():
    config = _tiny_zaya_config()
    qk_norm = ZayaQKNorm(config, scaling=config.head_dim**-0.5)
    qk_norm.temp.data = torch.tensor([2.0, 3.0])
    qk_norm.set_context_parallel_attributes(object(), cp_rank=1, cp_world_size=2)

    query_states = torch.ones(1, 2, 2, config.head_dim)
    key_states = torch.ones(1, 2, 1, config.head_dim)

    query_out, key_out = qk_norm(query_states, key_states)

    expected_norm = config.head_dim**0.5
    assert torch.allclose(query_out.norm(p=2, dim=-1), torch.full((1, 2, 2), expected_norm))
    assert torch.allclose(key_out, torch.full_like(key_out, 3.0))


def _zaya_position_embeddings(config: ZayaConfig, hidden_states: torch.Tensor):
    rope_parameters = config.rope_parameters["hybrid"]
    rope_config = copy.copy(config)
    rope_config.rope_parameters = rope_parameters
    rotary_emb = ZayaRotaryEmbedding(
        RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_parameters["rope_type"],
            model_config=rope_config,
        )
    ).to(hidden_states.device)
    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    return rotary_emb(hidden_states, position_ids)


def _run_zaya_attention_cp_parity(
    rank: int,
    world_size: int,
    init_file: str,
    attention_name: str,
    backend: str,
    device_type: str,
    check_backward: bool,
):
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device(device_type)
        torch.manual_seed(0)
        config = _tiny_zaya_config()
        attention_cls = {"sdpa": ZayaSPDAAttention, "flash": ZayaFlashAttention}[attention_name]
        dtype = torch.bfloat16 if attention_name == "flash" else torch.float32
        full_attention = attention_cls(config, layer_idx=0).to(device=device, dtype=dtype)
        cp_attention = attention_cls(config, layer_idx=0).to(device=device, dtype=dtype)
        cp_attention.load_state_dict(full_attention.state_dict())
        cp_attention.set_context_parallel_attributes(dist.group.WORLD, rank, world_size)
        full_attention.train(check_backward)
        cp_attention.train(check_backward)

        hidden_states = torch.randn(1, 8, config.hidden_size, device=device, dtype=dtype)
        position_embeddings = _zaya_position_embeddings(config, hidden_states)
        cu_seqlens = torch.tensor([0, hidden_states.shape[1]], device=device, dtype=torch.int32)
        max_seqlen = hidden_states.shape[1]

        full_hidden_states = hidden_states.detach().clone().requires_grad_(check_backward)
        cp_hidden_states = (
            hidden_states.chunk(world_size, dim=1)[rank].detach().clone().contiguous().requires_grad_(check_backward)
        )

        full_output, _ = full_attention(
            full_hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        cp_output, _ = cp_attention(
            cp_hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        expected = full_output.chunk(world_size, dim=1)[rank]
        atol = 1e-2 if attention_name == "flash" else 1e-5
        rtol = 1e-2 if attention_name == "flash" else 1e-5
        max_diff = (cp_output - expected).abs().max().item()
        assert torch.allclose(cp_output, expected, atol=atol, rtol=rtol), max_diff

        if check_backward:
            grad = torch.randn_like(full_output)
            full_output.backward(grad)
            cp_output.backward(grad.chunk(world_size, dim=1)[rank])

            hidden_grad = full_hidden_states.grad.chunk(world_size, dim=1)[rank]
            max_hidden_grad_diff = (cp_hidden_states.grad - hidden_grad).abs().max().item()
            assert torch.allclose(cp_hidden_states.grad, hidden_grad, atol=atol, rtol=rtol), max_hidden_grad_diff

            for (name, full_param), cp_param in zip(full_attention.named_parameters(), cp_attention.parameters()):
                dist.all_reduce(cp_param.grad, op=dist.ReduceOp.SUM)
                max_grad_diff = (cp_param.grad - full_param.grad).abs().max().item()
                assert torch.allclose(cp_param.grad, full_param.grad, atol=atol, rtol=rtol), (name, max_grad_diff)
    finally:
        dist.destroy_process_group()


def test_zaya_sdpa_context_parallel_matches_non_cp_output_and_backward(tmp_path):
    world_size = 2
    init_file = tmp_path / "dist_init"

    mp.spawn(
        _run_zaya_attention_cp_parity,
        args=(world_size, init_file.as_posix(), "sdpa", "gloo", "cpu", True),
        nprocs=world_size,
        join=True,
    )


def test_zaya_flash_context_parallel_matches_non_cp_output_and_backward(tmp_path):
    pytest.importorskip("flash_attn")
    if torch.cuda.device_count() < 2:
        pytest.skip("Zaya FlashAttention CP parity test requires at least two CUDA devices")

    world_size = 2
    init_file = tmp_path / "dist_init"

    mp.spawn(
        _run_zaya_attention_cp_parity,
        args=(world_size, init_file.as_posix(), "flash", "nccl", "cuda", True),
        nprocs=world_size,
        join=True,
    )
