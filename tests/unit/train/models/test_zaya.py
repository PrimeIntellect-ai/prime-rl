# This script should be run with "https://github.com/JJJYmmm/transformers.git" which checks against the (likely) merged transformers PR for Zaya
import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.distributed.expert_parallel import ExpertParallel
from transformers import ZayaForCausalLM as HFZayaForCausalLM

# There is something wrong with the quack RMSNorm vs the FP32 implementation at least on (SM120)
import prime_rl.trainer.models.layers.norms as norms
norms._get_quack_rmsnorm = lambda: None

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbeddingConfig
from prime_rl.trainer.models.zaya import ZayaConfig
from prime_rl.trainer.models.zaya import ZayaForCausalLM as PrimeRLZayaForCausalLM
from prime_rl.trainer.models.zaya.modeling_zaya import (
    ZayaCCAProjection,
    ZayaDecoderLayer,
    ZayaFlashAttention,
    ZayaQKNorm,
    ZayaRotaryEmbedding,
    ZayaSPDAAttention,
)
from prime_rl.trainer.weights import load_state_dict
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

LOGITS_ATOL = 2e-2
EMBED_GRAD_ATOL = 2
_REQUIRES_TWO_CUDA_DEVICES_MSG = "Zaya distributed parity tests require at least two CUDA devices"


def _tiny_config(attn_implementation: str = "sdpa"):
    config = ZayaConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_hidden_size=16,
        num_hidden_layers=4,
        num_experts=3,
        num_attention_heads=4,
        num_query_groups=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        norm_epsilon=1e-5,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        moe_router_topk=1,
        zaya_mlp_expansion=8,
        zaya_use_mod=True,
        zaya_use_eda=True,
        add_bias_linear=False,
        attention_bias=False,
        lm_head_bias=False,
        tie_word_embeddings=True,
        use_cache=False,
        use_grouped_mm=False,
    )
    config._attn_implementation = attn_implementation
    # HF `ZayaModel` uses `layer_types` and `rope_parameters[layer_type]`; Prime expects `rope_parameters["hybrid"]`.
    config.layer_types = ["hybrid"] * config.num_hidden_layers
    config.rope_parameters = {
        "hybrid": {
            "rope_type": "default",
            "rope_theta": float(config.rope_theta),
            "partial_rotary_factor": float(config.partial_rotary_factor),
        }
    }
    return config


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state_dict.items()}


def get_model_pairs(
    hf_attn_implementation: str = "sdpa",
    prime_attn_implementation: str | None = None,
    dtype: torch.dtype = torch.float32,
):
    if prime_attn_implementation is None:
        prime_attn_implementation = hf_attn_implementation
    hf_config = _tiny_config(hf_attn_implementation)
    prime_config = _tiny_config(prime_attn_implementation)

    with torch.device("cuda"), default_dtype(dtype):
        hf_model = HFZayaForCausalLM(hf_config)
        prime_model = PrimeRLZayaForCausalLM._from_config(prime_config)

    with torch.no_grad():
        state_dict = _clone_state_dict(hf_model.state_dict())
        prime_state_keys = set(prime_model.state_dict().keys())
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)
    assert prime_state_keys - set(state_dict.keys()) == set()
    return hf_model, prime_model


def _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids, attention_mask=None) -> None:
    hf_output = hf_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    prime_output = prime_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    hf_output.logits.float().sum().backward()
    prime_output["logits"].float().sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=LOGITS_ATOL), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )

    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=EMBED_GRAD_ATOL), (
        f"Max grad diff: {grad_diff.abs().max()}"
    )


class _PassthroughPrimeZayaBlock(nn.Module):
    def forward(self, hidden_states, prev_router_hidden_states=None, routed_experts=None):
        return hidden_states, prev_router_hidden_states


class _PassthroughHfZayaMoe(nn.Module):
    """HF `ZayaSparseMoeBlock` returns `(hidden_states, prev_router_hidden_states)`."""

    def forward(self, hidden_states, prev_router_hidden_states=None):
        return hidden_states, prev_router_hidden_states


def test_zaya_attn_only() -> None:
    hf_model, prime_model = get_model_pairs()

    for layer in hf_model.model.layers:
        layer.mlp = _PassthroughHfZayaMoe()
    for layer in prime_model.model.layers:
        layer.mlp = _PassthroughPrimeZayaBlock()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)


def test_zaya_mlp_only() -> None:
    hf_model, prime_model = get_model_pairs()

    def identity_attn_hf(hidden_states, *args, **kwargs):
        return hidden_states, None

    def identity_attn_prime(
        hidden_states,
        *args,
        **kwargs,
    ):
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = identity_attn_hf
    for layer in prime_model.model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.forward = identity_attn_prime

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)


def test_zaya_packed_matches_unpacked() -> None:
    _, prime_model = get_model_pairs()
    prime_model.eval()

    with torch.device("cuda"), default_dtype(torch.float32), torch.no_grad():
        first_ids = torch.randint(0, prime_model.config.vocab_size, (1, 5))
        second_ids = torch.randint(0, prime_model.config.vocab_size, (1, 7))
        packed_ids = torch.cat([first_ids, second_ids], dim=1)
        packed_position_ids = torch.cat(
            [
                torch.arange(first_ids.shape[1]),
                torch.arange(second_ids.shape[1]),
            ]
        ).unsqueeze(0)

        first_logits = prime_model(input_ids=first_ids, position_ids=torch.arange(first_ids.shape[1]).unsqueeze(0))[
            "logits"
        ]
        second_logits = prime_model(
            input_ids=second_ids,
            position_ids=torch.arange(second_ids.shape[1]).unsqueeze(0),
        )["logits"]
        packed_logits = prime_model(input_ids=packed_ids, position_ids=packed_position_ids)["logits"]

    expected_logits = torch.cat([first_logits, second_logits], dim=1)
    logits_diff = packed_logits - expected_logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=LOGITS_ATOL), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )


@pytest.mark.slow
def test_zaya() -> None:
    snapshot = Path(snapshot_download(repo_id="JJJYmmm/ZAYA1-8B-HF", repo_type="model"))
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # hf_model = HFZayaForCausalLM.from_pretrained("Zyphra/ZAYA1-8B", torch_dtype=dtype) # Original Zyphra weights (official)
    hf_model = HFZayaForCausalLM.from_pretrained("JJJYmmm/ZAYA1-8B-HF", torch_dtype=dtype) # HF PR 
    hf_model.to(device)
    attn_impl = getattr(
        hf_model.config,
        "_attn_implementation",
        getattr(hf_model.config, "attn_implementation", "sdpa"),
    )
    prime_config = ZayaConfig.from_pretrained(snapshot)
    prime_config._attn_implementation = attn_impl

    prime_model = PrimeRLZayaForCausalLM._from_config(prime_config)
    sd = load_state_dict(snapshot)
    PrimeRLZayaForCausalLM.convert_to_prime(sd)
    prime_model.load_state_dict(sd, strict=False)

    prime_model.to(device=device, dtype=dtype)
    prime_model.eval()
    hf_model.eval()

    vocab = hf_model.config.vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (4, 16), device=device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(4, -1)

    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        prime_out = prime_model(input_ids=input_ids, position_ids=position_ids)

    hf_logits = hf_out.logits.float().cpu()
    prime_logits = prime_out["logits"].float().cpu()
    max_abs = (prime_logits - hf_logits).abs().max().item()

    assert torch.allclose(prime_logits, hf_logits, atol=5e-2), (
        f"Forward logits mismatch max abs diff {max_abs} (atol=5e-2)"
    )


def test_zaya_tiny_roundtrip() -> None:
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_from_prime_model = HFZayaForCausalLM(hf_model.config)
        converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
        hf_from_prime_model.load_state_dict(converted_state_dict)

    hf_model.zero_grad(set_to_none=True)
    hf_from_prime_model.zero_grad(set_to_none=True)
    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    hf_from_prime_output = hf_from_prime_model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    hf_output.logits.sum().backward()
    hf_from_prime_output.logits.sum().backward()

    logits_diff = hf_from_prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=LOGITS_ATOL), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_from_prime_model.model.embed_tokens.weight.grad - hf_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=EMBED_GRAD_ATOL), (
        f"Max grad diff: {grad_diff.abs().max()}"
    )


def test_zaya_attention_mask() -> None:
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -3:] = 0

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids, attention_mask)


def _run_zaya_moe_expert_parallel_parity(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        torch.manual_seed(0)
        config = _tiny_zaya_config()
        config.num_experts = world_size
        config.num_experts_per_tok = 1
        config.moe_router_topk = 1
        config.use_grouped_mm = False

        with torch.device(device), default_dtype(torch.float32):
            local_layer = ZayaDecoderLayer(config, layer_idx=0)
            ep_layer = ZayaDecoderLayer(config, layer_idx=0)
        ep_layer.load_state_dict(local_layer.state_dict())
        parallelize_module(
            ep_layer.mlp.experts,
            DeviceMesh("cuda", list(range(world_size))),
            ExpertParallel(),
        )

        def identity_attn(hidden_states, *args, **kwargs):
            return hidden_states, None

        local_layer.self_attn.forward = identity_attn
        ep_layer.self_attn.forward = identity_attn
        local_layer.train()
        ep_layer.train()

        hidden_states = torch.randn(2, 8, config.hidden_size, device=device, requires_grad=True)
        ep_hidden_states = hidden_states.detach().clone().requires_grad_()
        routed_experts = (torch.arange(hidden_states.shape[1], device=device).reshape(1, -1, 1) % world_size).expand(
            hidden_states.shape[0], -1, -1
        )

        local_output, _ = local_layer(hidden_states, routed_experts=routed_experts)
        ep_output, _ = ep_layer(ep_hidden_states, routed_experts=routed_experts)

        max_diff = (ep_output - local_output).abs().max().item()
        assert torch.allclose(ep_output, local_output, atol=1e-5, rtol=1e-5), max_diff

        grad = torch.randn_like(local_output)
        local_output.backward(grad)
        ep_output.backward(grad)

        max_hidden_grad_diff = (ep_hidden_states.grad - hidden_states.grad).abs().max().item()
        assert torch.allclose(ep_hidden_states.grad, hidden_states.grad, atol=1e-5, rtol=1e-5), max_hidden_grad_diff
    finally:
        dist.destroy_process_group()

@pytest.mark.gpu
def test_zaya_moe_expert_parallel_matches_local_output_and_backward(tmp_path) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip(_REQUIRES_TWO_CUDA_DEVICES_MSG)
    world_size = 2
    init_file = tmp_path / "dist_init"

    mp.spawn(
        _run_zaya_moe_expert_parallel_parity,
        args=(world_size, init_file.as_posix()),
        nprocs=world_size,
        join=True,
    )


def test_zaya_flash_attention_2() -> None:
    pytest.importorskip("flash_attn")
    torch.manual_seed(0)
    hf_model, prime_model = get_model_pairs(prime_attn_implementation="flash_attention_2", dtype=torch.bfloat16)

    with torch.device("cuda"):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)


def _tiny_zaya_config() -> ZayaConfig:
    config = ZayaConfig(
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
    )
    config._attn_implementation = "sdpa"
    return config


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
    qk_norm = ZayaQKNorm(config)
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
    if torch.cuda.device_count() < 2:
        pytest.skip(_REQUIRES_TWO_CUDA_DEVICES_MSG)
    world_size = 2
    init_file = tmp_path / "dist_init"

    mp.spawn(
        _run_zaya_attention_cp_parity,
        args=(world_size, init_file.as_posix(), "sdpa", "nccl", "cuda", True),
        nprocs=world_size,
        join=True,
    )


def test_zaya_flash_context_parallel_matches_non_cp_output_and_backward(tmp_path):
    pytest.importorskip("flash_attn")
    if torch.cuda.device_count() < 2:
        pytest.skip(_REQUIRES_TWO_CUDA_DEVICES_MSG)

    world_size = 2
    init_file = tmp_path / "dist_init"

    mp.spawn(
        _run_zaya_attention_cp_parity,
        args=(world_size, init_file.as_posix(), "flash", "nccl", "cuda", True),
        nprocs=world_size,
        join=True,
    )