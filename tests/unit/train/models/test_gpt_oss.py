import os

import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM
from prime_rl.trainer.models.gpt_oss.attention import GptOssAttention, substitute_gpt_oss_ulysses_attention


def _config() -> GptOssConfig:
    return GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=4,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_experts_per_tok=2,
        sliding_window=4,
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
        attn_implementation="flash_attention_4",
        use_cache=False,
    )


def test_gpt_oss_checkpoint_conversion_roundtrip():
    with torch.device("meta"):
        model = PrimeRLGptOssForCausalLM(_config())

    prime_state_dict = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    expected_prime = {name: tensor.clone() for name, tensor in prime_state_dict.items()}
    hf_state_dict = model.convert_to_hf(prime_state_dict)

    assert model.is_hf_state_dict(hf_state_dict)
    assert not model.is_prime_state_dict(hf_state_dict)

    roundtrip = model.convert_to_prime(hf_state_dict)
    assert roundtrip.keys() == expected_prime.keys()
    for name, tensor in roundtrip.items():
        torch.testing.assert_close(tensor, expected_prime[name])


@pytest.mark.gpu
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_gpt_oss_attention_matches_flash_attention_4(layer_type: str):
    if torch.cuda.get_device_capability()[0] not in (9, 10, 11):
        pytest.skip("GPT-OSS learned sinks require SM90 or SM100/SM110")
    flash_attn_cute = pytest.importorskip("flash_attn.cute")

    torch.manual_seed(0)
    config = _config()
    config.head_dim = 64
    config.layer_types = [layer_type]
    attention = GptOssAttention(config, layer_idx=0).cuda().to(torch.bfloat16)
    inputs = [
        torch.randn(8, heads, 64, device="cuda", dtype=torch.bfloat16)
        for heads in (config.num_attention_heads, config.num_key_value_heads, config.num_key_value_heads)
    ]
    cu_seqlens = torch.tensor([0, 5, 8], device="cuda", dtype=torch.int32)
    window_size = (attention.sliding_window - 1, 0) if attention.sliding_window is not None else (None, None)

    reference_inputs = [tensor.clone().requires_grad_() for tensor in inputs]
    reference_sinks = attention.sinks.detach().clone().requires_grad_()
    reference, _ = flash_attn_cute.flash_attn_varlen_func(
        *reference_inputs,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=5,
        max_seqlen_k=5,
        softmax_scale=attention.scaling,
        causal=True,
        window_size=window_size,
        learnable_sink=reference_sinks,
    )
    reference.float().square().sum().backward()

    actual_inputs = [tensor.clone().requires_grad_() for tensor in inputs]
    actual = attention.compute_attention(*actual_inputs, cu_seqlens, 5)
    actual.float().square().sum().backward()

    torch.testing.assert_close(actual, reference)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, reference_input.grad)
    torch.testing.assert_close(attention.sinks.grad, reference_sinks.grad)


@pytest.fixture(scope="module")
def cp_process_group():
    if int(os.environ.get("WORLD_SIZE", 1)) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")
    if torch.cuda.get_device_capability()[0] not in (9, 10, 11):
        pytest.skip("GPT-OSS learned sinks require SM90 or SM100/SM110")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    yield dist.group.WORLD
    dist.destroy_process_group()


@pytest.mark.gpu
def test_gpt_oss_context_parallel_attention(cp_process_group):
    local_rank = int(os.environ["LOCAL_RANK"])
    process_group = cp_process_group
    original_compute_attention = GptOssAttention.compute_attention

    try:
        torch.manual_seed(0)
        config = _config()
        config.head_dim = 64
        attention = GptOssAttention(config, layer_idx=0).cuda().to(torch.bfloat16)
        query = torch.randn(8, 4, 64, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for tensor in (attention.sinks, query, key, value):
                dist.broadcast(tensor, src=0)
        cu_seqlens = torch.tensor([0, 5, 8], device="cuda", dtype=torch.int32)

        reference_query = query.detach().clone().requires_grad_()
        reference_key = key.detach().clone().requires_grad_()
        reference_value = value.detach().clone().requires_grad_()
        reference = original_compute_attention(
            attention,
            reference_query,
            reference_key,
            reference_value,
            cu_seqlens,
            5,
        )
        reference_output = reference.detach().clone()
        reference.float().square().sum().backward()
        reference_sink_grad = attention.sinks.grad.detach().clone()
        attention.sinks.grad = None

        substitute_gpt_oss_ulysses_attention(process_group)

        local_slice = slice(local_rank * 4, (local_rank + 1) * 4)
        local_query = query[local_slice].detach().clone().requires_grad_()
        local_key = key[local_slice].detach().clone().requires_grad_()
        local_value = value[local_slice].detach().clone().requires_grad_()
        actual = attention.compute_attention(local_query, local_key, local_value, cu_seqlens, 5)
        actual.float().square().sum().backward()

        gathered = [torch.empty_like(actual) for _ in range(2)]
        dist.all_gather(gathered, actual)
        torch.testing.assert_close(torch.cat(gathered), reference_output, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(local_query.grad, reference_query.grad[local_slice], rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(local_key.grad, reference_key.grad[local_slice], rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(local_value.grad, reference_value.grad[local_slice], rtol=5e-2, atol=5e-2)

        sink_grad = attention.sinks.grad
        dist.all_reduce(sink_grad)
        torch.testing.assert_close(sink_grad, reference_sink_grad, rtol=5e-2, atol=5e-2)
    finally:
        GptOssAttention.compute_attention = original_compute_attention
