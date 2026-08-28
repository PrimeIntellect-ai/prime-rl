import os

import pytest
import torch
import torch.distributed as dist
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as HFGptOssForCausalLM

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM
from prime_rl.trainer.models.gpt_oss.attention import (
    GptOssAttention,
    substitute_gpt_oss_ring_attention,
    substitute_gpt_oss_ulysses_attention,
)
from prime_rl.utils.cp import setup_cp_attention_params


def _config(attn_implementation: str = "flash_attention_4") -> GptOssConfig:
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
        attn_implementation=attn_implementation,
    )


def test_gpt_oss_checkpoint_conversion_roundtrip():
    hf_config = _config("eager")
    prime_config = _config()
    with torch.device("meta"):
        hf_model = HFGptOssForCausalLM(hf_config)
        prime_model = PrimeRLGptOssForCausalLM(prime_config)

    hf_state_dict = {name: torch.randn(tensor.shape) for name, tensor in hf_model.state_dict().items()}
    expected_hf = {name: tensor.clone() for name, tensor in hf_state_dict.items()}
    prime_state_dict = prime_model.convert_to_prime(hf_state_dict)

    assert prime_model.is_prime_state_dict(prime_state_dict)
    assert not prime_model.is_hf_state_dict(prime_state_dict)
    assert set(prime_state_dict) == set(prime_model.state_dict())
    for name, tensor in prime_state_dict.items():
        assert tensor.shape == prime_model.state_dict()[name].shape, name

    roundtrip = prime_model.convert_to_hf(prime_state_dict)
    assert roundtrip.keys() == expected_hf.keys()
    for name, tensor in roundtrip.items():
        torch.testing.assert_close(tensor, expected_hf[name])


@pytest.mark.gpu
def test_gpt_oss_matches_hf():
    hf_config = _config("eager")
    prime_config = _config()
    with torch.device("cuda"):
        hf_model = HFGptOssForCausalLM(hf_config).to(torch.bfloat16)
        prime_model = PrimeRLGptOssForCausalLM(prime_config).to(torch.bfloat16)

    state_dict = hf_model.state_dict()
    prime_model.convert_to_prime(state_dict)
    prime_model.load_state_dict(state_dict)

    hidden_states = torch.randn(2, 8, hf_config.hidden_size, device="cuda", dtype=torch.bfloat16)
    expected, _ = hf_model.model.layers[0].mlp(hidden_states)
    actual = prime_model.model.layers[0].mlp(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    input_ids = torch.randint(0, hf_config.vocab_size, (1, 12), device="cuda")
    seq_lens = torch.tensor([5, 7], device="cuda")
    position_ids = torch.cat([torch.arange(5, device="cuda"), torch.arange(7, device="cuda")]).unsqueeze(0)
    with torch.no_grad():
        expected = torch.cat(
            [
                hf_model.model(input_ids=input_ids[:, :5]).last_hidden_state,
                hf_model.model(input_ids=input_ids[:, 5:]).last_hidden_state,
            ],
            dim=1,
        )
        actual = prime_model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_lens=seq_lens,
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    with torch.no_grad():
        expected_logits = torch.cat(
            [hf_model(input_ids=input_ids[:, :5]).logits, hf_model(input_ids=input_ids[:, 5:]).logits],
            dim=1,
        )
        actual_logits = prime_model(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_lens=seq_lens,
        )["logits"]
    torch.testing.assert_close(actual_logits, expected_logits, rtol=3e-2, atol=3e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("cp_style", ["ring", "ulysses"])
def test_gpt_oss_context_parallel_attention(cp_style: str):
    if int(os.environ.get("WORLD_SIZE", 1)) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    process_group = dist.group.WORLD
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

        position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]], device="cuda")
        setup_cp_attention_params(
            position_ids,
            process_group,
            seq_lens=torch.tensor([5, 3], device="cuda"),
            cp_style=cp_style,
        )
        if cp_style == "ring":
            substitute_gpt_oss_ring_attention(process_group, heads_k_stride=1)
        else:
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
        dist.destroy_process_group()
