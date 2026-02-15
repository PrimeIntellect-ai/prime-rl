import pytest
import torch
from torch import nn
from transformers import GptOssForCausalLM as HFGptOssForCausalLM

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    hf_config = GptOssConfig(
        vocab_size=256,
        head_dim=64,
        max_position_embeddings=512,
        hidden_size=256,
        moe_intermediate_size=128,
        intermediate_size=128,
        norm_topk_prob=True,
        num_attention_heads=16,
        num_experts=4,
        num_experts_per_tok=2,
        num_key_value_heads=2,
        num_hidden_layers=4,
        rope_theta=10000.0,
        mlp_only_layers=[1],
        use_grouped_mm=False,
        initializer_range=0.02,
    )
    hf_config._attn_implementation = "eager"
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFGptOssForCausalLM._from_config(hf_config)
        prime_model = PrimeRLGptOssForCausalLM._from_config(hf_config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


class IdentityMLP(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):
        batch_seq = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        router_logits = torch.zeros(batch_seq, self.num_experts, device=x.device, dtype=x.dtype)
        return x, router_logits


def test_gpt_oss_attn_only():
    hf_model, prime_model = get_model_pairs()
    for layer in hf_model.model.layers:
        layer.mlp = IdentityMLP()
    for layer in prime_model.model.layers:
        layer.mlp = IdentityMLP()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()
    print(prime_output)
    print(hf_output)

    logits_diff = prime_output["logits"] - hf_output.logits
    print(logits_diff.abs().max())
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"

def test_gpt_oss_mlp_only():
    hf_model, prime_model = get_model_pairs()

    def identity_attn(hidden_states: torch.Tensor, *args, **kwargs):
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = identity_attn
    for layer in prime_model.model.layers:
        layer.self_attn.forward = identity_attn

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    print(logits_diff.abs().max())
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_gpt_oss():
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    print(logits_diff.abs().max())
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_from_prime_model = HFGptOssForCausalLM._from_config(hf_model.config)
        converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
        hf_from_prime_model.load_state_dict(converted_state_dict)

    hf_from_prime_output = hf_from_prime_model(input_ids, position_ids)
    hf_from_prime_output.logits.sum().backward()

    logits_diff = hf_from_prime_output.logits - hf_output.logits
    print(logits_diff.abs().max())
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_from_prime_model.model.embed_tokens.weight.grad - hf_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"
