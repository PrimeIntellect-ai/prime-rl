import pytest
import torch
from torch import nn
from transformers import Glm4MoeForCausalLM as HFGlm4MoeForCausalLM

from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs() -> tuple[HFGlm4MoeForCausalLM, PrimeRLGlm4MoeForCausalLM]:
    hf_config = Glm4MoeConfig(
        hidden_size=1024,
        intermediate_size=2048,
        max_position_embeddings=4096,
        moe_intermediate_size=256,
        norm_topk_prob=True,
        num_attention_heads=16,
        num_key_value_heads=4,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=1,
        num_hidden_layers=3,
        rope_theta=1000000.0,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
    )
    # TODO: We should test this path because it's the most performant
    # But the grad seems to be off in attn because of precision
    # hf_config._attn_implementation = "flash_attention_2"
    hf_config._attn_implementation = "sdpa"
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFGlm4MoeForCausalLM._from_config(hf_config)
        prime_model = PrimeRLGlm4MoeForCausalLM._from_config(hf_config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_glm4_moe_attn_only() -> None:
    hf_model, prime_model = get_model_pairs()
    for layer in hf_model.model.layers:
        layer.mlp = nn.Identity()
    for layer in prime_model.model.layers:
        layer.mlp = nn.Identity()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_glm4_moe_mlp_only() -> None:
    hf_model, prime_model = get_model_pairs()

    def foo(hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = foo
    for layer in prime_model.model.layers:
        layer.self_attn.forward = foo

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_glm4_moe() -> None:
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_from_prime_model = HFGlm4MoeForCausalLM._from_config(hf_model.config)
        converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
        hf_from_prime_model.load_state_dict(converted_state_dict)

    hf_from_prime_output = hf_from_prime_model(input_ids, position_ids)
    hf_from_prime_output.logits.sum().backward()

    logits_diff = hf_from_prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_from_prime_model.model.embed_tokens.weight.grad - hf_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


if __name__ == "__main__":
    test_glm4_moe_mlp_only()
