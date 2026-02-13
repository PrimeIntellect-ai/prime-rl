import pytest
import torch
from torch import nn
from transformers import GlmMoeDsaForCausalLM as HFGlmMoeDsaForCausalLM

from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaForCausalLM as PrimeRLGlmMoeDsaForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs() -> tuple[HFGlmMoeDsaForCausalLM, PrimeRLGlmMoeDsaForCausalLM]:
    config = GlmMoeDsaConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        v_head_dim=64,
        qk_nope_head_dim=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        max_position_embeddings=4096,
        rope_interleave=True,
        use_grouped_mm=False,
    )
    config._attn_implementation = "sdpa"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFGlmMoeDsaForCausalLM._from_config(config)
        prime_model = PrimeRLGlmMoeDsaForCausalLM._from_config(config)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_glm_moe_dsa_attn_only() -> None:
    hf_model, prime_model = get_model_pairs()

    # Replace MLPs with identity to isolate attention
    for layer in hf_model.model.layers:
        layer.mlp = nn.Identity()
    for layer in prime_model.model.layers:
        layer.mlp = nn.Identity()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_glm_moe_dsa_mlp_only() -> None:
    hf_model, prime_model = get_model_pairs()

    # Replace attention with identity function to isolate MLP/MoE
    def hf_attn_identity(hidden_states, *args, **kwargs):
        return hidden_states, None

    def prime_attn_identity(hidden_states, *args, **kwargs):
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = hf_attn_identity
    for layer in prime_model.model.layers:
        layer.self_attn.forward = prime_attn_identity

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_glm_moe_dsa() -> None:
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"

    # HF -> PrimeRL -> HF roundtrip
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_from_prime_model = HFGlmMoeDsaForCausalLM._from_config(hf_model.config)
        converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
        hf_from_prime_model.load_state_dict(converted_state_dict)

    hf_from_prime_output = hf_from_prime_model(input_ids=input_ids, position_ids=position_ids)
    hf_from_prime_output.logits.sum().backward()

    logits_diff = hf_from_prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_from_prime_model.model.embed_tokens.weight.grad - hf_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


if __name__ == "__main__":
    test_glm_moe_dsa_mlp_only()
