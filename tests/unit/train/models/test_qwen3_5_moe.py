import pytest
import torch
from transformers import Qwen3_5MoeForCausalLM as HFQwen3_5MoeForCausalLM

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as PrimeRLQwen3_5MoeForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    config = Qwen3_5MoeConfig(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        use_grouped_mm=False,
    )
    config._attn_implementation = "sdpa"
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3_5MoeForCausalLM._from_config(config)
        prime_model = PrimeRLQwen3_5MoeForCausalLM._from_config(config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_qwen3_5_moe():
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_qwen3_5_moe_roundtrip():
    """Verify HF → PrimeRL → HF weight conversion is lossless at the state_dict level."""
    hf_model, prime_model = get_model_pairs()

    # Get original HF state_dict and the PrimeRL-converted version
    original_hf_sd = hf_model.state_dict()
    prime_sd = prime_model.state_dict()

    # Convert PrimeRL → per-expert HF format
    converted_hf_sd = PrimeRLQwen3_5MoeForCausalLM.convert_to_hf(dict(prime_sd))

    # Also convert original HF (fused) to per-expert format for comparison

    # First convert original HF → PrimeRL, then back to per-expert HF
    orig_prime_sd = dict(original_hf_sd)
    PrimeRLQwen3_5MoeForCausalLM.convert_to_prime(orig_prime_sd)
    orig_roundtripped = dict(orig_prime_sd)
    PrimeRLQwen3_5MoeForCausalLM.convert_to_hf(orig_roundtripped)

    # All non-expert keys should match exactly, expert keys should match after roundtrip
    for key in orig_roundtripped:
        assert key in converted_hf_sd, f"Missing key: {key}"
        assert torch.equal(orig_roundtripped[key], converted_hf_sd[key]), f"Mismatch at {key}"


def test_qwen3_5_moe_router_replay():
    """When routed_experts are provided, the model uses them instead of computing routing."""
    _, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, prime_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    out_normal = prime_model(input_ids, position_ids=position_ids)

    num_layers = prime_model.config.num_hidden_layers
    topk = prime_model.config.num_experts_per_tok
    routed_experts = torch.randint(0, prime_model.config.num_experts, (1, 100, num_layers, topk), device="cuda")

    prime_model.zero_grad()
    out_replay = prime_model(input_ids, position_ids=position_ids, routed_experts=routed_experts)

    assert out_replay["logits"].shape == out_normal["logits"].shape

    out_replay["logits"].sum().backward()
    assert prime_model.model.embed_tokens.weight.grad is not None


def test_qwen3_5_moe_cp_patching():
    """Verify substitute_ring_attn patches Qwen3_5MoeGatedFlashAttention._compute_attention."""
    from unittest.mock import MagicMock

    from prime_rl.trainer.models.layers.attn import substitute_ring_attn
    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedFlashAttention

    original_method = Qwen3_5MoeGatedFlashAttention._compute_attention

    mock_group = MagicMock()
    substitute_ring_attn(process_group=mock_group, heads_k_stride=1)

    assert Qwen3_5MoeGatedFlashAttention._compute_attention is not original_method

    # Restore to avoid polluting other tests
    Qwen3_5MoeGatedFlashAttention._compute_attention = original_method


if __name__ == "__main__":
    test_qwen3_5_moe()
