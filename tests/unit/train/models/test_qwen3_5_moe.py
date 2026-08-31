import pytest
import torch
from transformers import Qwen3_5MoeForCausalLM as HFQwen3_5MoeForCausalLM

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeTextConfig
from prime_rl.utils.cp import setup_model_cp
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    config = Qwen3_5MoeTextConfig(
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
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_model = HFQwen3_5MoeForCausalLM._from_config(config)
        prime_model = Qwen3_5ForCausalLM._from_config(config)
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

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(
        input_ids,
        position_ids=position_ids,
        seq_lens=torch.tensor([input_ids.shape[1]], device="cuda"),
    )
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=1e-0), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=1000), f"Max grad diff: {grad_diff.abs().max()}"

    packed_position_ids = torch.arange(1, 51, device="cuda").repeat(2).unsqueeze(0)
    with torch.no_grad():
        packed = prime_model(
            input_ids,
            position_ids=packed_position_ids,
            seq_lens=torch.tensor([50, 50], device="cuda"),
        )["logits"]
        unpacked = torch.cat(
            [
                prime_model(
                    input_ids[:, start : start + 50],
                    position_ids=packed_position_ids[:, :50],
                    seq_lens=torch.tensor([50], device="cuda"),
                )["logits"]
                for start in (0, 50)
            ],
            dim=1,
        )
    torch.testing.assert_close(packed, unpacked, atol=0.03, rtol=0.01)


def test_qwen3_5_moe_roundtrip():
    hf_model, prime_model = get_model_pairs()

    # Get original HF state_dict and the PrimeRL-converted version
    original_hf_sd = hf_model.state_dict()
    prime_sd = prime_model.state_dict()
    assert prime_model.is_hf_state_dict(original_hf_sd)
    assert not prime_model.is_prime_state_dict(original_hf_sd)
    assert prime_model.is_prime_state_dict(prime_sd)
    assert not prime_model.is_hf_state_dict(prime_sd)

    converted_hf_sd = prime_model.convert_to_hf(dict(prime_sd))
    orig_prime_sd = dict(original_hf_sd)
    prime_model.convert_to_prime(orig_prime_sd)
    orig_roundtripped = dict(orig_prime_sd)
    prime_model.convert_to_hf(orig_roundtripped)

    for key in orig_roundtripped:
        assert key in converted_hf_sd, f"Missing key: {key}"
        assert torch.equal(orig_roundtripped[key], converted_hf_sd[key]), f"Mismatch at {key}"


def test_qwen3_5_moe_router_replay():
    """When routed_experts are provided, the model uses them instead of computing routing."""
    _, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        input_ids = torch.randint(0, prime_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    seq_lens = torch.tensor([input_ids.shape[1]], device="cuda")
    out_normal = prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)

    num_layers = prime_model.config.num_hidden_layers
    topk = prime_model.config.num_experts_per_tok
    routed_experts = torch.randint(0, prime_model.config.num_experts, (1, 100, num_layers, topk), device="cuda")

    prime_model.zero_grad()
    out_replay = prime_model(
        input_ids,
        position_ids=position_ids,
        routed_experts=routed_experts,
        seq_lens=seq_lens,
    )

    assert out_replay["logits"].shape == out_normal["logits"].shape

    out_replay["logits"].sum().backward()
    assert prime_model.model.embed_tokens.weight.grad is not None


def test_qwen3_5_moe_cp_patching():
    from unittest.mock import MagicMock

    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention
    from prime_rl.trainer.models.layers.attn import FlashAttention, substitute_ring_attn
    from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention

    originals = {cls: cls._compute_attention for cls in (FlashAttention, AfmoeFlashAttention)}
    try:
        mock_group = MagicMock()
        substitute_ring_attn(process_group=mock_group, heads_k_stride=1)
        assert Qwen3_5Attention._compute_attention is FlashAttention._compute_attention
        assert Qwen3_5Attention._compute_attention is not originals[FlashAttention]
    finally:
        for cls, method in originals.items():
            cls._compute_attention = method


def test_qwen3_5_moe_context_parallel_setup_hook():
    from unittest.mock import MagicMock

    config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

    linear_layer = model.model.layers[0]
    model.model.layers[0] = torch.nn.Sequential(linear_layer)
    cp_group = MagicMock()
    setup_model_cp(model, cp_group, cp_rank=1, cp_world_size=2)

    assert model.model.context_parallel_group is cp_group
    assert model.model.context_parallel_rank == 1
    assert model.model.context_parallel_world_size == 2
    assert linear_layer.linear_attn.context_parallel_group is cp_group
    assert linear_layer.linear_attn.context_parallel_world_size == 2


if __name__ == "__main__":
    test_qwen3_5_moe()
