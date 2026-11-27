import pytest
import torch
import torch.nn.functional as F
from transformers import LagunaConfig as HFLagunaConfig
from transformers import LagunaForCausalLM as HFLagunaForCausalLM

from prime_rl.trainer.models.laguna import LagunaConfig, LagunaForCausalLM
from prime_rl.trainer.models.laguna.modeling_laguna import LagunaFlashAttention
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

CONFIG = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_attention_heads_per_layer=[4, 8],
    num_key_value_heads=2,
    head_dim=16,
    mlp_layer_types=["dense", "dense"],
    max_position_embeddings=128,
    rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
)


def _sdpa_attention(self, q, k, v, cu_seqlens, max_seqlen):
    """Flash attention needs a GPU; run the single causal document through SDPA instead."""
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True, enable_gqa=True
    )
    return out.transpose(0, 1)


@pytest.mark.parametrize(("gating", "per_head"), [(True, True), ("per-head", True), ("per-element", False)])
def test_laguna_gating_matches_hf(monkeypatch, gating, per_head):
    monkeypatch.setattr(LagunaFlashAttention, "_compute_attention", _sdpa_attention)
    torch.manual_seed(0)
    hf_model = HFLagunaForCausalLM(HFLagunaConfig(**CONFIG, gating=gating))
    with torch.no_grad():
        for layer in hf_model.model.layers:
            # Unit-scale g_proj weights spread the gates enough that a misplaced gate changes the output.
            layer.self_attn.g_proj.weight.normal_()
    prime_model = LagunaForCausalLM.from_config(
        LagunaConfig(**CONFIG, gating=gating), attn_implementation="flash_attention_2", dtype=torch.float32
    )

    for layer, num_heads in zip(prime_model.model.layers, CONFIG["num_attention_heads_per_layer"]):
        gate_size = num_heads if per_head else num_heads * CONFIG["head_dim"]
        assert layer.self_attn.g_proj.weight.shape == (gate_size, CONFIG["hidden_size"])
    prime_model.load_state_dict(prime_model.convert_to_prime(hf_model.state_dict()))
    inject_prime_lm_head(prime_model, chunk_size=None)

    input_ids = torch.randint(0, CONFIG["vocab_size"], (1, 32))
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        prime_logits = prime_model(input_ids, seq_lens=torch.tensor([input_ids.shape[1]]))["logits"]
    torch.testing.assert_close(prime_logits, hf_logits)
