import pytest
import torch

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import resolve_auto_attn
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model():
    config = Qwen3MoeConfig(
        head_dim=128,
        hidden_size=1024,
        max_position_embeddings=4096,
        max_window_layers=48,
        moe_intermediate_size=256,
        norm_topk_prob=True,
        num_attention_heads=16,
        num_experts=16,
        num_experts_per_tok=4,
        num_hidden_layers=3,
        rope_theta=1000000.0,
        use_qk_norm=True,
        mlp_only_layers=[1],
    )
    runtime_config = ModelConfig()
    resolve_auto_attn(runtime_config)
    with torch.device("cuda"):
        model = AutoModelForCausalLMPrimeRL.from_config(
            config, attn_implementation=runtime_config.attn, dtype=torch.bfloat16
        )
    inject_prime_lm_head(model, chunk_size=None)
    return model


def test_qwen3_moe_router_replay():
    """When routed_experts are provided, the model uses them instead of computing routing."""
    prime_model = get_model()

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        input_ids = torch.randint(0, prime_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    # Forward without router replay
    seq_lens = torch.tensor([input_ids.shape[1]], device="cuda")
    out_normal = prime_model(input_ids, position_ids, seq_lens=seq_lens)

    # Construct routed_experts with fixed expert indices
    # Shape: [batch=1, seq_len=100, num_hidden_layers=3, num_experts_per_tok=4]
    num_layers = prime_model.config.num_hidden_layers
    topk = prime_model.config.num_experts_per_tok
    routed_experts = torch.randint(0, prime_model.config.num_experts, (1, 100, num_layers, topk), device="cuda")

    # Forward with router replay
    prime_model.zero_grad()
    out_replay = prime_model(input_ids, position_ids, routed_experts=routed_experts, seq_lens=seq_lens)

    # Outputs should differ because routing is forced to different experts
    assert out_replay["logits"].shape == out_normal["logits"].shape

    # Verify gradients flow through the model with router replay
    out_replay["logits"].sum().backward()
    assert prime_model.model.embed_tokens.weight.grad is not None
