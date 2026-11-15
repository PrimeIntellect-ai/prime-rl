import pytest
import torch
from torch import nn
from transformers import Glm4MoeForCausalLM as HFGlm4MoeForCausalLM

from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.parity import assert_matches_hf, hf_checkpoint_state_dict

pytestmark = [pytest.mark.gpu]

TOL = 2e-2


def get_model_pairs(tmp_path) -> tuple[HFGlm4MoeForCausalLM, PrimeRLGlm4MoeForCausalLM]:
    torch.manual_seed(0)
    config_kwargs = dict(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        max_position_embeddings=512,
        moe_intermediate_size=128,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hidden_layers=3,
        rope_theta=1000000.0,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
    )
    hf_config = Glm4MoeConfig(**config_kwargs, n_group=1, topk_group=1)
    prime_config = Glm4MoeConfig(**config_kwargs)
    hf_config._attn_implementation = "flash_attention_2"
    prime_config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_model = HFGlm4MoeForCausalLM._from_config(hf_config)
        prime_model = PrimeRLGlm4MoeForCausalLM._from_config(prime_config)
    with torch.no_grad():
        state_dict = hf_checkpoint_state_dict(hf_model, tmp_path)
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    # Training code wraps the LM head; tests should mirror that (so forward can accept labels/temperature).
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


class _IdentityMLP(nn.Identity):
    def forward(self, x, **kwargs):
        return super().forward(x)


def test_glm4_moe_attn_only(tmp_path) -> None:
    hf_model, prime_model = get_model_pairs(tmp_path)
    for layer in hf_model.model.layers:
        layer.mlp = nn.Identity()
    for layer in prime_model.model.layers:
        layer.mlp = _IdentityMLP()
    assert_matches_hf(hf_model, prime_model, TOL)


def test_glm4_moe_mlp_only(tmp_path) -> None:
    hf_model, prime_model = get_model_pairs(tmp_path)

    def foo(hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = foo
    for layer in prime_model.model.layers:
        layer.self_attn.forward = foo
    assert_matches_hf(hf_model, prime_model, TOL)


def test_glm4_moe(tmp_path) -> None:
    hf_model, prime_model = get_model_pairs(tmp_path)
    assert_matches_hf(hf_model, prime_model, TOL)
