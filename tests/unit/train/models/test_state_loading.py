from unittest.mock import MagicMock

import pytest
import torch

from prime_rl.configs.trainer import DebugModelConfig, ModelConfig
from prime_rl.trainer.model import load_dcp_from_hf
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def model() -> LagunaForCausalLM:
    config = LagunaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
    )
    with torch.device("meta"):
        return LagunaForCausalLM(config)


def test_load_dcp_from_hf_keeps_checkpoint_expert_bias(model, tmp_path, monkeypatch):
    """Checkpoint values for the persistent `expert_bias` buffer must survive loading."""
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4])

    def fake_dcp_load(state_dict, storage_reader=None):
        buffer = state_dict["model.layers.1.mlp.expert_bias"]
        buffer.copy_(expected.to(device=buffer.device, dtype=buffer.dtype))

    monkeypatch.setattr("prime_rl.trainer.model.dcp_load", fake_dcp_load)
    monkeypatch.setattr("prime_rl.trainer.model.load_state_dict_keys", lambda path: model.state_dict().keys())
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)

    load_dcp_from_hf(model, ModelConfig(name=str(tmp_path)), parallel_dims=MagicMock())

    expert_bias = model.model.layers[1].mlp.expert_bias
    torch.testing.assert_close(expert_bias.cpu(), expected.to(expert_bias.dtype))


@pytest.fixture
def glm4_moe_model() -> Glm4MoeForCausalLM:
    config = Glm4MoeConfig(
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=128,
        moe_intermediate_size=32,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hidden_layers=2,
        rope_theta=1000000.0,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
        use_grouped_mm=False,
        vocab_size=64,
    )
    with torch.device("meta"):
        return Glm4MoeForCausalLM(config)


def test_load_dcp_from_hf_random_init_zeros_expert_bias(glm4_moe_model, tmp_path, monkeypatch):
    """Under debug.random_init=True, dcp_load is skipped entirely, so init_buffers_post_meta is the
    only thing that can overwrite to_empty()'s undefined expert_bias contents. Regression test for
    glm4_moe, which (unlike laguna) never zeroed expert_bias before this fix."""
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)

    real_to_empty = glm4_moe_model.to_empty

    def poisoning_to_empty(*, device):
        real_to_empty(device=device)
        with torch.no_grad():
            for layer in glm4_moe_model.model.layers:
                if getattr(layer.mlp, "expert_bias", None) is not None:
                    layer.mlp.expert_bias.fill_(1.0)
        return glm4_moe_model

    monkeypatch.setattr(glm4_moe_model, "to_empty", poisoning_to_empty)

    config = ModelConfig(name=str(tmp_path), debug=DebugModelConfig(random_init=True))
    load_dcp_from_hf(glm4_moe_model, config, parallel_dims=MagicMock())

    for layer in glm4_moe_model.model.layers:
        if getattr(layer.mlp, "expert_bias", None) is not None:
            torch.testing.assert_close(layer.mlp.expert_bias.cpu(), torch.zeros_like(layer.mlp.expert_bias.cpu()))
