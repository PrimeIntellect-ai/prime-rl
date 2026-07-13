import inspect
from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM

from prime_rl.trainer.models.layers.attn import FlashAttention, substitute_ring_attn
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5Model
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedFlashAttention
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig


def _tiny_text_config(attn_impl: str = "sdpa") -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
    )
    config._attn_implementation = attn_impl
    return config


def _tiny_moe_config(attn_impl: str = "sdpa") -> Qwen3_5MoeConfig:
    config = Qwen3_5MoeConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        use_grouped_mm=False,
    )
    config._attn_implementation = attn_impl
    return config


def test_qwen3_5_dense_matches_hf_state_keys_on_meta():
    config = _tiny_text_config()
    with torch.device("meta"):
        hf_model = HFQwen3_5ForCausalLM(config)
        prime_model = Qwen3_5ForCausalLM(config)

    assert set(prime_model.state_dict()) == set(hf_model.state_dict())
    for name, tensor in prime_model.state_dict().items():
        assert tensor.shape == hf_model.state_dict()[name].shape, name


@pytest.mark.parametrize("attn_impl", ["flash_attention_3", "kernels-community/vllm-flash-attn3"])
def test_qwen3_5_full_attention_uses_custom_class(attn_impl: str):
    config = _tiny_text_config(attn_impl=attn_impl)
    with torch.device("meta"):
        model = Qwen3_5Model(config)

    assert isinstance(model.layers[1].self_attn, Qwen3_5GatedFlashAttention)
    assert model.config._attn_implementation == "flash_attention_3"
    assert "ALL_ATTENTION_FUNCTIONS" not in inspect.getsource(type(model.layers[1].self_attn).forward)


def test_qwen3_5_moe_full_attention_normalizes_fa3_hub_alias():
    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedFlashAttention,
        Qwen3_5MoeModel,
    )

    config = _tiny_moe_config(attn_impl="kernels-community/vllm-flash-attn3")
    with torch.device("meta"):
        model = Qwen3_5MoeModel(config)

    assert isinstance(model.layers[1].self_attn, Qwen3_5MoeGatedFlashAttention)
    assert model.config._attn_implementation == "flash_attention_3"


def test_qwen3_5_ring_patches_dense_flash_attention():
    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention
    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedFlashAttention

    originals = {
        cls: cls._compute_attention
        for cls in (FlashAttention, AfmoeFlashAttention, Qwen3_5MoeGatedFlashAttention, Qwen3_5GatedFlashAttention)
    }
    try:
        substitute_ring_attn(process_group=MagicMock(), heads_k_stride=1)
        assert Qwen3_5GatedFlashAttention._compute_attention is not originals[Qwen3_5GatedFlashAttention]
    finally:
        for cls, method in originals.items():
            cls._compute_attention = method
