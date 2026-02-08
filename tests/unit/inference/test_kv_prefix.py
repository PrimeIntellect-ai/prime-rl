import pytest
import torch
from torch import nn

from prime_rl.inference.vllm.kv_prefix import (
    apply_kv_prefix_state_dict,
    get_layer_kv_prefix,
    split_kv_prefix_state_dict,
)


class FakeAttention(nn.Module):
    def __init__(self, layer_name: str, num_kv_heads: int, head_size: int):
        super().__init__()
        self.layer_name = layer_name
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.register_buffer("_k_scale", torch.tensor(1.0))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FakeAttention("model.layers.0.self_attn.attn", num_kv_heads=2, head_size=4),
                FakeAttention("model.layers.1.self_attn.attn", num_kv_heads=2, head_size=4),
            ]
        )


class FakeNonFlashAttentionImpl:
    pass


FakeNonFlashAttentionImpl.__module__ = "vllm.v1.attention.backends.triton_attn"


def test_split_kv_prefix_state_dict_filters_kv_tensors():
    weights = [
        ("model.layers.0.self_attn.kv_prefix_key", torch.ones(2, 3, 4)),
        ("model.layers.0.self_attn.kv_prefix_value", torch.ones(2, 3, 4)),
        ("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4)),
    ]

    filtered_iter, kv_prefix_state = split_kv_prefix_state_dict(weights)
    filtered_weights = list(filtered_iter)

    assert len(kv_prefix_state) == 2
    assert len(filtered_weights) == 1
    assert filtered_weights[0][0] == "model.layers.0.self_attn.q_proj.weight"


def test_apply_kv_prefix_state_dict_applies_prefix_to_matching_layer():
    model = FakeModel()

    kv_prefix_state = {
        "model.layers.0.self_attn.kv_prefix_key": torch.arange(24, dtype=torch.float32).view(2, 3, 4),
        "model.layers.0.self_attn.kv_prefix_value": torch.arange(24, dtype=torch.float32).view(2, 3, 4) + 100,
    }

    applied = apply_kv_prefix_state_dict(model, kv_prefix_state)

    assert applied == 1
    layer = model.layers[0]
    prefix = get_layer_kv_prefix(layer)
    assert prefix is not None

    key, value, prefix_tokens = prefix
    assert prefix_tokens == 3
    assert key.shape == (3, 2, 4)
    assert value.shape == (3, 2, 4)
    assert torch.equal(key, kv_prefix_state["model.layers.0.self_attn.kv_prefix_key"].transpose(0, 1))
    assert torch.equal(value, kv_prefix_state["model.layers.0.self_attn.kv_prefix_value"].transpose(0, 1))

def test_apply_kv_prefix_state_dict_clears_existing_prefix_when_empty_update():
    model = FakeModel()
    kv_prefix_state = {
        "model.layers.0.self_attn.kv_prefix_key": torch.ones(2, 2, 4),
        "model.layers.0.self_attn.kv_prefix_value": torch.ones(2, 2, 4),
    }

    apply_kv_prefix_state_dict(model, kv_prefix_state)
    cleared_layers = apply_kv_prefix_state_dict(model, {})

    assert cleared_layers == 0
    assert get_layer_kv_prefix(model.layers[0]) is None

def test_apply_kv_prefix_state_dict_raises_for_unknown_layer():
    model = FakeModel()
    kv_prefix_state = {
        "model.layers.99.self_attn.kv_prefix_key": torch.ones(2, 2, 4),
        "model.layers.99.self_attn.kv_prefix_value": torch.ones(2, 2, 4),
    }

    with pytest.raises(ValueError, match="No matching vLLM attention layer"):
        apply_kv_prefix_state_dict(model, kv_prefix_state)


def test_apply_kv_prefix_state_dict_raises_for_non_flash_attention_backend():
    model = FakeModel()
    model.layers[0].impl = FakeNonFlashAttentionImpl()
    kv_prefix_state = {
        "model.layers.0.self_attn.kv_prefix_key": torch.ones(2, 2, 4),
        "model.layers.0.self_attn.kv_prefix_value": torch.ones(2, 2, 4),
    }

    with pytest.raises(ValueError, match="requires FlashAttention backend"):
        apply_kv_prefix_state_dict(model, kv_prefix_state)
