"""CPU-only checks for the DSA bootstrap mechanism (converting a dense checkpoint, which has
no `indexer.*` keys, into a DSA-capable model class) -- mirrors LoRA's strip/reinit pattern,
see `strip_indexer_from_state_dict`'s and `Indexer._init_indexer_parameters`'s docstrings.
"""

import torch
from torch import nn

from prime_rl.trainer.model import freeze_all_except_indexer
from prime_rl.trainer.models.layers.dsa import Indexer, SparseMlaAttentionArgs, strip_indexer_from_state_dict


def test_strip_indexer_from_state_dict_removes_only_indexer_keys():
    state_dict = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.zeros(1),
        "model.layers.0.self_attn.indexer.wq_b.weight": torch.zeros(1),
        "model.layers.0.self_attn.indexer.wk.weight": torch.zeros(1),
        "model.layers.0.self_attn.indexer.weights_proj.weight": torch.zeros(1),
        "model.layers.0.self_attn.indexer.k_norm.weight": torch.zeros(1),
        "model.layers.1.mlp.experts.w1": torch.zeros(1),
    }
    stripped = strip_indexer_from_state_dict(state_dict)
    assert set(stripped.keys()) == {
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.1.mlp.experts.w1",
    }


def test_strip_indexer_from_state_dict_is_a_no_op_without_indexer_keys():
    state_dict = {"model.layers.0.self_attn.q_a_proj.weight": torch.zeros(1)}
    assert strip_indexer_from_state_dict(state_dict) == state_dict


def _tiny_indexer() -> Indexer:
    args = SparseMlaAttentionArgs(
        hidden_size=32,
        num_attention_heads=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        qk_head_dim=16,
        v_head_dim=8,
        attention_bias=False,
        rms_norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=64,
    )
    return Indexer(args)


def test_init_indexer_parameters_overwrites_garbage_with_a_real_init():
    indexer = _tiny_indexer()
    # Simulate the meta->real materialization gap _init_indexer_parameters exists to fix:
    # a dcp_load that never wrote these params leaves them exactly as constructed above,
    # so corrupt them here to prove the method actually overwrites every parameter.
    for param in indexer.parameters():
        param.data.fill_(float("nan"))

    indexer._init_indexer_parameters(generator=torch.Generator().manual_seed(0))

    for name, param in indexer.named_parameters():
        assert torch.isfinite(param).all(), f"{name} was not (re)initialized"
    assert torch.equal(indexer.k_norm.weight, torch.ones_like(indexer.k_norm.weight))
    assert torch.equal(indexer.k_norm.bias, torch.zeros_like(indexer.k_norm.bias))


def test_init_indexer_parameters_is_deterministic_given_a_generator():
    indexer_a = _tiny_indexer()
    indexer_a._init_indexer_parameters(generator=torch.Generator().manual_seed(42))

    indexer_b = _tiny_indexer()
    indexer_b._init_indexer_parameters(generator=torch.Generator().manual_seed(42))

    assert torch.equal(indexer_a.wq_b.weight, indexer_b.wq_b.weight)


def test_freeze_all_except_indexer_freezes_everything_but_the_indexer():
    model = nn.Module()
    model.embed = nn.Linear(4, 4)
    model.self_attn = nn.Module()
    model.self_attn.indexer = _tiny_indexer()

    freeze_all_except_indexer(model)

    assert not model.embed.weight.requires_grad
    assert all(p.requires_grad for p in model.self_attn.indexer.parameters())


def test_freeze_all_except_indexer_raises_without_an_indexer():
    model = nn.Module()
    model.embed = nn.Linear(4, 4)

    try:
        freeze_all_except_indexer(model)
        raise AssertionError("expected freeze_all_except_indexer to raise")
    except ValueError:
        pass
