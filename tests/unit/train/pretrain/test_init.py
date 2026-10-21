import itertools

import torch

from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.models.init import normal_shard
from prime_rl.trainer.models.prime_moe import PrimeMoeConfig, PrimeMoeForCausalLM
from prime_rl.trainer.pretrain.parametrization import param_groups


def test_normal_shard_is_independent_of_sharding():
    shape = (37, 5, 3)
    full = normal_shard("w", shape, shape, (0, 0, 0), seed=0, device=torch.device("cpu"))
    for parts in (2, 3, 8):
        bounds = [len(chunk) for chunk in torch.arange(shape[0]).chunk(parts)]
        offsets = [0, *itertools.accumulate(bounds)][:-1]
        shards = [
            normal_shard("w", shape, (rows, *shape[1:]), (offset, 0, 0), seed=0, device=torch.device("cpu"))
            for rows, offset in zip(bounds, offsets)
        ]
        assert torch.equal(torch.cat(shards), full)
    column_shard = normal_shard("w", shape, (37, 2, 3), (0, 3, 0), seed=0, device=torch.device("cpu"))
    assert torch.equal(column_shard, full[:, 3:5])
    assert not torch.equal(normal_shard("w", shape, shape, (0, 0, 0), seed=1, device=torch.device("cpu")), full)


def test_moe_param_groups_cover_every_parameter():
    config = PrimeMoeConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hidden_layers=2,
        first_k_dense_replace=1,
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("meta"):
        model = PrimeMoeForCausalLM._from_config(config)
    groups = param_groups(model, MuonConfig(lr=1.0, weight_decay=0.1), ep_enabled=False)
    group_of = {param: group for group in groups for param in group["params"]}
    assert sum(len(group["params"]) for group in groups) == len(group_of) == len(list(model.parameters()))
    moe = model.model.layers[1].mlp
    assert group_of[moe.experts.up_proj]["algorithm"] == group_of[moe.router.gate.weight]["algorithm"] == "muon"
    assert group_of[model.lm_head.weight]["algorithm"] == "adamw"
    assert group_of[model.model.norm.weight]["weight_decay"] == 0.0
    assert all(group["lr"] == 1.0 for group in groups)
