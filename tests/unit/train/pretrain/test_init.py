import itertools
from types import SimpleNamespace

import torch

from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.models.init import normal_shard
from prime_rl.trainer.models.prime_moe import PrimeMoeConfig, PrimeMoeForCausalLM
from prime_rl.trainer.pretrain.optim import param_groups
from prime_rl.trainer.utils import count_parameters


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


def small_moe(**kwargs) -> PrimeMoeForCausalLM:
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
        **kwargs,
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("meta"):
        return PrimeMoeForCausalLM._from_config(config)


def test_count_parameters_moe():
    model = small_moe()
    counts = count_parameters(model)
    embedding = model.model.embed_tokens.weight.numel() + model.lm_head.weight.numel()
    experts = sum(param.numel() for param in model.model.layers[1].mlp.experts.parameters())
    assert counts["total"] == sum(param.numel() for param in model.parameters())
    assert counts["embedding"] == embedding
    assert counts["non_embedding"] == counts["total"] - embedding
    assert counts["active_non_embedding"] == counts["non_embedding"] - experts * 6 // 8


def test_gated_residual_projections_train_with_adamw():
    model = small_moe(hc_count=4, hc_lowrank=32)
    groups = param_groups(model, MuonConfig(lr=1.0), SimpleNamespace(ep_enabled=False))
    algorithm = {param: group["algorithm"] for group in groups for param in group["params"]}
    assert sum(len(group["params"]) for group in groups) == len(algorithm) == len(list(model.parameters()))
    hyper_connection = [param for name, param in model.named_parameters() if "hyper_connection" in name]
    assert hyper_connection and all(algorithm[param] == "adamw" for param in hyper_connection)
    assert algorithm[model.model.layers[0].self_attn.q_proj.weight] == "muon"
    assert param_groups(small_moe(), MuonConfig(lr=1.0), SimpleNamespace(ep_enabled=False)) is None
