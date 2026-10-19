import copy
import itertools

import torch
from torch.optim import AdamW
from transformers import LlamaConfig, LlamaForCausalLM

from prime_rl.configs.pretrain import PretrainAdamWConfig, PretrainMuonConfig
from prime_rl.configs.trainer import InitConfig
from prime_rl.trainer.models.init import init_weights, normal_shard
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


def test_completep_folded_multipliers_match_explicit_multipliers():
    """CompleteP folds its 1/m_L residual and 1/m_N logit multipliers into the weights. Training
    the folded model must match the multiplier form of CompleteP (Dey et al. 2025, Table 1)."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        tie_word_embeddings=False,
    )
    optim = PretrainAdamWConfig(lr=1e-2, eps=1e-10, weight_decay=0.1)
    m_n, m_l = 2.0, 2.0

    folded = LlamaForCausalLM(config)
    init_weights(folded, InitConfig(), m_n, m_l)
    explicit = copy.deepcopy(folded)
    with torch.no_grad():
        for name, param in explicit.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                param.mul_(m_l)
            elif "lm_head" in name:
                param.mul_(m_n)
    for layer in explicit.model.layers:
        layer.self_attn.o_proj.register_forward_hook(lambda module, args, out: out / m_l)
        layer.mlp.down_proj.register_forward_hook(lambda module, args, out: out / m_l)
    explicit.lm_head.register_forward_pre_hook(lambda module, args: (args[0] / m_n,))

    def explicit_scales(name: str) -> tuple[float, float, float]:
        if "embed_tokens" in name or "lm_head" in name:
            return 1.0, 1 / m_n, 1.0
        if "norm" in name:
            return 1.0, 1 / m_n, 0.0
        return 1 / m_n, 1 / (m_n * m_l), m_n

    explicit_groups = []
    for name, param in explicit.named_parameters():
        lr, eps, wd = explicit_scales(name)
        explicit_groups.append(
            dict(params=[param], lr=optim.lr * lr, eps=optim.eps * eps, weight_decay=optim.weight_decay * wd)
        )
    optimizers = {
        "explicit": AdamW(explicit_groups, betas=(0.9, 0.95)),
        "folded": AdamW(param_groups(folded, optim, m_n, m_l, ep_enabled=False), betas=(0.9, 0.95)),
    }
    models = {"explicit": explicit, "folded": folded}

    tokens = torch.randint(0, 128, (4, 16))
    for _ in range(5):
        for key, model in models.items():
            model(input_ids=tokens, labels=tokens).loss.backward()
            optimizers[key].step()
            optimizers[key].zero_grad()
    with torch.no_grad():
        torch.testing.assert_close(
            folded(input_ids=tokens).logits, explicit(input_ids=tokens).logits, rtol=1e-4, atol=1e-4
        )


def test_completep_moe_groups_cover_experts_and_scale_the_router():
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
    groups = param_groups(model, PretrainMuonConfig(lr=1.0, weight_decay=0.1), m_n=2.0, m_l=1.0, ep_enabled=False)
    lr = {param: group["lr"] for group in groups for param in group["params"]}
    assert sum(len(group["params"]) for group in groups) == len(lr) == len(list(model.parameters()))
    moe = model.model.layers[1].mlp
    # m_N = 2: Muon matrices get lr / sqrt(m_N) (residual outputs also / m_L = 1), the router lr / m_N.
    assert lr[moe.experts.up_proj] == lr[moe.experts.down_proj] == 2**-0.5
    assert lr[moe.router.gate.weight] == 0.5
