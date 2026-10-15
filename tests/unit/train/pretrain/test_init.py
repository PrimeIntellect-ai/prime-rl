import copy
import itertools

import pytest
import torch
from torch.optim import AdamW
from transformers import LlamaConfig, LlamaForCausalLM

from prime_rl.configs.pretrain import PretrainAdamWConfig, PretrainMuonConfig
from prime_rl.configs.trainer import InitConfig
from prime_rl.trainer.models.fusions import fuse_qkv_projections, get_model_packed_parameters
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.init import init_weights, normal_shard
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.pretrain.hyperball import Hyperball
from prime_rl.trainer.pretrain.parametrization import head_block_partitions, param_groups


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
    config = Glm4MoeConfig(
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
        model = Glm4MoeForCausalLM._from_config(config)
    groups = param_groups(model, PretrainMuonConfig(lr=1.0, weight_decay=0.1), m_n=2.0, m_l=1.0, ep_enabled=False)
    lr = {param: group["lr"] for group in groups for param in group["params"]}
    assert sum(len(group["params"]) for group in groups) == len(lr) == len(list(model.parameters()))
    moe = model.model.layers[1].mlp
    # m_N = 2: Muon matrices get lr / sqrt(m_N) (residual outputs also / m_L = 1), the router lr / m_N.
    assert lr[moe.experts.up_proj] == lr[moe.experts.down_proj] == 2**-0.5
    assert lr[moe.router.gate.weight] == 0.5


@pytest.mark.parametrize("heads_per_block", [1, 2, None])
def test_muon_orthogonalizes_each_head_block_of_the_packed_qkv(heads_per_block):
    muon = pytest.importorskip("dion.muon")
    head_dim = 16
    attention = FlashAttention(
        AttentionConfig(
            hidden_size=128,
            head_dim=head_dim,
            num_attention_heads=4,
            num_key_value_heads=2,
            is_causal=True,
            attention_bias=False,
            use_qk_norm=False,
            rms_norm_eps=1e-6,
        )
    )
    fuse_qkv_projections(attention)
    weight = attention.qkv_proj.weight
    if heads_per_block is None:
        (packed_info,) = get_model_packed_parameters(attention)
        partitions = {weight: packed_info.spec.muon_matrix_partitions(weight)}
        blocks = (64, 32, 32)
    else:
        partitions = head_block_partitions(attention, heads_per_block)
        blocks = (heads_per_block * head_dim,) * (128 // (heads_per_block * head_dim))
    assert partitions[weight] == blocks

    weight.grad = torch.randn_like(weight)
    before = weight.detach().clone()
    optimizer = muon.Muon(
        [weight],
        fsdp_mesh_dim=0,
        world_mesh=None,
        lr=1.0,
        mu=0.0,
        weight_decay=0.0,
        adjust_lr=None,
        matrix_partitions=partitions,
    )
    optimizer.step()
    # With no momentum and no learning-rate adjustment, each block moves by its own orthogonalized gradient
    # (Muon casts the momentum to bfloat16 before Newton-Schulz).
    update = before - weight.detach()
    for block_update, block_grad in zip(update.split(blocks), weight.grad.split(blocks)):
        expected = muon.zeropower_via_newtonschulz5(block_grad.bfloat16()).float()
        torch.testing.assert_close(block_update, expected, atol=1e-6, rtol=0)


def test_hyperball_keeps_each_matrix_norm_and_steps_by_lr_times_it():
    torch.manual_seed(0)
    matrix, experts = torch.nn.Parameter(torch.randn(8, 16)), torch.nn.Parameter(torch.randn(4, 8, 16) * 0.1)
    head = torch.nn.Parameter(torch.randn(8, 16))
    optimizer = torch.optim.SGD([dict(params=[matrix, experts], lr=0.1, hyperball=True), dict(params=[head], lr=0.1)])
    Hyperball(optimizer)
    before = [p.detach().clone() for p in (matrix, experts, head)]
    for p in (matrix, experts, head):
        p.grad = torch.randn_like(p)
    optimizer.step()

    for p, w in zip((matrix, experts), before):
        norm = w.norm(dim=(-2, -1), keepdim=True)
        trial = w - 0.1 * norm * p.grad / p.grad.norm(dim=(-2, -1), keepdim=True)
        torch.testing.assert_close(p.detach(), trial * norm / trial.norm(dim=(-2, -1), keepdim=True))
        torch.testing.assert_close(p.detach().norm(dim=(-2, -1)), w.norm(dim=(-2, -1)))
    torch.testing.assert_close(head.detach(), before[2] - 0.1 * head.grad)
