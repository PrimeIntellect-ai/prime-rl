import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.models.fusions import apply_model_fusions, get_model_packed_parameters
from prime_rl.trainer.models.layers.ngram import NGramTable
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextForCausalLM, Qwen3_8FlashNextTextConfig
from prime_rl.trainer.optim import _create_muon_optimizer, muon_matrix_partitions, muon_param_groups
from prime_rl.trainer.optim.muon import NEWTON_SCHULZ, Muon, newton_schulz


@pytest.mark.parametrize("schedule, tolerance", [("deepseek_v4", 0.02), ("polar_express", 0.25), ("step3p5", 0.02)])
def test_orthogonalization_singular_values(schedule, tolerance):
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 96)]:
        X = newton_schulz(torch.randn(1, *shape), NEWTON_SCHULZ[schedule], 1e-7)
        assert (torch.linalg.svdvals(X[0].float()) - 1).abs().max() < tolerance


def reference_sinkhorn(W, grads, lr, momentum, update_rms, tau, epsilon, iters):
    """DeepSeek-V4.1 Algorithm 1, written out on the whole matrix in float64."""
    M = torch.zeros_like(W)
    for G in grads:
        M = momentum * M + (1 - momentum) * G
        U = momentum * M + (1 - momentum) * G
        rho = U.norm(dim=1)
        U = torch.where((rho <= tau * rho.mean()).unsqueeze(1), 0.0, U)
        for k in range(1, iters + 1):
            U = U / (U.norm(dim=1 if k % 2 else 0, keepdim=True) + epsilon)
        W = W - lr * update_rms * math.sqrt(W.shape[1]) * U
    return W


def run_sinkhorn(params, grads):
    group = dict(params=params, algorithm="sinkhorn", weight_decay=0.0, lr_scale=1.0, tau=1e-3, iters=11)
    optimizer = Muon([group], lr=0.01, mu=0.9, epsilon=1e-8)
    for step_grads in grads:
        for p, grad in zip(params, step_grads):
            p.grad = grad
        optimizer.step()
    return optimizer


def test_sinkhorn_matches_reference():
    torch.manual_seed(0)
    W, grads = torch.randn(64, 16), torch.randn(3, 64, 16)
    grads[0, 5] *= 1e-6  # a near-zero row is masked
    param = nn.Parameter(W.clone())
    run_sinkhorn([param], grads[:, None])
    expected = reference_sinkhorn(W.double(), grads.double(), 0.01, 0.9, 0.2, 1e-3, 1e-8, 11)
    torch.testing.assert_close(param.detach(), expected.float(), rtol=1e-4, atol=1e-6)


def sharded_sinkhorn(rank, world_size, init_file, weights, grads, shard_dim):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    mesh = init_device_mesh("cpu", (world_size,))
    expected = [nn.Parameter(w.clone()) for w in weights]
    sharded = [nn.Parameter(distribute_tensor(w.clone(), mesh, [Shard(shard_dim)])) for w in weights]
    sharded_grads = [[distribute_tensor(g, mesh, [Shard(shard_dim)]) for g in step] for step in grads]
    states = [run_sinkhorn(expected, grads).state, run_sinkhorn(sharded, sharded_grads).state]
    for actual, full in zip(sharded, expected):
        torch.testing.assert_close(actual.full_tensor(), full.detach(), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(states[1][actual]["momentum"].full_tensor(), states[0][full]["momentum"])
    dist.destroy_process_group()


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_sharded_sinkhorn_matches_unsharded(tmp_path, shard_dim):
    """Uneven row (or column) shards."""
    torch.manual_seed(0)
    weights = [torch.randn(25, 13), torch.randn(13, 25)]
    grads = [[torch.randn_like(w) for w in weights] for _ in range(3)]
    mp.spawn(sharded_sinkhorn, args=(2, tmp_path / "init", weights, grads, shard_dim), nprocs=2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.q_proj = nn.Linear(8, 8, bias=True)
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=8, eps=1e-6))
        self.sinks = nn.Parameter(torch.zeros(2))
        self.lm_head = nn.Linear(8, 32, bias=False)


def test_muon_param_groups_route_dsv41_recipe():
    model = TinyModel()
    named_params = list(model.named_parameters())

    def groups(**config):
        optimizer = _create_muon_optimizer(MuonConfig(weight_decay=0.1, **config), named_params, model, lr=1.0)
        return [
            (g["algorithm"], g["lr"], g["weight_decay"], [p.shape for p in g["params"]]) for g in optimizer.param_groups
        ]

    adamw_shapes = [(2,), (32, 8), (8,), (8,), (32, 8)]
    assert groups() == [("muon", 1.0, 0.1, [(8, 8)]), ("adamw", 1.0, 0.1, adamw_shapes)]
    recipe = groups(update_rms=0.18, decay_bias_and_scales=False, embeddings={})
    assert recipe == [
        ("muon", 1.0, 0.1, [(8, 8)]),
        ("adamw", 1.0, 0.1, [(8,)]),  # the norm weight
        ("adamw", 1.0, 0.0, [(2,), (8,)]),  # the sinks and the bias
        ("sinkhorn", 1.0, 0.0, [(32, 8), (32, 8)]),
    ]


def test_update_scales_keep_the_group_lr():
    def step(update_rms, lr_scale):
        torch.manual_seed(0)
        weight, embedding = nn.Parameter(torch.randn(16, 8)), nn.Parameter(torch.randn(32, 8))
        groups = [
            dict(params=[weight], algorithm="muon", weight_decay=0.1),
            dict(params=[embedding], algorithm="sinkhorn", weight_decay=0.0, lr_scale=lr_scale, tau=1e-3, iters=11),
        ]
        optimizer = Muon(groups, lr=0.1, update_rms=update_rms)
        before = weight.detach().clone(), embedding.detach().clone()
        weight.grad, embedding.grad = torch.randn(16, 8), torch.randn(32, 8)
        optimizer.step()
        assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.1]
        return before[0], weight.detach() - before[0], embedding.detach() - before[1]

    weight, weight_delta, embedding_delta = step(0.2, 1.0)
    _, scaled_weight_delta, scaled_embedding_delta = step(0.1, 5.0)
    decay = -0.1 * 0.1 * weight  # lr * weight_decay does not depend on update_rms
    torch.testing.assert_close(scaled_weight_delta - decay, 0.5 * (weight_delta - decay))
    torch.testing.assert_close(scaled_embedding_delta, 2.5 * embedding_delta)


QWEN3_8 = dict(
    vocab_size=32,
    bos_token_id=31,
    eos_token_id=31,
    hidden_size=16,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=1,
    linear_num_value_heads=1,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    hc_count=2,
    hc_lowrank=4,
    num_experts=2,
    num_experts_per_tok=1,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    ple_layer_ids=[1],
    ple_embed_dim=16,
    heads_per_ngram=2,
    ngram_vocab_size_base=17,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=3,
)


def build_on_meta(model_type):
    with torch.device("meta"):
        if model_type == "qwen3_8":
            model = Qwen3_8FlashNextForCausalLM(Qwen3_8FlashNextTextConfig(**QWEN3_8))
        else:
            kwargs = dict(dsa_backend="tilelang") if model_type == "deepseek_v4" else {}
            model = AutoModelForCausalLMPrimeRL.from_config(CONFIG_MAPPING[model_type](**kwargs))
    apply_model_fusions(model, ["qkv", "gate_up"])
    return model


@pytest.mark.parametrize(
    "model_type", ["qwen3", "qwen3_5_text", "glm5_next_text", "qwen3_8", "nemotron_h", "deepseek_v4", "gpt_oss"]
)
def test_default_muon_routing_and_norms(model_type):
    """By default Muon takes every trainable matrix except embeddings, the LM head, n-gram tables and expert biases
    (depthwise convolutions and grouped linears included), and splits only runtime-packed parameters.
    Without decay for scales, every normalization weight is still decayed."""
    model = build_on_meta(model_type)
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ngram = {p for module in model.modules() if isinstance(module, NGramTable) for p in module.parameters()}
    groups = muon_param_groups(MuonConfig(), named_params, model)
    muon = {p for group in groups if group["algorithm"] == "muon" for p in group["params"]}
    excluded = ("lm_head", "embed_tokens")
    assert muon == {
        p
        for n, p in named_params
        if p.ndim >= 2
        and not any(e in n for e in excluded)
        and p not in ngram
        and not ("mlp.experts" in n and n.endswith("_bias"))
    }
    packed = {
        info.parameter: info.spec.muon_matrix_partitions(info.parameter) for info in get_model_packed_parameters(model)
    }
    assert muon_matrix_partitions(model, groups, None) == {
        p: s for p, s in packed.items() if s is not None and p in muon
    }

    groups = muon_param_groups(MuonConfig(decay_bias_and_scales=False), named_params, model)
    undecayed = {p for group in groups if group.get("weight_decay") == 0.0 for p in group["params"]}
    norm_weights = [p for n, p in named_params if p.ndim == 1 and "norm" in n.split(".")[-2]]
    assert norm_weights and not any(p in undecayed for p in norm_weights)
