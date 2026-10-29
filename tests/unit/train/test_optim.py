import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.optim import muon_param_groups
from prime_rl.trainer.optim.muon import newton_schulz_deepseek_v4, sinkhorn_update

SINKHORN = dict(lr=0.1, momentum=0.9, update_rms=0.18, tau=1e-3, epsilon=1e-20, iters=11)


def reference_sinkhorn(W, G, M, lr, momentum, update_rms, tau, epsilon, iters):
    """DeepSeek-V4.1 Algorithm 1, written out on the whole matrix in float64."""
    M = momentum * M + (1 - momentum) * G
    U = momentum * M + (1 - momentum) * G
    rho = U.norm(dim=1)
    U = torch.where((rho <= tau * rho.mean()).unsqueeze(1), 0.0, U)
    for k in range(1, iters + 1):
        if k % 2 == 1:
            U = U / (U.norm(dim=1, keepdim=True) + epsilon)
        else:
            U = U / (U.norm(dim=0, keepdim=True) + epsilon)
    return W - lr * update_rms * math.sqrt(W.shape[1]) * U, M


def sharded_sinkhorn(rank, world_size, init_file, shard_dim, inputs, expected):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    W, G, M = (t.chunk(world_size, dim=shard_dim)[rank].clone() for t in inputs)
    groups = [dist.group.WORLD]
    sinkhorn_update(
        W,
        G,
        M,
        shape=tuple(inputs[0].shape),
        row_groups=groups if shard_dim == 1 else [],
        col_groups=groups if shard_dim == 0 else [],
        **SINKHORN,
    )
    for actual, full in zip((W, M), expected):
        torch.testing.assert_close(actual, full.chunk(world_size, dim=shard_dim)[rank].float(), rtol=1e-4, atol=1e-6)
    dist.destroy_process_group()


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_sharded_sinkhorn_update_matches_reference(tmp_path, shard_dim):
    torch.manual_seed(0)
    W, G, M = torch.randn(3, 64, 16)
    G[5] *= 1e-6  # a near-zero row is masked
    M[5] = 0
    expected = reference_sinkhorn(W.double(), G.double(), M.double(), **SINKHORN)
    assert expected[0][5].equal(W[5].double())
    mp.spawn(sharded_sinkhorn, args=(2, tmp_path / "init", shard_dim, (W, G, M), expected), nprocs=2)


def test_deepseek_v4_newton_schulz_pins_singular_values():
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 96)]:
        singular_values = torch.linalg.svdvals(newton_schulz_deepseek_v4(torch.randn(shape)).float())
        assert (singular_values - 1).abs().max() < 0.02


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
        result = muon_param_groups(MuonConfig(weight_decay=0.1, **config), named_params, model, 1.0, False)
        return [(g["algorithm"], g["lr"], g["weight_decay"], [p.shape for p in g["params"]]) for g in result]

    adamw_shapes = [(2,), (32, 8), (8,), (8,), (32, 8)]
    assert groups() == [("muon", 1.0, 0.1, [(8, 8)]), ("adamw", 1.0, 0.1, adamw_shapes)]
    recipe = groups(update_rms=0.18, decay_bias_and_scales=False, embedding_optimizer="sinkhorn")
    assert recipe == [
        ("muon", pytest.approx(0.9), pytest.approx(0.1 / 0.9), [(8, 8)]),
        ("adamw", 1.0, 0.1, [(8,)]),  # the norm weight
        ("adamw", 1.0, 0.0, [(2,), (8,)]),  # the sinks and the bias
        ("sinkhorn", 1.0, 0.0, [(32, 8), (32, 8)]),
    ]
