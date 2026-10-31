import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dion.muon import zeropower_via_newtonschulz5
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.optim import muon_param_groups
from prime_rl.trainer.optim.muon import NEWTON_SCHULZ, Muon, sinkhorn_update

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


# Polar Express (arXiv 2505.16932, Appendix A) before the 1.01 safety factor
REFERENCE_POLAR_EXPRESS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


@pytest.mark.parametrize("schedule, tolerance", [("deepseek_v4", 0.02), ("polar_express", 0.25), ("step3p5", 0.02)])
def test_orthogonalization_singular_values(schedule, tolerance):
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 96)]:
        singular_values = torch.linalg.svdvals(NEWTON_SCHULZ[schedule](torch.randn(shape)).float())
        assert (singular_values - 1).abs().max() < tolerance


def reference_dimuon(W, grads, lr, beta, update_rms):
    """Dimuon (github.com/anpaure/dimuon, dimuon.py) on one matrix in float64, with our update norm."""
    M = torch.zeros_like(W)

    def conditioner(energy):
        return (energy / energy.max() + 1e-4) ** -0.25 if energy.max() > 0 else torch.ones_like(energy)

    for G in grads:
        L, C = conditioner(M.square().sum(1))[:, None], conditioner(M.square().sum(0))[None, :]
        M = beta * M + (1 - beta) * G
        X = L * M * C
        transpose = X.shape[0] > X.shape[1]
        X = X.T if transpose else X
        X = X / (1.01 * X.norm())
        for a, b, c in REFERENCE_POLAR_EXPRESS[:5]:
            A = X @ X.T
            X = a / 1.01 * X + (b / 1.01**3 * A + c / 1.01**5 * A @ A) @ X
        D = L * (X.T if transpose else X) * C
        W = W - lr * update_rms * math.sqrt(W.numel()) * D / D.norm()
    return W


def reference_muown(W, grads, lr, momentum, betas, eps, weight_decay):
    """Muown (github.com/kcc-lion/muown, optim/muown.py) on one matrix, with dion's Newton-Schulz."""
    g = v_norm = W.norm(dim=1, keepdim=True)
    m_v, m_g, v_g = torch.zeros_like(W), torch.zeros_like(g), torch.zeros_like(g)
    for step, grad in enumerate(grads, 1):
        u = W / g
        v = u * v_norm
        grad_g = (grad * u).sum(dim=1, keepdim=True)
        grad_v = (g / v_norm) * (grad - u * grad_g)
        m_v = momentum * m_v + grad_v
        update = zeropower_via_newtonschulz5(grad_v + momentum * m_v).float()
        v_new = v - lr * 0.2 * max(W.shape) ** 0.5 * update
        m_g = betas[0] * m_g + (1 - betas[0]) * grad_g
        v_g = betas[1] * v_g + (1 - betas[1]) * grad_g**2
        g = g - lr * (m_g / (1 - betas[0] ** step)) / ((v_g / (1 - betas[1] ** step)).sqrt() + eps)
        v_norm = v_new.norm(dim=1, keepdim=True)
        W = g * v_new / v_norm - lr * weight_decay * W
        g = W.norm(dim=1, keepdim=True)
    return W


def run_variant(variant, params, grads, partitions=None, mesh=None, **kwargs):
    optimizer = Muon(
        [dict(params=params, algorithm="muon", weight_decay=0.1 if variant == "muown" else 0.0)],
        lr=0.01,
        mu=0.9,
        nesterov=variant == "muown",
        betas=(0.9, 0.95),
        epsilon=1e-8,
        update_rms=0.2,
        variant=variant,
        fsdp_mesh_dim=0,
        world_mesh=mesh,
        distributed_mesh=mesh,
        adjust_lr="rms_norm",
        matrix_partitions=partitions,
        **kwargs,
    )
    for step_grads in grads:
        for p, grad in zip(params, step_grads):
            p.grad = grad
        optimizer.step()
    return optimizer


@pytest.mark.parametrize("bf16_momentum", [False, True])
def test_dimuon_matches_reference(bf16_momentum):
    torch.manual_seed(0)
    W = torch.randn(24, 16)
    grads = torch.randn(3, 24, 16)
    whole = nn.Parameter(W.clone())
    run_variant("dimuon", [whole], grads[:, None], bf16_momentum=bf16_momentum)
    # a packed parameter trains like its matrices on their own
    packed = nn.Parameter(W.clone())
    run_variant("dimuon", [packed], grads[:, None], {packed: (16, 8)}, bf16_momentum=bf16_momentum)
    for result, row_blocks in [(whole, [slice(None)]), (packed, [slice(0, 16), slice(16, 24)])]:
        for rows in row_blocks:
            expected = reference_dimuon(W[rows].double(), grads[:, rows].double(), 0.01, 0.9, 0.2)
            delta, expected_delta = result[rows].detach() - W[rows], expected.float() - W[rows]
            assert (delta - expected_delta).norm() < 0.03 * expected_delta.norm()


def test_muown_matches_reference():
    torch.manual_seed(0)
    W = torch.randn(24, 8)
    grads = torch.randn(3, 24, 8)
    params = [nn.Parameter(W.clone())]
    run_variant("muown", params, grads[:, None], {params[0]: (8, 8, 8)})
    expected = torch.cat(
        [reference_muown(W[i : i + 8], grads[:, i : i + 8], 0.01, 0.9, (0.9, 0.95), 1e-8, 0.1) for i in (0, 8, 16)]
    )
    delta, expected_delta = params[0].detach() - W, expected - W
    assert (delta - expected_delta).norm() < 0.01 * expected_delta.norm()


def sharded_variants(rank, world_size, init_file, weights, grads, shard_dim):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    mesh = init_device_mesh("cpu", (world_size,))

    def shard(t):
        return distribute_tensor(t, mesh, [Shard(0 if shard_dim == 0 else t.ndim - 1)])

    for variant in ["dimuon", "muown"]:
        expected = [nn.Parameter(w.clone()) for w in weights]
        expected_state = run_variant(variant, expected, grads, {expected[0]: (16, 9), expected[1]: (16, 9)}).state
        sharded = [nn.Parameter(shard(w.clone())) for w in weights]
        sharded_grads = [[shard(g) for g in step] for step in grads]
        partitions = {sharded[0]: (16, 9), sharded[1]: (16, 9)}
        sharded_state = run_variant(variant, sharded, sharded_grads, partitions, mesh=mesh).state
        for actual, full in zip(sharded, expected):
            torch.testing.assert_close(actual.full_tensor(), full.detach(), rtol=1e-4, atol=1e-6)
            # the state, Muown's per-row state included, is laid out as without sharding
            for key, value in expected_state[full].items():
                torch.testing.assert_close(sharded_state[actual][key].full_tensor(), value, rtol=1e-4, atol=1e-6)
    dist.destroy_process_group()


@pytest.mark.parametrize("shard_dim", [0, -1])
def test_sharded_variants_match_unsharded(tmp_path, shard_dim):
    """Two packed matrices with uneven row shards (or column shards) and one batch of experts sharded by expert
    (or by column)."""
    torch.manual_seed(0)
    weights = [torch.randn(25, 12), torch.randn(25, 12), torch.randn(2, 12, 8)]
    grads = [[torch.randn_like(w) for w in weights] for _ in range(3)]
    mp.spawn(sharded_variants, args=(2, tmp_path / "init", weights, grads, shard_dim), nprocs=2)


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
    recipe = groups(update_rms=0.18, decay_bias_and_scales=False, embeddings={"type": "sinkhorn"})
    assert recipe == [
        ("muon", 1.0, 0.1, [(8, 8)]),
        ("adamw", 1.0, 0.1, [(8,)]),  # the norm weight
        ("adamw", 1.0, 0.0, [(2,), (8,)]),  # the sinks and the bias
        ("sinkhorn", 1.0, 0.0, [(32, 8), (32, 8)]),
    ]


def test_update_scales_hold_at_the_schedule_floor():
    def final_step(update_rms, lr_scale):
        torch.manual_seed(0)
        weight, embedding = nn.Parameter(torch.randn(16, 8)), nn.Parameter(torch.randn(32, 8))
        groups = [
            dict(params=[weight], algorithm="muon", weight_decay=0.1),
            dict(params=[embedding], algorithm="sinkhorn", weight_decay=0.0, lr_scale=lr_scale, tau=1e-3, iters=11),
        ]
        optimizer = Muon(
            groups,
            lr=1.0,
            update_rms=update_rms,
            fsdp_mesh_dim=0,
            world_mesh=None,
            adjust_lr="rms_norm",
            newton_schulz_func=NEWTON_SCHULZ["deepseek_v4"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=4, eta_min=0.1)
        for _ in range(4):
            scheduler.step()
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.1, 0.1])
        before = weight.detach().clone(), embedding.detach().clone()
        weight.grad, embedding.grad = torch.randn(16, 8), torch.randn(32, 8)
        optimizer.step()
        return before[0], weight.detach() - before[0], embedding.detach() - before[1]

    weight, weight_delta, embedding_delta = final_step(0.2, 1.0)
    _, scaled_weight_delta, scaled_embedding_delta = final_step(0.1, 5.0)
    decay = -0.1 * 0.1 * weight  # lr * weight_decay does not depend on update_rms
    torch.testing.assert_close(scaled_weight_delta - decay, 0.5 * (weight_delta - decay))
    torch.testing.assert_close(scaled_embedding_delta, 2.5 * embedding_delta)
