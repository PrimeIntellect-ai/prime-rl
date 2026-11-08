import copy
import math

import pytest
import torch

from prime_rl.trainer.optim.muon import NEWTON_SCHULZ, Muon, newton_schulz, orthogonalize, plan_buckets

DEFAULT = NEWTON_SCHULZ["default"]


def newton_schulz_fp32(G: torch.Tensor) -> torch.Tensor:
    X = G.mT if G.size(-2) > G.size(-1) else G
    X = X / X.norm(dim=(-2, -1), keepdim=True)
    for a, b, c in DEFAULT.coefficients:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if G.size(-2) > G.size(-1) else X


def newton_schulz_bf16(G: torch.Tensor) -> torch.Tensor:
    """The plain bf16 Newton-Schulz Muon has used so far (no fused GEMM epilogues)."""
    X = G.bfloat16()
    X = X.mT if G.size(-2) > G.size(-1) else X
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-8)
    for a, b, c in DEFAULT.coefficients:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if G.size(-2) > G.size(-1) else X


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 1024, 3072), (1, 3072, 1024), (8, 768, 2048), (4, 100, 60)])
def test_newton_schulz_is_as_accurate_as_bf16(shape):
    torch.manual_seed(0)
    G = torch.randn(shape, device="cuda")
    reference = newton_schulz_fp32(G)

    def error(X: torch.Tensor) -> float:
        return ((X.float() - reference).norm() / reference.norm()).item()

    assert error(newton_schulz(G, DEFAULT, 1e-8)) <= 1.25 * error(newton_schulz_bf16(G))


def test_orthogonalize_packed_matrices_each_on_their_own():
    torch.manual_seed(0)
    partitions = [
        (48, 16, 16),
        (8,) * 4,
        (8,) * 6 + (32,),  # per-head queries and keys, whole values
        (8, 16) * 3 + (8,),  # MLA [W_UK; W_UV] per head
    ]
    matrices = [torch.randn(3, sum(parts), 32) for parts in partitions]
    outputs = orthogonalize(matrices, partitions, DEFAULT, [1e-8] * len(partitions))
    for out, x, parts in zip(outputs, matrices, partitions):
        expected = torch.cat([newton_schulz(part, DEFAULT, 1e-8) for part in x.split(parts, dim=-2)], dim=-2)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=0)


def test_plan_buckets_keeps_order_and_balances_every_bucket():
    # Layers of a dense model: q, k, v, o, gate, up, down
    costs = [2.0, 0.1, 0.1, 2.0, 4.8, 4.8, 4.8] * 36
    nbytes = [32, 8, 8, 32, 96, 96, 96] * 36
    buckets = plan_buckets(costs, nbytes, world_size=8, bucket_bytes=100)
    assert [i for bucket in buckets for i, _ in bucket] == list(range(len(costs)))
    for bucket in buckets:
        assert sum(nbytes[i] for i, _ in bucket) <= 8 * 100
        loads = [sum(costs[i] for i, owner in bucket if owner == rank) for rank in range(8)]
        assert max(loads) - min(loads) <= max(costs[i] for i, _ in bucket)


def reference_muon_step(params, grads, states, partitions, lr=0.02, mu=0.95, weight_decay=0.1):
    """Muon with Nesterov momentum and an update RMS of 0.2, one matrix at a time."""
    for name, X in params.items():
        M = states[name]
        M.mul_(mu).add_(grads[name])
        U = (M * mu + grads[name]).bfloat16()
        parts = partitions.get(name, (X.size(-2),))
        updates = []
        for part in U.split(parts, dim=-2):
            O = newton_schulz(part.reshape(-1, *part.shape[-2:]), DEFAULT, 1e-8).reshape(part.shape)
            updates.append(O.float() * lr * 0.2 * math.sqrt(max(part.shape[-2:])))
        X.mul_(1 - lr * weight_decay).sub_(torch.cat(updates, dim=-2))


def test_muon_matches_reference():
    shapes = {"wide": (16, 48), "tall": (40, 24), "qkv": (48, 16), "experts": (3, 32, 16)}
    partitions = {"qkv": (32, 8, 8), "experts": (16, 16)}
    torch.manual_seed(0)
    params = {name: torch.nn.Parameter(torch.randn(shape)) for name, shape in shapes.items()}
    reference = {name: p.detach().clone() for name, p in params.items()}
    states = {name: torch.zeros(shape) for name, shape in shapes.items()}
    optimizer = Muon(
        [dict(params=list(params.values()), algorithm="muon")],
        lr=0.02,
        weight_decay=0.1,
        matrix_partitions={params[name]: parts for name, parts in partitions.items()},
    )
    for _ in range(3):
        grads = {name: torch.randn(shape) for name, shape in shapes.items()}
        for name, p in params.items():
            p.grad = grads[name]
        optimizer.step()
        reference_muon_step(reference, grads, states, partitions)
    for name, p in params.items():
        torch.testing.assert_close(p.detach(), reference[name], atol=1e-5, rtol=1e-5)


def test_subset_steps_match_the_full_step():
    """An offloaded step updates one chunk of units at a time, after the trainer has moved on (zero_grad)."""

    def train(chunked: bool) -> list[torch.Tensor]:
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(shape)) for shape in [(16, 48), (40, 24), (3, 32, 16), (24,)]]
        optimizer = Muon(
            [dict(params=params[:3], algorithm="muon"), dict(params=params[3:], algorithm="adamw")],
            lr=0.02,
            matrix_partitions={params[2]: (16, 16)},
            bucket_bytes=2048,
        )
        for step in range(3):
            for p in params:
                p.grad = torch.randn(p.shape)
            if chunked and step > 0:
                subset_step = optimizer.begin_subset_step()
                optimizer.zero_grad()
                for unit in reversed(subset_step.units):
                    subset_step.step_params(unit)
            else:
                optimizer.step()
        assert [group["step"] for group in optimizer.param_groups] == [3, 3]
        return [p.detach() for p in params]

    for reference, chunked in zip(train(False), train(True)):
        assert torch.equal(reference, chunked)


def test_resume_from_state_dict_matches_continuous_training():
    """load_state_dict replaces the param group dicts: the optimizer must keep working on the new ones."""

    def make():
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(shape)) for shape in [(16, 48), (3, 32, 16), (24,)]]
        groups = [dict(params=params[:2], algorithm="muon"), dict(params=params[2:], algorithm="adamw")]
        return params, Muon(groups, lr=0.02, matrix_partitions={params[1]: (16, 16)})

    def train(params, optimizer, steps):
        for step in steps:
            torch.manual_seed(step)
            for p in params:
                p.grad = torch.randn(p.shape)
            optimizer.step()

    params, optimizer = make()
    train(params, optimizer, range(2))
    saved = ([p.detach().clone() for p in params], copy.deepcopy(optimizer.state_dict()))
    train(params, optimizer, range(2, 4))

    resumed_params, resumed = make()
    with torch.no_grad():
        for p, value in zip(resumed_params, saved[0]):
            p.copy_(value)
    resumed.load_state_dict(saved[1])
    train(resumed_params, resumed, range(2, 4))
    for p, q in zip(params, resumed_params):
        assert torch.equal(p, q)
