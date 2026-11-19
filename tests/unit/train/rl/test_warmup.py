from collections.abc import Callable

import dion
import pytest
import torch
from torch import nn

from prime_rl.trainer.optim import OptimizerLike
from prime_rl.trainer.optim.sign_sgd import SignSGD
from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer
from prime_rl.trainer.rl.warmup import dry_optimizer_step, warmup_micro_batches


def make_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 32)).cuda()


def muon(model: nn.Module) -> dion.Muon:
    groups = [
        dict(params=[p for p in model.parameters() if p.ndim == 2], algorithm="muon"),
        dict(params=[p for p in model.parameters() if p.ndim == 1], algorithm="adamw"),
    ]
    return dion.Muon(groups, lr=1e-2, weight_decay=0.1, fsdp_mesh_dim=0, world_mesh=None, adjust_lr="rms_norm")


OPTIMIZERS: dict[str, Callable[[nn.Module], OptimizerLike]] = {
    "adamw_fused": lambda model: torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1, fused=True),
    "adamw_offload": lambda model: CPUOffloadOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    ),
    "muon_offload": lambda model: CPUOffloadOptimizer(muon(model)),
    "sign_sgd": lambda model: SignSGD(model.parameters(), lr=1e-2, weight_decay=0.1),
}


def backward(model: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    model(torch.randn(16, 64, device="cuda")).square().mean().backward()


def run_steps(model: nn.Module, optimizer: OptimizerLike, num_steps: int) -> None:
    for step in range(num_steps):
        backward(model, seed=step)
        optimizer.step()
        optimizer.zero_grad()


@pytest.mark.gpu
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_dry_step_does_not_change_training(name: str) -> None:
    reference_model = make_model()
    reference_optimizer = OPTIMIZERS[name](reference_model)
    run_steps(reference_model, reference_optimizer, num_steps=3)

    model = make_model()
    optimizer = OPTIMIZERS[name](model)
    initial = [param.detach().clone() for param in model.parameters()]
    backward(model, seed=100)
    assert dry_optimizer_step(optimizer, model, max_norm=1.0)
    for param, initial_param in zip(model.parameters(), initial):
        assert torch.equal(param, initial_param)
        assert param.grad is None
    run_steps(model, optimizer, num_steps=3)

    for reference_param, param in zip(reference_model.parameters(), model.parameters()):
        assert torch.equal(reference_param, param)


@pytest.mark.gpu
def test_dry_step_skips_existing_state() -> None:
    model = make_model()
    optimizer = OPTIMIZERS["muon_offload"](model)
    run_steps(model, optimizer, num_steps=2)
    params = [param.detach().clone() for param in model.parameters()]
    state = {
        (id(param), key): value.clone() for param, values in optimizer.state.items() for key, value in values.items()
    }
    steps = [group["step"] for group in optimizer.param_groups]

    backward(model, seed=100)
    assert not dry_optimizer_step(optimizer, model, max_norm=1.0)

    assert all(param.grad is None for param in model.parameters())
    assert all(torch.equal(param, saved) for param, saved in zip(model.parameters(), params))
    for param, values in optimizer.state.items():
        for key, value in values.items():
            assert torch.equal(value, state[(id(param), key)])
    assert [group["step"] for group in optimizer.param_groups] == steps


@pytest.mark.parametrize("seq_len,multiple", [(131072, 4), (4096, 1), (65536, 256), (100, 32)])
def test_warmup_micro_batches(seq_len: int, multiple: int) -> None:
    full, near_full, long = warmup_micro_batches(seq_len, multiple)
    assert sum(full) == seq_len // multiple * multiple
    assert sum(near_full) == sum(long) == sum(full) - multiple
    assert max(full) < sum(full) // 8
    assert max(long) == sum(long) // 2
    assert all(doc_len > 0 for doc_len in full + near_full + long)
