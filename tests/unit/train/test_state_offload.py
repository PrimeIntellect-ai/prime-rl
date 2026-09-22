from typing import NamedTuple

import pytest
import torch
from torch.optim import Optimizer

from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer

pytestmark = [pytest.mark.gpu]


class TrainedPair(NamedTuple):
    reference_model: torch.nn.Module
    reference_optimizer: Optimizer
    offloaded_model: torch.nn.Module
    offloaded_optimizer: CPUOffloadOptimizer


def make_model(seed: int) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 32))
    return model.cuda()


def run_steps(model: torch.nn.Module, optimizer: Optimizer, num_steps: int, seed: int) -> None:
    torch.manual_seed(seed)
    for _ in range(num_steps):
        batch = torch.randn(16, 64, device="cuda")
        loss = model(batch).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def assert_params_equal(reference_model: torch.nn.Module, model: torch.nn.Module) -> None:
    for (name, reference_param), (_, param) in zip(reference_model.named_parameters(), model.named_parameters()):
        assert torch.equal(reference_param, param), f"param mismatch: {name}"


def assert_states_offloaded(optimizer: CPUOffloadOptimizer) -> None:
    for param in optimizer.state:
        for key, value in optimizer.state[param].items():
            assert value.device.type == "cpu", f"state '{key}' on {value.device}, expected cpu"
            assert value.is_pinned(), f"state '{key}' not pinned"


def assert_state_dicts_equal(saved: dict, restored: dict) -> None:
    assert saved["param_groups"] == restored["param_groups"]
    assert saved["state"].keys() == restored["state"].keys()
    for index in saved["state"]:
        assert saved["state"][index].keys() == restored["state"][index].keys()
        for key, saved_value in saved["state"][index].items():
            restored_value = restored["state"][index][key]
            if isinstance(saved_value, torch.Tensor):
                assert torch.equal(saved_value.cpu(), restored_value.cpu()), f"state[{index}]['{key}'] mismatch"
            else:
                assert saved_value == restored_value, f"state[{index}]['{key}'] mismatch"


def state_data_ptrs(optimizer: CPUOffloadOptimizer) -> dict[tuple[int, str], int]:
    return {
        (id(param), key): value.data_ptr() for param in optimizer.state for key, value in optimizer.state[param].items()
    }


@pytest.fixture
def trained_pair() -> TrainedPair:
    reference_model = make_model(0)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-2)
    run_steps(reference_model, reference_optimizer, num_steps=3, seed=1)

    offloaded_model = make_model(0)
    offloaded_optimizer = CPUOffloadOptimizer(torch.optim.AdamW(offloaded_model.parameters(), lr=1e-2))
    run_steps(offloaded_model, offloaded_optimizer, num_steps=3, seed=1)

    return TrainedPair(reference_model, reference_optimizer, offloaded_model, offloaded_optimizer)


def test_offloaded_steps_match_plain_adamw(trained_pair: TrainedPair) -> None:
    assert_params_equal(trained_pair.reference_model, trained_pair.offloaded_model)


def test_states_are_on_cpu_and_pinned_between_steps(trained_pair: TrainedPair) -> None:
    assert_states_offloaded(trained_pair.offloaded_optimizer)


def test_state_dict_round_trip_is_exact_and_training_continues(trained_pair: TrainedPair) -> None:
    optimizer = trained_pair.offloaded_optimizer

    saved = optimizer.state_dict()
    optimizer.load_state_dict(saved)
    restored = optimizer.state_dict()

    assert_state_dicts_equal(saved, restored)
    assert_states_offloaded(optimizer)

    run_steps(trained_pair.reference_model, trained_pair.reference_optimizer, num_steps=1, seed=2)
    run_steps(trained_pair.offloaded_model, optimizer, num_steps=1, seed=2)
    assert_params_equal(trained_pair.reference_model, trained_pair.offloaded_model)


def test_offload_reuses_pinned_buffers_across_steps() -> None:
    model = make_model(0)
    optimizer = CPUOffloadOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-2))
    run_steps(model, optimizer, num_steps=2, seed=1)

    pointers_before = state_data_ptrs(optimizer)

    run_steps(model, optimizer, num_steps=1, seed=2)

    pointers_after = state_data_ptrs(optimizer)
    assert pointers_before.keys() == pointers_after.keys()
    for key, pointer in pointers_before.items():
        assert pointers_after[key] == pointer, f"state buffer reallocated for {key}"
