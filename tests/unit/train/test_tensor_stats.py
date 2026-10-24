import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.utils import TensorStats

pytestmark = [pytest.mark.gpu]


@pytest.fixture(scope="module", autouse=True)
def process_group():
    if dist.is_initialized():
        yield
        return
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29512", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


def test_tensor_stats_match_torch():
    torch.manual_seed(0)
    stats = TensorStats()
    names = ["math", "code", "empty"]
    all_values, all_groups, losses = [], [], []
    for _ in range(3):
        values = torch.randn(1000, device="cuda")
        mask = torch.rand(1000, device="cuda") > 0.3
        groups = torch.randint(0, 2, (1000,), device="cuda")
        loss = torch.rand((), device="cuda")
        stats.add("entropy/all", values, mask)
        stats.add_grouped("entropy", values, mask, groups, names)
        stats.add("empty", values, torch.zeros_like(mask))
        stats.add("loss", loss)
        all_values.append(values[mask])
        all_groups.append(groups[mask])
        losses.append(loss)

    metrics = stats.compute_stats()

    values, groups, losses = torch.cat(all_values), torch.cat(all_groups), torch.stack(losses)
    expected = {
        "entropy/all": values,
        "entropy/math": values[groups == 0],
        "entropy/code": values[groups == 1],
        "loss": losses,
    }
    for key, key_values in expected.items():
        assert metrics[f"{key}/mean"] == pytest.approx(key_values.mean().item(), rel=1e-5)
        assert metrics[f"{key}/std"] == pytest.approx(key_values.std().item(), rel=1e-5)
        assert metrics[f"{key}/min"] == pytest.approx(key_values.min().item())
        assert metrics[f"{key}/max"] == pytest.approx(key_values.max().item())
        assert f"{key}/median" not in metrics
    assert not any(key.startswith(("entropy/empty", "empty/")) for key in metrics)
    assert stats.compute_stats() == {}
