from typing import Generator

import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.utils import Tensors

pytestmark = [pytest.mark.gpu]


@pytest.fixture(autouse=True, scope="module")
def init_process_group(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    if dist.is_initialized():
        yield
        return

    init_file = tmp_path_factory.mktemp("dist") / "init"
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_tensors_compute_stats_handles_integer_tensors() -> None:
    counts = torch.tensor([504, 0, 0, 0], dtype=torch.int64)
    expected = counts.float()

    tensors = Tensors()
    tensors["mtp_token_count"].append(counts)

    stats = tensors.compute_stats()

    assert stats["mtp_token_count/mean"] == pytest.approx(expected.mean().item())
    assert stats["mtp_token_count/median"] == pytest.approx(torch.median(expected).item())
    assert stats["mtp_token_count/std"] == pytest.approx(expected.std().item())
    assert stats["mtp_token_count/min"] == pytest.approx(expected.min().item())
    assert stats["mtp_token_count/max"] == pytest.approx(expected.max().item())
