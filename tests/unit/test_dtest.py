import pytest
import torch
import torch.distributed as dist

from tests.dtest import DTest

pytestmark = [pytest.mark.gpu]


def fn_for_traceback_testing():
    print("I should fail")
    assert False, "asserting False"


class TestDTest(DTest):
    def test_basic(self) -> None:
        print(f"{self.rank=}")

    def test_all_reduce(self) -> None:
        t = torch.arange(self.world_size, device=self.device)
        dist.all_reduce(t)
        expected = sum(range(self.world_size)) * self.world_size
        assert t.sum().item() == expected

    def test_fail(self) -> None:
        fn_for_traceback_testing()

    def test_nice_printing(self) -> None:
        self.print_rank(f"Hi from {self.rank=}")

    def test_default_world_size(self) -> None:
        self.print_rank(f"{self.world_size=}")

    @pytest.mark.parametrize("n", (2, 3, 4))
    def test_parametrize(self, n) -> None:
        self.print_rank(f"{n=}")

    def test_root_error_visible_on_hang(self) -> None:
        """Rank 0 fails; rank 1 is stuck in a collective rank 0 won't enter. Verifies mp.spawn's
        terminate-the-others-on-first-failure behavior rather than hanging until NCCL timeout."""
        if dist.get_rank() == 0:
            assert False, "intentional rank 0 failure"
        else:
            dist.barrier()


class TestOtherDefaultWorldSize(DTest):
    default_world_size = 4

    def test_default_world_size(self) -> None:
        assert self.world_size == 4
        self.print_rank(f"{self.world_size=}")


def test_regular():
    """A plain function test still runs normally — confirms the new conftest.py hook's
    `issubclass(cls, DTest)` guard is a true no-op for non-DTest tests."""
    assert True


@pytest.mark.parametrize("prop", ("rank", "world_size"))
def test_rank_props_raise_outside_worker(prop: str) -> None:
    with pytest.raises(RuntimeError, match="only available inside a spawned worker"):
        getattr(DTest(), prop)
