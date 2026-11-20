import pytest
import torch

from prime_rl.trainer.rl.step import sparse_head_backward


def micro_batch(loss_mask: list[bool]) -> dict:
    """Three tokens whose support rows hold the first two tokens but not the third."""
    return {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "loss_mask": torch.tensor([loss_mask]),
        "rl_weights": None,
        "ce_weights": None,
        "ref_kl_weights": None,
        "support_ids": torch.tensor([5, 1, 6, 2, 9, 3]),
        "support_offsets": torch.tensor([0, 2, 4, 6]),
        "support_replayed": torch.ones(3, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "loss_mask,expected",
    [
        ([False, True, False], True),
        ([False, True, True], False),
        ([False, False, False], False),
    ],
)
def test_sparse_head_backward(loss_mask: list[bool], expected: bool) -> None:
    assert sparse_head_backward(micro_batch(loss_mask)) is expected


def test_sparse_head_backward_needs_rl_only() -> None:
    batch = micro_batch([False, True, False])
    batch["ce_weights"] = torch.ones(1, 3)
    assert not sparse_head_backward(batch)
