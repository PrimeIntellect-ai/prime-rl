import pytest
import torch
from torch.nn import functional as F

from prime_rl.trainer.rl.hindsight import hindsight_loss


@pytest.mark.parametrize("eligible", [3, 1, 0])
def test_gcsl_episode_mean_and_gradient(eligible):
    torch.manual_seed(19)
    logits = torch.randn(3, 2, 4, requires_grad=True)
    picked = torch.tensor([[0, 3], [2, 2], [1, 0]])
    loss = hindsight_loss(logits[:eligible], picked[:eligible], episode_count=3)
    gold = F.cross_entropy(logits[:eligible].reshape(-1, 4), picked[:eligible].reshape(-1), reduction="sum") / 3
    torch.testing.assert_close(loss, gold)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(gold, logits)[0])


def test_invalid_episode_count():
    with pytest.raises(ValueError):
        hindsight_loss(torch.zeros(2, 1, 4), torch.zeros(2, 1, dtype=torch.long), episode_count=1)
