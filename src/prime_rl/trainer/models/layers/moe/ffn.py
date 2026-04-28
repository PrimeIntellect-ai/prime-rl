import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.moe.kernels import relu2


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class BCFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(F.silu(torch.matmul(x, self.w1.T)) * torch.matmul(x, self.w3.T), self.w2.T)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class BCNonGatedFeedForward(nn.Module):
    """Non-gated feed-forward network used as the shared expert in NemotronH.

    Uses relu2 activation: down_proj(relu2(up_proj(x))).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(relu2(self.up_proj(x)))
