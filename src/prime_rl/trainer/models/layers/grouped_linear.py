import torch
from torch import nn


class GroupedLinear(nn.Linear):
    """Block-diagonal linear that projects each of `n_groups` input groups independently to `out_features / n_groups` channels.

    Input is `(..., n_groups, in_features_per_group)`, output `(..., n_groups, out_features / n_groups)`.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int):
        super().__init__(in_features_per_group, out_features, bias=False)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)
