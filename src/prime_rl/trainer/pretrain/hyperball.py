import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _matrix_norms(local: torch.Tensor, param: nn.Parameter) -> torch.Tensor:
    """Frobenius norm of every matrix (the last two dims) of ``param``, from its local shard ``local``. Shards of
    the matrix dims are summed over their process groups; experts sharded along dim 0 are whole on each rank."""
    squares = local.float().square().sum(dim=(-2, -1), keepdim=True)
    if isinstance(param, DTensor):
        for mesh_dim, placement in enumerate(param.placements):
            if placement.is_shard() and placement.dim >= param.ndim - 2:
                dist.all_reduce(squares, group=param.device_mesh.get_group(mesh_dim))
    return squares.sqrt()


class Hyperball:
    """Hyperball (Wen et al. 2026, https://arxiv.org/abs/2606.16899) for the optimizer groups marked
    ``hyperball``. Each matrix W of radius R = |W|_F takes the step

        W <- R * normalize(W - lr * R * normalize(u)),

    where u is the group's base update (Muon or Adam, without weight decay or shape adjustment), so the norm and the
    relative step size of every matrix are fixed. The base optimizer applies u; the hooks recover it from the
    weights before and after its step and replace the step by the hyperball step."""

    def __init__(self, optimizer: Optimizer):
        self.groups = [group for group in optimizer.param_groups if group.get("hyperball")]
        self.before: list[list[torch.Tensor]] = []
        optimizer.register_step_pre_hook(self.save_weights)
        optimizer.register_step_post_hook(self.step)

    @torch.no_grad()
    def save_weights(self, *_) -> None:
        self.before = [[_local(param).clone() for param in group["params"]] for group in self.groups]

    @torch.no_grad()
    def step(self, *_) -> None:
        for group, before in zip(self.groups, self.before):
            for param, weight in zip(group["params"], before):
                update = weight - _local(param)
                radius = _matrix_norms(weight, param)
                # Floor as in Marin's implementation: an expert without tokens has no update.
                trial = weight - group["lr"] * radius * update / _matrix_norms(update, param).clamp_min(1e-10)
                _local(param).copy_(trial * (radius / _matrix_norms(trial, param)))
        self.before = []
