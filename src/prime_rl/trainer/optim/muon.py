import math
from collections import defaultdict
from itertools import chain

import dion
import torch
import torch.distributed as dist
from dion.opt_utils import AsyncRuntime, AsyncTask, to_local
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

DEEPSEEK_V4_NEWTON_SCHULZ = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2


def newton_schulz_deepseek_v4(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    """DeepSeek-V4's hybrid Newton-Schulz: 8 fast-converging steps, then 2 steps that pin the singular values at 1."""
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in DEEPSEEK_V4_NEWTON_SCHULZ:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _norms(U: Tensor, dim: int, groups: list[ProcessGroup]) -> Tensor:
    """L2 norms of U along ``dim``, summed over the process groups that shard that dimension."""
    squares = U.square().sum(dim=dim)
    for group in groups:
        dist.all_reduce(squares, group=group)
    return squares.sqrt()


def sinkhorn_update(
    X: Tensor,
    G: Tensor,
    M: Tensor,
    *,
    lr: float,
    momentum: float,
    update_rms: float,
    tau: float,
    epsilon: float,
    iters: int,
    shape: tuple[int, int],
    row_groups: list[ProcessGroup],
    col_groups: list[ProcessGroup],
) -> None:
    """Momentum update with Sinkhorn balancing (DeepSeek-V4.1, Algorithm 1) on local shards of an m x n matrix.

    ``row_groups`` shard the columns (row norms reduce over them), ``col_groups`` shard the rows (column norms and
    the mean row norm reduce over them)."""
    M.lerp_(G.to(M.dtype), 1 - momentum)
    U = G.to(M.dtype).lerp(M, momentum).float()
    row_norms = _norms(U, 1, row_groups)
    mean_row_norm = row_norms.sum()
    for group in col_groups:
        dist.all_reduce(mean_row_norm, group=group)
    mean_row_norm /= shape[0]
    U = U * (row_norms > tau * mean_row_norm).unsqueeze(1)
    for k in range(iters):
        if k % 2 == 0:
            U = U / (_norms(U, 1, row_groups) + epsilon).unsqueeze(1)
        else:
            U = U / (_norms(U, 0, col_groups) + epsilon).unsqueeze(0)
    X.sub_(U.to(X.dtype), alpha=lr * update_rms * math.sqrt(shape[1]))


def _shard_groups(param: Tensor, dim: int) -> list[ProcessGroup]:
    if not isinstance(param, DTensor):
        return []
    mesh = param.device_mesh
    return [
        mesh.get_group(mesh_dim)
        for mesh_dim, placement in enumerate(param.placements)
        if getattr(placement, "dim", None) == dim and mesh.size(mesh_dim) > 1
    ]


def _sinkhorn_task(param: Tensor, state: dict, group: dict):
    sinkhorn_update(
        to_local(param),
        to_local(param.grad),
        to_local(state["momentum"]),
        lr=group["lr"],
        momentum=group["mu"],
        update_rms=group["update_rms"],
        tau=group["tau"],
        epsilon=group["epsilon"],
        iters=group["iters"],
        shape=tuple(param.shape),
        row_groups=_shard_groups(param, 1),
        col_groups=_shard_groups(param, 0),
    )
    yield


class Muon(dion.Muon):
    """dion's Muon with one more algorithm: ``sinkhorn`` groups take the Sinkhorn-balanced momentum update."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = defaultdict(list)
        for group in self.param_groups:
            group["step"] += 1
            groups[group["algorithm"]].append(group)
        sinkhorn_tasks = (
            AsyncTask(_sinkhorn_task(p, self._get_or_initialize_state(p, "sinkhorn"), group))
            for group in groups["sinkhorn"]
            for p in group["params"]
            if p.grad is not None
        )
        tasks = chain(
            self._create_muon_tasks(groups["muon"]), self._create_adamw_tasks(groups["adamw"]), sinkhorn_tasks
        )
        AsyncRuntime(tasks, max_concurrent_tasks=3).run()
        return loss
