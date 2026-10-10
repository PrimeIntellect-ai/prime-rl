"""Momentum update with Sinkhorn balancing (DeepSeek-V4.1 tech report, section 2.5, Algorithm 1).

For an (m, n) matrix whose rows are tokens or n-gram ids (embedding tables, the LM head): Nesterov
momentum, then K alternating row / column l2 normalizations of the whole update matrix (rows first,
K odd, rows with norm <= tau x the mean row norm zeroed), scaled by sqrt(n) and applied with the
learning rate times gamma. No weight decay; one momentum buffer.

The normalized matrix is never written: it is `diag(r) G diag(c)` for the Nesterov update `G`, and
each normalization only updates a scale vector. A row step reads `G` once and also accumulates the
next column step's statistics. Rows may be sharded across ranks (FSDP's `Shard(0)`, row-sharded
Engram tables): row statistics are local, the column statistics and the mean row norm are summed
over the shard group.
"""

import math
from itertools import chain

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from dion import Muon
from dion.opt_utils import AsyncRuntime
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard

# Rows of up to this many columns (the Engram tables' 256) are normalized in one tile, which lets
# a row step accumulate the column statistics from the tile it already loaded.
_MAX_SINGLE_TILE_COLS = 512


@triton.jit
def _nesterov_kernel(grad, momentum, rho, n_rows, n_cols, beta, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """momentum = beta * momentum + (1 - beta) * grad; grad <- beta * momentum + (1 - beta) * grad; rho = row norms."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols < n_cols)[None, :]
        offsets = rows.to(tl.int64)[:, None] * n_cols + cols[None, :]
        g = tl.load(grad + offsets, mask=mask, other=0.0)
        m = beta * tl.load(momentum + offsets, mask=mask, other=0.0) + (1.0 - beta) * g
        g_hat = beta * m + (1.0 - beta) * g
        tl.store(momentum + offsets, m, mask=mask)
        tl.store(grad + offsets, g_hat, mask=mask)
        acc += tl.sum(g_hat * g_hat, axis=1)
    tl.store(rho + rows, tl.sqrt(acc), mask=row_mask)


@triton.jit
def _row_scale(r, t, eps):
    # A zero row of the normalized matrix stays zero; keeping its scale at zero is equivalent and
    # avoids an unbounded r / eps.
    rt = r * t
    return tl.where(rt > 0.0, r / (rt + eps), 0.0)


@triton.jit
def _row_step_single_tile_kernel(
    g_hat, r, c, col_sq, n_rows, n_cols, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Row normalization (updates `r`) fused with the next column step's statistics, `col_sq += sum_i (r_i G_ij)^2`."""
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols
    c_vals = tl.load(c + cols, mask=col_mask, other=0.0)
    col_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for block in range(tl.program_id(0), tl.cdiv(n_rows, BLOCK_M), tl.num_programs(0)):
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows
        offsets = rows.to(tl.int64)[:, None] * n_cols + cols[None, :]
        g = tl.load(g_hat + offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        gc = g * c_vals[None, :]
        r_new = _row_scale(tl.load(r + rows, mask=row_mask, other=0.0), tl.sqrt(tl.sum(gc * gc, axis=1)), eps)
        tl.store(r + rows, r_new, mask=row_mask)
        rg = r_new[:, None] * g
        col_acc += tl.sum(rg * rg, axis=0)
    tl.atomic_add(col_sq + cols, col_acc, mask=col_mask)


@triton.jit
def _row_step_kernel(g_hat, r, c, n_rows, n_cols, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Row normalization for rows wider than one tile (updates `r`)."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols
        offsets = rows.to(tl.int64)[:, None] * n_cols + cols[None, :]
        g = tl.load(g_hat + offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        gc = g * tl.load(c + cols, mask=col_mask, other=0.0)[None, :]
        acc += tl.sum(gc * gc, axis=1)
    r_new = _row_scale(tl.load(r + rows, mask=row_mask, other=0.0), tl.sqrt(acc), eps)
    tl.store(r + rows, r_new, mask=row_mask)


@triton.jit
def _col_stats_kernel(g_hat, r, col_sq, n_rows, n_cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """`col_sq += sum_i (r_i G_ij)^2` for one column tile (grid axis 0) over a strided set of row blocks (axis 1)."""
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols
    col_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for block in range(tl.program_id(1), tl.cdiv(n_rows, BLOCK_M), tl.num_programs(1)):
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows
        offsets = rows.to(tl.int64)[:, None] * n_cols + cols[None, :]
        g = tl.load(g_hat + offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        rg = tl.load(r + rows, mask=row_mask, other=0.0)[:, None] * g
        col_acc += tl.sum(rg * rg, axis=0)
    tl.atomic_add(col_sq + cols, col_acc, mask=col_mask)


@triton.jit
def _last_row_step_and_update_kernel(
    param, g_hat, r, c, n_rows, n_cols, eps, step_size, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """The final (row) normalization, then `param -= step_size * r_i * G_ij * c_j`."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols
        offsets = rows.to(tl.int64)[:, None] * n_cols + cols[None, :]
        g = tl.load(g_hat + offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        gc = g * tl.load(c + cols, mask=col_mask, other=0.0)[None, :]
        acc += tl.sum(gc * gc, axis=1)
    r_new = _row_scale(tl.load(r + rows, mask=row_mask, other=0.0), tl.sqrt(acc), eps)
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows.to(tl.int64)[:, None] * n_cols + cols[None, :]
        g = tl.load(g_hat + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c + cols, mask=col_mask, other=0.0)
        w = tl.load(param + offsets, mask=mask)
        w = w.to(tl.float32) - step_size * (r_new[:, None] * g * c_vals[None, :])
        tl.store(param + offsets, w.to(param.dtype.element_ty), mask=mask)


def _row_shard_groups(param: Tensor) -> list[dist.ProcessGroup]:
    """The process groups across which `param`'s rows are split (none for a local or replicated tensor)."""
    if not isinstance(param, DTensor):
        return []
    groups = []
    for dim, placement in enumerate(param.placements):
        if isinstance(placement, Shard) and placement.dim == 0:
            groups.append(param.device_mesh.get_group(dim))
        elif not placement.is_replicate():
            raise NotImplementedError(f"Sinkhorn update needs row-sharded or replicated parameters, got {placement}")
    return groups


def _all_reduce(tensor: Tensor, groups: list[dist.ProcessGroup]) -> None:
    for group in groups:
        dist.all_reduce(tensor, group=group)


@torch.no_grad()
def sinkhorn_update_(
    param: Tensor,
    grad: Tensor,
    momentum: Tensor,
    lr: float,
    beta: float,
    num_iters: int = 11,
    tau: float = 1e-3,
    eps: float = 1e-20,
    lr_scale: float = 0.18,
) -> None:
    """One step of Algorithm 1 on `param` (in place). `momentum` is updated and `grad` is overwritten with
    the Nesterov update. Row-sharded DTensors are reduced over their shard groups; every rank of a
    group must call this together."""
    if num_iters % 2 != 1:
        raise ValueError(f"Sinkhorn balancing needs an odd number of normalizations, got {num_iters}")
    groups = _row_shard_groups(param)
    n_rows_global, n_cols = param.shape
    param, grad, momentum = (t.to_local() if isinstance(t, DTensor) else t for t in (param, grad, momentum))
    for name, tensor in (("param", param), ("grad", grad), ("momentum", momentum)):
        if not (tensor.is_cuda and tensor.is_contiguous() and tensor.ndim == 2):
            raise ValueError(f"Sinkhorn update needs a contiguous 2-D CUDA {name}")
    if grad.dtype != torch.float32 or momentum.dtype != torch.float32:
        raise ValueError("Sinkhorn update keeps the Nesterov update in the fp32 gradient buffer")
    n_rows = param.shape[0]
    device = param.device

    single_tile = n_cols <= _MAX_SINGLE_TILE_COLS
    block_n = triton.next_power_of_2(n_cols) if single_tile else 256
    block_m = max(1, 8192 // block_n)
    row_grid = (max(1, triton.cdiv(n_rows, block_m)),)
    persistent = (max(1, min(row_grid[0], 8 * torch.cuda.get_device_properties(device).multi_processor_count)),)

    rho = torch.empty(n_rows, dtype=torch.float32, device=device)
    if n_rows:
        _nesterov_kernel[row_grid](grad, momentum, rho, n_rows, n_cols, beta, BLOCK_M=block_m, BLOCK_N=block_n)
    rho_sum = rho.sum()
    _all_reduce(rho_sum, groups)
    # The first row normalization (column scales all one) starts from `r = 1` on unmasked rows.
    r = (rho > tau * rho_sum / n_rows_global).float()
    c = torch.ones(n_cols, dtype=torch.float32, device=device)
    col_sq = torch.empty(n_cols, dtype=torch.float32, device=device)

    for _ in range(num_iters // 2):
        col_sq.zero_()
        if n_rows and single_tile:
            _row_step_single_tile_kernel[persistent](
                grad, r, c, col_sq, n_rows, n_cols, eps, BLOCK_M=block_m, BLOCK_N=block_n
            )
        elif n_rows:
            _row_step_kernel[row_grid](grad, r, c, n_rows, n_cols, eps, BLOCK_M=block_m, BLOCK_N=block_n)
            _col_stats_kernel[(triton.cdiv(n_cols, block_n), persistent[0])](
                grad, r, col_sq, n_rows, n_cols, BLOCK_M=block_m, BLOCK_N=block_n
            )
        _all_reduce(col_sq, groups)
        col_norm = c * col_sq.sqrt()
        c = torch.where(col_norm > 0, c / (col_norm + eps), 0.0)

    if n_rows:
        step_size = lr * lr_scale * math.sqrt(n_cols)
        _last_row_step_and_update_kernel[row_grid](
            param, grad, r, c, n_rows, n_cols, eps, step_size, BLOCK_M=block_m, BLOCK_N=block_n
        )


def sinkhorn_reference(
    weight: Tensor,
    grad: Tensor,
    momentum: Tensor,
    lr: float,
    beta: float,
    num_iters: int = 11,
    tau: float = 1e-3,
    eps: float = 1e-20,
    lr_scale: float = 0.18,
) -> tuple[Tensor, Tensor]:
    """Algorithm 1 as written, on a dense matrix: returns the new weight and momentum."""
    momentum = beta * momentum + (1 - beta) * grad
    g_hat = beta * momentum + (1 - beta) * grad
    rho = g_hat.norm(dim=1)
    u = torch.where((rho <= tau * rho.mean())[:, None], 0.0, g_hat)
    for k in range(1, num_iters + 1):
        dim = 1 if k % 2 == 1 else 0
        u = u / (u.norm(dim=dim, keepdim=True) + eps)
    delta = math.sqrt(weight.shape[1]) * u
    return weight - lr_scale * lr * delta, momentum


class MuonWithSinkhorn(Muon):
    """Dion's Muon with one more per-group algorithm, `"sinkhorn"` (`sinkhorn_update_`).

    A sinkhorn group takes `lr`, `mu` (momentum), `sinkhorn_iters`, `sinkhorn_tau`, `sinkhorn_eps` and
    `sinkhorn_lr_scale`; its state is one fp32 momentum buffer per parameter."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        groups = {"muon": [], "lion": [], "adamw": [], "sinkhorn": []}
        for group in self.param_groups:
            group["step"] += 1
            if group["algorithm"] not in groups:
                raise ValueError(f"Unknown algorithm: {group['algorithm']}")
            groups[group["algorithm"]].append(group)

        for group in groups["sinkhorn"]:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self._get_or_initialize_state(param, "sinkhorn")
                sinkhorn_update_(
                    param,
                    param.grad,
                    state["momentum"],
                    lr=group["lr"],
                    beta=group["mu"],
                    num_iters=group["sinkhorn_iters"],
                    tau=group["sinkhorn_tau"],
                    eps=group["sinkhorn_eps"],
                    lr_scale=group["sinkhorn_lr_scale"],
                )

        # Tasks start when created, so the runtime must draw them lazily.
        tasks = chain(
            self._create_muon_tasks(groups["muon"]),
            self._create_lion_tasks(groups["lion"]),
            self._create_adamw_tasks(groups["adamw"]),
        )
        AsyncRuntime(tasks, max_concurrent_tasks=3).run()
        return loss
