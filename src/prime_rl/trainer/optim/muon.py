"""DeepSeek-V4 Muon for FSDP-sharded parameters (DeepSeek-V4 tech report §2.4 and §3.4.1, V4.1 §2.5).

For every logically independent matrix `W` (each routed expert, each query head, ...):

    M = mu * M + G                                  momentum (fp32, on the local shard)
    O = HybridNewtonSchulz(mu * M + G)              Nesterov, 10 bf16 iterations
    W = W * (1 - lr * wd) - lr * gamma * sqrt(max(n, m)) * O       update RMS gamma (0.18)

Distribution follows the report's hybrid ZeRO assignment, adapted to FSDP2 shards:
- A parameter whose local shard holds whole matrices (the routed experts, sharded on their expert dimension over
  the expert data-parallel group, or an unsharded parameter) is orthogonalized where it lives, with no
  communication, in batches of same-shape matrices.
- Every other parameter (row- or column-sharded over the FSDP group) gets one owner rank, chosen by a greedy
  balance of Newton-Schulz FLOPs. One all-to-all per process group sends every owner the bf16 shards of its
  parameters, the owners orthogonalize whole matrices, and a second all-to-all returns the updated shards. Only
  real data is exchanged: no parameter is padded to the group size. The local work runs while the first
  all-to-all is in flight.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.distributed as dist
from dion.newton_schulz_triton import ns_line_1, ns_line_2
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.trainer.optim.sinkhorn import MuonWithSinkhorn, sinkhorn_update_

# DeepSeek-V4 §2.4: 8 iterations that drive the singular values close to 1, then 2 that settle them at 1.
HYBRID_NS_COEFFICIENTS = ((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2

# Newton-Schulz runs on batches of at most this many matrix elements (bf16), which bounds its workspace
# to about four times this many bytes.
_NS_BATCH_NUMEL = 64 * 1024 * 1024


def hybrid_newton_schulz(G: Tensor, epsilon: float = 1e-7, use_triton: bool = False) -> Tensor:
    """Approximately orthogonalizes each matrix of `G` (2-D, or 3-D for a batch): `U V^T` for `G = U S V^T`.

    The input is scaled to unit Frobenius norm in fp32, then iterated in bf16 (fp32 accumulation). The
    symmetric products `X X^T` and `b A + c A A` compute only one triangle with `use_triton` (about 20% faster on
    expert shapes), but called eagerly from an optimizer step, dion's autotuned Triton kernels keep the calling
    frames, and with them that step's gradients, alive until the next step (+19 GB/GPU on a 3-layer stage)."""
    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    norm = X.float().norm(dim=(-2, -1), keepdim=True)
    X = (X.float() / (norm + epsilon)).to(torch.bfloat16).contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    baddbmm = torch.baddbmm if X.ndim == 3 else torch.addmm
    for a, b, c in HYBRID_NS_COEFFICIENTS:
        if use_triton:
            ns_line_1(X, out=A)
            ns_line_2(A, alpha=c, beta=b, out=B)
        else:
            torch.matmul(X, X.mT, out=A)
            torch.add(torch.matmul(A, A), A, alpha=b / c, out=B).mul_(c)
        baddbmm(X, B, X, beta=a, out=C)
        X, C = C, X
    return X.mT if transpose else X


def update_scale(rows: int, cols: int, update_rms: float) -> float:
    """Scales an orthogonalized `rows x cols` matrix (RMS `1 / sqrt(max)`) to RMS `update_rms`."""
    return update_rms * math.sqrt(max(rows, cols))


def ns_flops(rows: int, cols: int) -> float:
    m, n = min(rows, cols), max(rows, cols)
    return len(HYBRID_NS_COEFFICIENTS) * (4 * m * m * n + 2 * m**3)


def matrices(tensor: Tensor, row_blocks: tuple[int, ...] | None) -> list[Tensor]:
    """Views of the independent matrices of a whole parameter: each matrix of the batch along the leading
    dimensions (none for a 2-D parameter), split into row blocks if given."""
    batch = tensor.view(-1, *tensor.shape[-2:]).unbind(0)
    if row_blocks is None:
        return list(batch)
    return [block for matrix in batch for block in matrix.split(list(row_blocks), dim=0)]


def orthogonalize_(
    pairs: list[tuple[Tensor, Tensor]],
    scale: Callable[[int, int], float],
    newton_schulz: Callable[[Tensor], Tensor] = hybrid_newton_schulz,
    apply: Callable[[Tensor, Tensor], None] | None = None,
) -> None:
    """For each `(src, dst)` matrix pair, `dst <- scale(rows, cols) * NS(src)` (or `apply(dst, that)`).

    Same-shape matrices are stacked into batched Newton-Schulz calls of at most `_NS_BATCH_NUMEL` elements."""
    by_shape = defaultdict(list)
    for src, dst in pairs:
        by_shape[tuple(src.shape)].append((src, dst))
    for (rows, cols), group in by_shape.items():
        batch = max(1, _NS_BATCH_NUMEL // (rows * cols))
        s = scale(rows, cols)
        for start in range(0, len(group), batch):
            chunk = group[start : start + batch]
            out = newton_schulz(torch.stack([src for src, _ in chunk]))
            out.mul_(s)
            for (_, dst), o in zip(chunk, out.unbind(0)):
                if apply is None:
                    dst.copy_(o)
                else:
                    apply(dst, o)


@dataclass
class MuonParam:
    """One Muon parameter's step inputs: its hyperparameters and how its matrices are laid out."""

    param: Tensor
    lr: float
    mu: float
    weight_decay: float
    row_blocks: tuple[int, ...] | None = None


@dataclass
class _Unit:
    """A slice `[start, start + count)` along `shard_dim` of a parameter whose matrices are split across `group`
    (the whole parameter, or one of its row blocks), orthogonalized whole by one rank (`owner`)."""

    item: MuonParam
    entry: int
    group: dist.ProcessGroup
    shard_dim: int
    start: int
    count: int
    row_blocks: tuple[int, ...] | None
    owner: int = -1
    # Per rank of `group`: (first index, length) of that rank's shard inside this unit, torch.chunk layout as
    # FSDP2's `Shard`, and the offset of that first index in the rank's own shard.
    pieces: list[tuple[int, int]] = field(default_factory=list)

    def piece_shape(self, rank: int) -> list[int]:
        shape = list(self.item.param.shape)
        shape[self.shard_dim] = self.pieces[rank][1]
        return shape

    def piece_numel(self, rank: int) -> int:
        return math.prod(self.piece_shape(rank))

    def cost(self) -> float:
        shape = list(self.item.param.shape)
        shape[self.shard_dim] = self.count
        return sum(ns_flops(*m.shape[-2:]) for m in matrices(torch.empty(shape, device="meta"), self.row_blocks))


def _chunk_sizes(size: int, world: int) -> list[int]:
    chunk = -(-size // world)
    return [max(0, min(chunk, size - rank * chunk)) for rank in range(world)]


def _matrix_shard_dim(param: Tensor) -> tuple[dist.ProcessGroup, int] | None:
    """The process group and tensor dimension over which `param`'s matrices are split, or None when the local
    shard holds whole matrices (unsharded, replicated, or sharded on a batch dimension only)."""
    if not isinstance(param, DTensor):
        return None
    matrix_dims = {param.ndim - 2, param.ndim - 1}
    sharded = [
        (mesh_dim, placement.dim)
        for mesh_dim, placement in enumerate(param.placements)
        if placement.is_shard() and param.device_mesh.size(mesh_dim) > 1 and placement.dim in matrix_dims
    ]
    if not sharded:
        return None
    if len(sharded) > 1:
        raise NotImplementedError(f"Muon needs at most one matrix dimension sharded, got {param.placements}")
    mesh_dim, tensor_dim = sharded[0]
    if any(p.is_shard() and param.device_mesh.size(i) > 1 for i, p in enumerate(param.placements) if i != mesh_dim):
        raise NotImplementedError(f"Muon cannot combine a matrix-dimension shard with others: {param.placements}")
    return param.device_mesh.get_group(mesh_dim), tensor_dim


def assign_owners(costs: list[float], world: int) -> list[int]:
    """Greedy longest-first assignment of items to `world` ranks, balancing the summed cost (deterministic)."""
    load = [0.0] * world
    owners = [0] * len(costs)
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        rank = min(range(world), key=lambda r: (load[r], r))
        owners[index] = rank
        load[rank] += costs[index]
    return owners


def _local(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _units(entry: int, item: MuonParam, group: dist.ProcessGroup, shard_dim: int) -> list[_Unit]:
    """A row-sharded 2-D parameter with row blocks is exchanged block by block, so its blocks (e.g. the query
    heads) spread over several owners; anything else goes whole to one owner."""
    size = item.param.shape[shard_dim]
    if item.row_blocks is None or item.param.ndim != 2 or shard_dim != 0:
        return [_Unit(item, entry, group, shard_dim, 0, size, item.row_blocks)]
    starts = [sum(item.row_blocks[:k]) for k in range(len(item.row_blocks))]
    return [_Unit(item, entry, group, shard_dim, start, count, None) for start, count in zip(starts, item.row_blocks)]


@torch.no_grad()
def muon_step(
    items: list[MuonParam],
    grads: list[Tensor],
    momenta: list[Tensor],
    update_rms: float,
    newton_schulz: Callable[[Tensor], Tensor] = hybrid_newton_schulz,
    overlap: Callable[[], None] | None = None,
) -> None:
    """One DeepSeek-V4 Muon step on `items` (in place); `momenta` are the fp32 momentum buffers.

    Every rank of each FSDP group must call this with the same parameters in the same order. `overlap`, if
    given, runs while the shards travel to their owners (e.g. the element-wise optimizers' work)."""
    scale = lambda rows, cols: update_scale(rows, cols, update_rms)  # noqa: E731
    local, sharded = [], []
    for item, grad, momentum in zip(items, grads, momenta):
        placement = _matrix_shard_dim(item.param)
        if placement is None:
            local.append((item, _local(grad), _local(momentum)))
        else:
            group, dim = placement
            sharded.append((group, item, dim, _local(grad), _local(momentum)))

    # Sharded parameters: momentum on the local shard, then one all-to-all per group to the owners.
    by_group = defaultdict(list)
    for entry in sharded:
        by_group[entry[0]].append(entry[1:])
    exchanges = []
    for group, entries in by_group.items():
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        units = [u for e, (item, dim, _, _) in enumerate(entries) for u in _units(e, item, group, dim)]
        for owner, unit in zip(assign_owners([u.cost() for u in units], world), units):
            unit.owner = owner
            chunk = -(-unit.item.param.shape[unit.shard_dim] // world)
            for r in range(world):
                lo = max(unit.start, r * chunk)
                hi = min(unit.start + unit.count, (r + 1) * chunk, unit.item.param.shape[unit.shard_dim])
                unit.pieces.append((lo, max(0, hi - lo)))
        updates = _nesterov([g for *_, g, _ in entries], [m for *_, m in entries], [i.mu for i, *_ in entries])
        chunks = [-(-item.param.shape[dim] // world) for item, dim, _, _ in entries]
        # Send buffer ordered by destination owner, then by unit order.
        order = sorted(range(len(units)), key=lambda k: (units[k].owner, k))
        send = torch.cat(
            [
                _narrow(
                    updates[units[k].entry],
                    units[k].shard_dim,
                    units[k].pieces[rank][0] - rank * chunks[units[k].entry],
                    units[k].pieces[rank][1],
                )
                for k in order
            ]
        )
        in_splits = [sum(units[k].piece_numel(rank) for k in order if units[k].owner == r) for r in range(world)]
        mine = [k for k in order if units[k].owner == rank]
        out_splits = [sum(units[k].piece_numel(r) for k in mine) for r in range(world)]
        recv = torch.empty(sum(out_splits), dtype=torch.bfloat16, device=send.device)
        work = dist.all_to_all_single(recv, send, out_splits, in_splits, group=group, async_op=True)
        exchanges.append((group, entries, units, order, mine, chunks, in_splits, out_splits, recv, work))
    del sharded

    # Whole-matrix parameters (the experts) are orthogonalized locally, chunk by chunk, while shards travel.
    _local_step(local, scale, newton_schulz)
    if overlap is not None:
        overlap()

    returns = []
    for group, entries, units, order, mine, chunks, in_splits, out_splits, recv, work in exchanges:
        world = dist.get_world_size(group)
        work.wait()
        # recv holds, per source rank, that rank's piece of each unit this rank owns.
        pieces = [
            recv_rank.split([units[k].piece_numel(r) for k in mine])
            for r, recv_rank in enumerate(recv.split(out_splits))
        ]
        full = [
            torch.cat([pieces[r][j].view(units[k].piece_shape(r)) for r in range(world)], dim=units[k].shard_dim)
            for j, k in enumerate(mine)
        ]
        orthogonalize_(
            [(m, m) for j, k in enumerate(mine) for m in matrices(full[j], units[k].row_blocks)], scale, newton_schulz
        )
        # Return each rank its piece of every owned unit, in the order that rank sent them.
        back = [
            _narrow(full[j], units[k].shard_dim, units[k].pieces[r][0] - units[k].start, units[k].pieces[r][1])
            for r in range(world)
            for j, k in enumerate(mine)
        ]
        send_back = torch.cat(back) if back else recv.new_empty(0)
        recv_back = torch.empty(sum(in_splits), dtype=torch.bfloat16, device=recv.device)
        work = dist.all_to_all_single(recv_back, send_back, in_splits, out_splits, group=group, async_op=True)
        returns.append((entries, units, order, chunks, recv_back, work))

    for entries, units, order, chunks, recv_back, work in returns:
        work.wait()
        rank = dist.get_rank(units[0].group)
        deltas = [torch.empty_like(_local(item.param), dtype=torch.bfloat16) for item, *_ in entries]
        for k, piece in zip(order, recv_back.split([units[k].piece_numel(rank) for k in order])):
            unit = units[k]
            lo, n = unit.pieces[rank]
            if n:
                deltas[unit.entry].narrow(unit.shard_dim, lo - rank * chunks[unit.entry], n).copy_(
                    piece.view(unit.piece_shape(rank))
                )
        _apply(
            [_local(item.param) for item, *_ in entries],
            deltas,
            [i.lr for i, *_ in entries],
            [i.weight_decay for i, *_ in entries],
        )


def _narrow(tensor: Tensor, dim: int, start: int, length: int) -> Tensor:
    """`tensor.narrow(dim, start, length)` flattened; empty for a zero length, wherever `start` points."""
    if length == 0:
        return tensor.new_empty(0)
    return tensor.narrow(dim, start, length).flatten()


def _nesterov(grads: list[Tensor], momenta: list[Tensor], mus: list[float]) -> list[Tensor]:
    """`M <- mu M + G` in place; returns `mu M + G` in bf16."""
    out = []
    for grad, momentum, mu in zip(grads, momenta, mus):
        momentum.mul_(mu).add_(grad)
        out.append(torch.add(grad, momentum, alpha=mu).to(torch.bfloat16))
    return out


def _apply(params: list[Tensor], deltas: list[Tensor], lrs: list[float], decays: list[float]) -> None:
    """`W <- W (1 - lr wd) - lr delta`."""
    for param, delta, lr, wd in zip(params, deltas, lrs, decays):
        if wd != 0.0:
            param.mul_(1 - lr * wd)
        param.add_(delta, alpha=-lr)


def _local_step(local: list[tuple[MuonParam, Tensor, Tensor]], scale, newton_schulz) -> None:
    """Muon on parameters whose local tensor holds whole matrices, in batches of same-shape matrices, without
    materializing more than one batch of updates."""
    by_shape = defaultdict(list)
    for item, grad, momentum in local:
        param = _local(item.param)
        for p, g, m in zip(*(matrices(t, item.row_blocks) for t in (param, grad, momentum))):
            by_shape[tuple(p.shape)].append((item, p, g, m))
    for (rows, cols), group in by_shape.items():
        batch = max(1, _NS_BATCH_NUMEL // (rows * cols))
        s = scale(rows, cols)
        for start in range(0, len(group), batch):
            chunk = group[start : start + batch]
            updates = []
            for item, _, g, m in chunk:
                m.mul_(item.mu).add_(g)
                updates.append(torch.add(g, m, alpha=item.mu).to(torch.bfloat16))
            out = newton_schulz(torch.stack(updates)).mul_(s)
            _apply(
                [p for _, p, _, _ in chunk],
                list(out.unbind(0)),
                [i.lr for i, *_ in chunk],
                [i.weight_decay for i, *_ in chunk],
            )


class DeepSeekMuon(MuonWithSinkhorn):
    """DeepSeek-V4 Muon (`muon_step`) for the `"muon"` groups; AdamW and the Sinkhorn update as in
    `MuonWithSinkhorn`, run while the Muon shards travel.

    `row_blocks` maps a parameter to the heights of the independent matrices stacked along its second-to-last
    dimension (heads of a query projection, groups of a grouped projection, fused expert gate and up
    projections, ...). `update_rms` is gamma. A muon group's `mu` is its
    momentum; `adjust_lr`, `nesterov` and `epsilon` do not apply to it."""

    def __init__(self, params, row_blocks=None, update_rms: float = 0.18, use_triton: bool = False, **kwargs):
        super().__init__(params, **kwargs)
        self._row_blocks = dict(row_blocks or {})
        for param, blocks in self._row_blocks.items():
            if sum(blocks) != param.shape[-2]:
                raise ValueError(f"Row blocks {blocks} do not split a parameter of shape {tuple(param.shape)}")
        self._update_rms = update_rms
        self._newton_schulz = lambda x: hybrid_newton_schulz(x, use_triton=use_triton)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        groups = {"muon": [], "adamw": [], "sinkhorn": []}
        for group in self.param_groups:
            group["step"] += 1
            if group["algorithm"] not in groups:
                raise ValueError(f"Unknown algorithm: {group['algorithm']}")
            groups[group["algorithm"]].append(group)

        items, grads, momenta = [], [], []
        for group in groups["muon"]:
            for param in group["params"]:
                if param.grad is None:
                    continue
                items.append(
                    MuonParam(param, group["lr"], group["mu"], group["weight_decay"], self._row_blocks.get(param))
                )
                grads.append(param.grad)
                momenta.append(self._get_or_initialize_state(param, "muon")["momentum"])

        def elementwise():
            for group in groups["sinkhorn"]:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    sinkhorn_update_(
                        param,
                        param.grad,
                        self._get_or_initialize_state(param, "sinkhorn")["momentum"],
                        lr=group["lr"],
                        beta=group["mu"],
                        num_iters=group["sinkhorn_iters"],
                        tau=group["sinkhorn_tau"],
                        eps=group["sinkhorn_eps"],
                        lr_scale=group["sinkhorn_lr_scale"],
                    )
            for group in groups["adamw"]:
                self._adamw_step(group)

        muon_step(items, grads, momenta, self._update_rms, self._newton_schulz, overlap=elementwise)
        return loss

    def _adamw_step(self, group: dict) -> None:
        """AdamW (torch's fused kernel) on the local shards; state in dion's `momentum` / `variance` keys."""
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        states = [self._get_or_initialize_state(p, "adamw") for p in params]
        step = torch.tensor(float(group["step"]), device=_local(params[0]).device)
        torch._fused_adamw_(
            [_local(p) for p in params],
            [_local(p.grad) for p in params],
            [_local(s["momentum"]) for s in states],
            [_local(s["variance"]) for s in states],
            [],
            [step] * len(params),
            amsgrad=False,
            lr=group["lr"],
            beta1=group["beta1"],
            beta2=group["beta2"],
            weight_decay=group["weight_decay"],
            eps=group["epsilon"],
            maximize=False,
        )
