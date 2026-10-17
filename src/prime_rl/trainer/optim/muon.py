import math
from collections.abc import Generator, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim import Optimizer


@torch.compile(fullgraph=True)
def newton_schulz(G: Tensor, coefficients: tuple[tuple[float, float, float], ...], eps: float = 1e-8) -> Tensor:
    """Approximately orthogonalize the last two dims of ``G`` in bfloat16; leading dims are a batch."""
    X = G.to(torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b, c in coefficients:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def orthogonalize(
    U: Tensor, partitions: tuple[int, ...] | None, coefficients: tuple[tuple[float, float, float], ...]
) -> Tensor:
    """Newton-Schulz of a whole momentum tensor, each partition along dim -2 on its own."""
    if U.ndim > 3:
        return orthogonalize(U.flatten(end_dim=-3), partitions, coefficients).reshape(U.shape)
    if partitions is None:
        return newton_schulz(U, coefficients)
    if len(set(partitions)) == 1:
        return newton_schulz(U.unflatten(-2, (len(partitions), partitions[0])), coefficients).flatten(-3, -2)
    return torch.cat([newton_schulz(part, coefficients) for part in U.split(partitions, dim=-2)], dim=-2)


@dataclass
class Chunk:
    """Parameters that step together: a batch of same-layout Muon matrices or one AdamW group."""

    group: dict
    params: list[Tensor]
    first_owner: int = 0


class Muon(Optimizer):
    """Muon for FSDP2 parameters, with AdamW for the param groups whose ``algorithm`` is ``"adamw"``.

    Muon matrices get momentum (optionally Nesterov), Newton-Schulz on the whole matrix, and an update scaled to
    ``update_rms * sqrt(max(fan_out, fan_in))`` (Moonlight's matching of AdamW's update RMS). The last two dims are
    the matrix and leading dims a batch (stacked experts). ``matrix_partitions`` maps a parameter to the sizes of
    independent matrices packed along dim -2 (fused qkv / gate_up, head blocks); each is orthogonalized and
    scaled on its own.

    A parameter sharded on a matrix dim is reassembled with an all-to-all over its shard group: same-layout
    parameters are batched by the group size so each rank orthogonalizes one whole matrix, and the results go back
    with a second all-to-all. Parameters sharded only on batch dims (experts under EP) hold whole matrices and
    are orthogonalized locally. State and param group keys follow dion's Muon, so its checkpoints load.
    """

    def __init__(
        self,
        param_groups: list[dict],
        lr: float,
        newton_schulz_coefficients: Sequence[tuple[float, float, float]],
        mu: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        epsilon: float = 1e-8,
        matrix_partitions: dict[Tensor, tuple[int, ...]] | None = None,
        update_rms: float = 0.2,
        max_in_flight: int = 3,
    ):
        defaults = dict(
            lr=lr,
            mu=mu,
            nesterov=nesterov,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            algorithm="muon",
            step=0,
        )
        super().__init__(param_groups, defaults)
        self.matrix_partitions = matrix_partitions or {}
        for param, partitions in self.matrix_partitions.items():
            if sum(partitions) != param.size(-2):
                raise ValueError(f"Matrix partitions {partitions} do not sum to dim -2 of a {tuple(param.shape)} param")
        self.coefficients = tuple(map(tuple, newton_schulz_coefficients))
        self.update_rms = update_rms
        self.max_in_flight = max_in_flight
        self._row_scales: dict[Tensor, Tensor] = {}

    def chunks(self) -> list[Chunk]:
        """Every parameter with a gradient, exactly once, in a fixed order that is the same on every rank."""
        chunks = []
        next_owner: dict[dist.ProcessGroup | None, int] = {}
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if group["algorithm"] == "adamw":
                if params:
                    chunks.append(Chunk(group, params))
                continue
            batches: dict[tuple, list[Tensor]] = {}
            for p in params:
                placements = p.placements if isinstance(p, DTensor) else None
                batches.setdefault((p.shape, placements, p.dtype, self.matrix_partitions.get(p)), []).append(p)
            for batch in batches.values():
                _, process_group, size, _ = shard_group(batch[0])
                for i in range(0, len(batch), size):
                    # Owners continue round-robin across batches, so short batches do not all land on the first ranks
                    chunk = Chunk(group, batch[i : i + size], next_owner.get(process_group, 0))
                    next_owner[process_group] = (chunk.first_owner + len(chunk.params)) % size
                    chunks.append(chunk)
        return chunks

    def step_chunk(self, chunk: Chunk) -> None:
        """Update the chunk's parameters (and only them) on the current stream."""
        for _ in self._step_chunk(chunk):
            pass

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        # Round-robin over a few chunks so their all-to-alls overlap other chunks' Newton-Schulz
        running = []
        for chunk in self.chunks():
            running.append(self._step_chunk(chunk))
            while len(running) >= self.max_in_flight:
                running = [task for task in running if _advance(task)]
        while running:
            running = [task for task in running if _advance(task)]

    @torch.no_grad()
    def _step_chunk(self, chunk: Chunk) -> Generator[None, None, None]:
        if chunk.group["algorithm"] == "adamw":
            self._adamw(chunk)
            return
        group, params = chunk.group, chunk.params
        X = [_local(p) for p in params]
        M = [_local(self._state(p, "momentum")) for p in params]
        U = _update_momentum(M, [_local(p.grad) for p in params], torch.tensor(group["mu"]), group["nesterov"])
        O = yield from self.orthogonalize_chunk(params, U, chunk.first_owner)
        shard_dim, _, size, rank = shard_group(params[0])
        scale = self._update_scale(params[0], shard_dim, size, rank)
        decay = torch.tensor(1 - group["lr"] * group["weight_decay"])
        _apply_update(X, O, decay, scale * group["lr"])

    def orthogonalize_chunk(
        self, params: list[Tensor], U: list[Tensor], first_owner: int = 0
    ) -> Generator[None, None, list[Tensor]]:
        """Newton-Schulz of the chunk's local bfloat16 momentum shards ``U``, returned in the same layout. Yields while
        its all-to-alls are in flight."""
        partitions = self.matrix_partitions.get(params[0])
        shard_dim, process_group, size, rank = shard_group(params[0])
        if process_group is None:
            O = [orthogonalize(u, partitions, self.coefficients) for u in U]
        else:
            # Rank (first_owner + j) % size owns params[j]: it gathers every rank's shard of it, orthogonalizes, and
            # sends the shards back. FSDP shards like torch.chunk, so trailing ranks may hold fewer (or no) rows.
            full = params[0].size(shard_dim)
            step = math.ceil(full / size)
            sizes = [max(0, min(step, full - i * step)) for i in range(size)]
            owned_index = [(r - first_owner) % size for r in range(size)]
            owned = owned_index[rank] < len(params)
            empty = U[0].narrow(shard_dim, 0, 0)
            send = [U[j] if j < len(U) else empty for j in owned_index]
            recv = [U[0].new_empty(_resize(U[0].shape, shard_dim, n if owned else 0)) for n in sizes]
            work = dist.all_to_all(recv, send, group=process_group, async_op=True)
            yield
            work.wait()
            if owned:
                matrix = orthogonalize(torch.cat(recv, dim=shard_dim), partitions, self.coefficients)
                send = [s.contiguous() for s in matrix.split(sizes, dim=shard_dim)]
            else:
                send = recv
            O = [torch.empty_like(u) for u in U]
            recv = [O[j] if j < len(O) else empty for j in owned_index]
            work = dist.all_to_all(recv, send, group=process_group, async_op=True)
            yield
            work.wait()

        return O

    def _update_scale(self, param: Tensor, shard_dim: int | None, size: int, rank: int) -> Tensor:
        """Learning rate multiplier: a scalar, or one per local row of dim -2 for partitioned parameters."""
        fan_in = param.size(-1)
        partitions = self.matrix_partitions.get(param)
        if partitions is None:
            return torch.tensor(self.update_rms * math.sqrt(max(param.size(-2), fan_in)))
        if param not in self._row_scales:
            rows = torch.cat([torch.full((n,), self.update_rms * math.sqrt(max(n, fan_in))) for n in partitions])
            if shard_dim == param.ndim - 2:
                start = min(rank * math.ceil(len(rows) / size), len(rows))
                rows = rows[start : start + _local(param).size(-2)]
            self._row_scales[param] = rows.unsqueeze(-1).to(_local(param).device)
        return self._row_scales[param]

    def _adamw(self, chunk: Chunk) -> None:
        group = chunk.group
        group["step"] += 1
        X = [_local(p) for p in chunk.params]
        M = [_local(self._state(p, "momentum")) for p in chunk.params]
        V = [_local(self._state(p, "variance")) for p in chunk.params]
        G = [_local(p.grad).to(M[0].dtype) for p in chunk.params]
        beta1, beta2, step = group["beta1"], group["beta2"], group["step"]
        torch._foreach_lerp_(M, G, 1 - beta1)
        torch._foreach_lerp_(V, torch._foreach_mul(G, G), 1 - beta2)
        denom = torch._foreach_sqrt(V)
        torch._foreach_div_(denom, math.sqrt(1 - beta2**step))
        torch._foreach_add_(denom, group["epsilon"])
        torch._foreach_mul_(X, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_addcdiv_(X, M, denom, value=-group["lr"] / (1 - beta1**step))

    def _state(self, param: Tensor, key: str) -> Tensor:
        state = self.state[param]
        if key not in state:
            state[key] = torch.zeros_like(param)
        return state[key]


@torch.compile(fullgraph=True)
def _update_momentum(M: list[Tensor], G: list[Tensor], mu: Tensor, nesterov: bool) -> list[Tensor]:
    """M = mu * M + G in place; returns the bfloat16 Newton-Schulz input (Nesterov: mu * M + G, else M)."""
    G = [g.to(M[0].dtype) for g in G]
    torch._foreach_mul_(M, mu)
    torch._foreach_add_(M, G)
    U = torch._foreach_add(torch._foreach_mul(M, mu), G) if nesterov else M
    return [u.to(torch.bfloat16) for u in U]


@torch.compile(fullgraph=True)
def _apply_update(X: list[Tensor], O: list[Tensor], decay: Tensor, lr: Tensor) -> None:
    """Decoupled weight decay, then X -= lr * O (``lr`` a scalar or one per row of dim -2)."""
    for x, o in zip(X, O):
        x.mul_(decay).sub_(o * lr)


def shard_group(param: Tensor) -> tuple[int | None, dist.ProcessGroup | None, int, int]:
    """(matrix dim the parameter is sharded on, its shard group, group size, rank in group), or no group when every
    rank holds whole matrices: unsharded parameters, or parameters sharded only on batch dims (experts)."""
    if not isinstance(param, DTensor):
        return None, None, 1, 0
    matrix_shards = [
        (mesh_dim, placement.dim)
        for mesh_dim, placement in enumerate(param.placements)
        if isinstance(placement, (Shard, _StridedShard))
        and param.device_mesh.size(mesh_dim) > 1
        and placement.dim >= param.ndim - 2
    ]
    if not matrix_shards:
        return None, None, 1, 0
    if len(matrix_shards) > 1:
        raise NotImplementedError(f"Muon supports one sharded matrix dim, got placements {param.placements}")
    mesh_dim, dim = matrix_shards[0]
    if isinstance(param.placements[mesh_dim], _StridedShard):
        raise NotImplementedError(f"Muon does not support {param.placements[mesh_dim]} on a matrix dim")
    mesh = param.device_mesh
    return dim, mesh.get_group(mesh_dim), mesh.size(mesh_dim), mesh.get_local_rank(mesh_dim)


def _local(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _resize(shape: torch.Size, dim: int, size: int) -> list[int]:
    shape = list(shape)
    shape[dim] = size
    return shape


def _advance(task: Generator) -> bool:
    try:
        next(task)
        return True
    except StopIteration:
        return False
