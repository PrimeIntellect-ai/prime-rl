"""Muon (https://kellerjordan.github.io/posts/muon/) for FSDP2 and expert-parallel models.

A Muon parameter is updated with its orthogonalized Nesterov momentum, scaled to an RMS of ``update_rms``;
the parameters of Sinkhorn groups (embeddings) get Sinkhorn-balanced momentum, and every other parameter gets
AdamW. Orthogonalization needs whole matrices:

- A parameter whose matrix dims are not sharded (an expert stack sharded only along the expert dim, a
  replicated or a single-device parameter) is orthogonalized where it lives, as one batch of matrices.
- A parameter sharded along a matrix dim is gathered onto one rank of its shard group, orthogonalized there
  and scattered back, with uneven all-to-alls and no padding. Ranks own matrices by Newton-Schulz FLOPs, and
  the matrices move in buckets: bucket k+1 gathers while bucket k is orthogonalized.

Matrices packed into one parameter (runtime fusions, per-head partitions) are orthogonalized each on its own,
and every Newton-Schulz runs batched over all matrices of one shape. The state ("momentum", plus "variance"
for AdamW) and the param group keys are those of dion's Muon, so its checkpoints load as they are.
"""

import functools
import heapq
import importlib.util
import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim import Optimizer

from prime_rl.utils.logger import get_logger

# Whole matrices each rank receives per bucket. Bounds the step's transient memory to a few buckets.
BUCKET_BYTES = 256 * 2**20
# CTAs per NCCL collective on Muon's communicators
MAX_CTAS = 4


@dataclass(frozen=True)
class NewtonSchulz:
    """Odd polynomial iterations X <- aX + (bA + cA²)X with A = XXᵀ, one (a, b, c) per iteration, on
    X = G / (safety·‖G‖_F + epsilon), normalized in bf16 and iterated in ``dtype``."""

    coefficients: tuple[tuple[float, float, float], ...]
    safety: float = 1.0
    dtype: torch.dtype = torch.bfloat16


# Polar Express (arXiv 2505.16932), divided by the safety factor except for the last polynomial, which repeats
POLAR_EXPRESS = tuple(
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for a, b, c in (
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    )
) + ((1.875, -1.25, 0.375),)

NEWTON_SCHULZ = {
    # Per-step quintic coefficients
    "default": NewtonSchulz(
        (
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        )
    ),
    # DeepSeek-V4: 8 fast-converging steps, then 2 that pin the singular values at 1
    "deepseek_v4": NewtonSchulz(((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2),
    "polar_express": NewtonSchulz(POLAR_EXPRESS[:5], safety=1.01),
    # Step 3.5 Flash: 6 Polar Express steps in fp16 (bf16 spiked)
    "step3p5": NewtonSchulz(POLAR_EXPRESS[:6], safety=1.01, dtype=torch.float16),
}


@functools.cache
def _quack_gemm_symmetric():
    """QuACK's symmetric GEMM (computes one triangle and mirrors it), or None without QuACK or before SM80."""
    if importlib.util.find_spec("quack") is None or torch.cuda.get_device_capability()[0] < 8:
        return None
    from quack.gemm_interface import gemm_symmetric

    return gemm_symmetric


@torch.compile(dynamic=True)
def _normalize(X: Tensor, safety: float, epsilon: float) -> Tensor:
    """X / (safety·‖X‖ + epsilon) per row of X [B, m·n], as one fused reduction. Flat rows keep the compiled
    graphs few: one per device and batch size class."""
    return X / (X.norm(dim=-1, keepdim=True) * safety + epsilon)


def newton_schulz(G: Tensor, schedule: NewtonSchulz, epsilon: float) -> Tensor:
    """Orthogonalize every matrix of the batch ``G`` [B, m, n]. Returns [B, m, n] in ``schedule.dtype``.

    The Gram matrix A is the smaller of XXᵀ and XᵀX. For tall matrices this iterates X <- aX + X(bA + cA²) on
    A = XᵀX, the transpose of the wide iteration, without transposing X."""
    X = G.to(torch.bfloat16).reshape(G.size(0), -1)
    X = _normalize(X, schedule.safety, epsilon).reshape(G.shape).to(schedule.dtype)
    tall = X.size(-2) > X.size(-1)
    # QuACK needs 16-byte aligned rows
    gemm_symmetric = _quack_gemm_symmetric() if X.is_cuda and X.size(-1) % 8 == 0 and X.size(-2) % 8 == 0 else None
    gram = gemm_symmetric or torch.bmm
    for a, b, c in schedule.coefficients:
        A = gram(X.mT, X) if tall else gram(X, X.mT)
        if gemm_symmetric is not None:
            B = gemm_symmetric(A, A, C=A, alpha=c, beta=b)
        else:
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, X, B, beta=a) if tall else torch.baddbmm(X, B, X, beta=a)
    return X


def _batch(stacks: list[Tensor]) -> Tensor:
    """Concatenate stacks of matrices [B_i, m, n], as a view when they lie back to back in memory."""
    first = stacks[0]
    adjacent = all(
        x.is_contiguous() and x.untyped_storage().data_ptr() == first.untyped_storage().data_ptr() for x in stacks
    )
    for previous, x in zip(stacks, stacks[1:]):
        adjacent = adjacent and x.storage_offset() == previous.storage_offset() + previous.numel()
    if not adjacent:
        return torch.cat(stacks)
    size = (sum(x.size(0) for x in stacks), *first.shape[1:])
    return first.as_strided(size, (first[0].numel(), first.size(-1), 1))


@functools.cache
def _segments(partitions: tuple[int, ...]) -> tuple[tuple[int, tuple[int, ...], int], ...]:
    """Cut row partitions into runs of a repeated pattern: (first row, pattern, repeats). Each position of a
    pattern is then one strided batch of matrices, e.g. the heads of queries and keys before a whole value
    projection, or MLA's [W_UK; W_UV] per head."""
    segments, i, row = [], 0, 0
    while i < len(partitions):
        period, repeats = 1, 1
        for p in range(1, (len(partitions) - i) // 2 + 1):
            pattern, r = partitions[i : i + p], 1
            while partitions[i + r * p : i + (r + 1) * p] == pattern:
                r += 1
            if r > 1 and p * r > period * repeats:
                period, repeats = p, r
        pattern = partitions[i : i + period]
        segments.append((row, pattern, repeats))
        i += period * repeats
        row += sum(pattern) * repeats
    return tuple(segments)


def orthogonalize(
    matrices: Sequence[Tensor],
    partitions: Sequence[tuple[int, ...]],
    schedule: NewtonSchulz,
    epsilons: Sequence[float],
) -> list[Tensor]:
    """Orthogonalize tensors [..., M, n] whose dim -2 packs matrices of the given row counts, one Newton-Schulz
    per matrix shape over all of them. Leading dims are batches of matrices."""
    outputs: list[Tensor | None] = [None] * len(matrices)
    classes = defaultdict(list)
    for i, (x, parts, epsilon) in enumerate(zip(matrices, partitions, epsilons)):
        x = x.reshape(-1, *x.shape[-2:])
        n = x.size(-1)
        if len(set(parts)) == 1:
            # Equal partitions are a reshape: [B, P·r, n] -> [B·P, r, n]
            classes[parts[0], n, epsilon].append((i, None, x.reshape(-1, parts[0], n)))
            continue
        outputs[i] = torch.empty(matrices[i].shape, dtype=schedule.dtype, device=x.device)
        output = outputs[i].view(x.shape)
        for start, pattern, repeats in _segments(tuple(parts)):
            # [B, repeats, sum(pattern), n]: each position of the pattern is one batch [B·repeats, rows, n]
            source, target = (
                t.narrow(-2, start, sum(pattern) * repeats).unflatten(-2, (repeats, sum(pattern))) for t in (x, output)
            )
            for offset, rows in zip(_offsets(pattern), pattern):
                part = source.narrow(-2, offset, rows).reshape(-1, rows, n)
                classes[rows, n, epsilon].append((i, target.narrow(-2, offset, rows), part))
    for (_, _, epsilon), members in classes.items():
        Y = newton_schulz(_batch([x for _, _, x in members]), schedule, epsilon)
        for (i, target, x), y in zip(members, Y.split([x.size(0) for _, _, x in members])):
            if target is None:
                outputs[i] = y.reshape(matrices[i].shape)
            else:
                target.copy_(y.view(target.shape))
    return outputs


def plan_buckets(
    costs: Sequence[float], nbytes: Sequence[int], world_size: int, bucket_bytes: int
) -> list[list[tuple[int, int]]]:
    """Cut the matrices, in order, into buckets of about ``bucket_bytes`` per rank, and give each matrix an owner
    rank: (matrix, owner) pairs per bucket. Within a bucket, owners balance the Newton-Schulz cost, longest
    first onto the least loaded rank, so every bucket keeps all ranks equally busy. Keeping the order keeps a
    bucket to a few consecutive layers."""
    buckets, bucket, size = [], [], 0
    for i, n in enumerate(nbytes):
        if bucket and size + n > world_size * bucket_bytes:
            buckets.append(bucket)
            bucket, size = [], 0
        bucket.append(i)
        size += n
    buckets += [bucket] * bool(bucket)
    owned = []
    for bucket in buckets:
        loads = [(0.0, rank) for rank in range(world_size)]
        pairs = []
        for i in sorted(bucket, key=lambda i: (-costs[i], i)):
            load, rank = heapq.heappop(loads)
            pairs.append((i, rank))
            heapq.heappush(loads, (load + costs[i], rank))
        owned.append(sorted(pairs))
    return owned


def newton_schulz_cost(shape: Sequence[int], partitions: Sequence[int]) -> float:
    """FLOPs of one Newton-Schulz iteration over a tensor [..., M, n] packing matrices of ``partitions`` rows."""
    batch = math.prod(shape[:-2])
    return batch * sum(
        4 * min(rows, shape[-1]) ** 2 * max(rows, shape[-1]) + 2 * min(rows, shape[-1]) ** 3 for rows in partitions
    )


def _offsets(sizes: Sequence[int]) -> list[int]:
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    return offsets


def _local(tensor: Tensor) -> Tensor:
    # The attribute, not to_local(): that is an autograd function, and the step reads thousands of local tensors
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _shape_class(matrix: "_Matrix") -> tuple:
    """Matrices with equal keys orthogonalize as one batch (see orthogonalize)."""
    return _local(matrix.param).shape[-1], matrix.partitions


def _num_shards(param: Tensor) -> int:
    if not isinstance(param, DTensor):
        return 1
    mesh = param.device_mesh
    return math.prod(mesh.size(d) for d, p in enumerate(param.placements) if isinstance(p, (Shard, _StridedShard)))


@dataclass(frozen=True)
class _Sharding:
    """How a parameter's shard group splits one of its matrix dims, in torch.chunk layout."""

    dim: int
    group: ProcessGroup
    rank: int
    sizes: tuple[int, ...]

    @property
    def offsets(self) -> list[int]:
        return _offsets(self.sizes)


_MUON_GROUPS: dict[tuple[tuple[int, ...], ...], ProcessGroup] = {}


def _muon_group(mesh: DeviceMesh, mesh_dim: int) -> ProcessGroup:
    """This rank's group along ``mesh_dim``, as a communicator of Muon's own: on FSDP's, Muon's all-to-alls
    would queue behind (or ahead of) FSDP's all-gathers. Every rank creates every group, in the same order."""
    layout = tuple(map(tuple, mesh.mesh.movedim(mesh_dim, -1).reshape(-1, mesh.size(mesh_dim)).tolist()))
    if layout not in _MUON_GROUPS:
        options = None
        if dist.get_backend() == "nccl":
            # Muon moves little data: few channels keep the communicator's buffers small (2.8 GiB per rank on
            # 8 GPUs with NCCL's default) and leave the SMs to Newton-Schulz
            options = dist.ProcessGroupNCCL.Options()
            options.config.max_ctas = MAX_CTAS
        groups = [list(ranks) for ranks in layout]
        _MUON_GROUPS[layout], _ = dist.new_subgroups_by_enumeration(groups, pg_options=options)
        _warmup_all_to_all(_MUON_GROUPS[layout])
    return _MUON_GROUPS[layout]


def _matrix_sharding(param: Tensor) -> _Sharding | None:
    """The sharding of ``param``'s matrix dims, or None when every rank holds whole matrices."""
    if not isinstance(param, DTensor):
        return None
    mesh = param.device_mesh
    shards = [
        (mesh_dim, placement)
        for mesh_dim, placement in enumerate(param.placements)
        if isinstance(placement, (Shard, _StridedShard)) and mesh.size(mesh_dim) > 1 and placement.dim >= param.ndim - 2
    ]
    if not shards:
        return None
    if len(shards) > 1 or isinstance(shards[0][1], _StridedShard):
        raise NotImplementedError(f"Muon gathers matrices over one plain Shard, got placements {param.placements}")
    mesh_dim, placement = shards[0]
    group = _muon_group(mesh, mesh_dim)
    # The all-to-all splits are in group rank order, DTensor's shards in mesh order
    coordinate = list(mesh.get_coordinate())
    coordinate[mesh_dim] = slice(None)
    if mesh.mesh[tuple(coordinate)].tolist() != dist.get_process_group_ranks(group):
        raise ValueError(f"Muon needs the shard ranks of {param.placements} in ascending order")
    size, world_size = param.shape[placement.dim], mesh.size(mesh_dim)
    chunk = -(-size // world_size)
    sizes = tuple(max(0, min(chunk, size - rank * chunk)) for rank in range(world_size))
    rank = mesh.get_local_rank(mesh_dim)
    assert param.to_local().shape[placement.dim] == sizes[rank]
    return _Sharding(placement.dim, group, rank, sizes)


def _shard_groups(param: Tensor, dim: int) -> tuple[ProcessGroup, ...]:
    """Muon's communicators over the mesh dims that shard ``dim`` of ``param``."""
    if not isinstance(param, DTensor):
        return ()
    mesh = param.device_mesh
    return tuple(
        _muon_group(mesh, mesh_dim)
        for mesh_dim, placement in enumerate(param.placements)
        if isinstance(placement, (Shard, _StridedShard)) and placement.dim == dim and mesh.size(mesh_dim) > 1
    )


def _all_reduce(tensor: Tensor, groups: Sequence[ProcessGroup]) -> Tensor:
    for group in groups:
        dist.all_reduce(tensor, group=group)
    return tensor


def _norms(U: Tensor, dim: int, groups: Sequence[ProcessGroup]) -> Tensor:
    """L2 norms of the local shard ``U`` along ``dim``, summed over the communicators that shard that dim."""
    return _all_reduce(U.square().sum(dim=dim), groups).sqrt()


@dataclass(frozen=True)
class _Sinkhorn:
    """A Sinkhorn parameter [m, n] and the communicators that shard its rows and its columns."""

    param: Tensor
    group: int  # index into param_groups
    row_shards: tuple[ProcessGroup, ...]
    column_shards: tuple[ProcessGroup, ...]


@dataclass(frozen=True)
class _Matrix:
    """A Muon parameter: its matrix partitions along dim -2, sharding and update scale."""

    param: Tensor
    group: int  # index into param_groups, whose dicts load_state_dict replaces
    partitions: tuple[int, ...]
    sharding: _Sharding | None
    lr_ratio: float | Tensor  # per local row for packed parameters

    def local_shape(self, rank: int) -> list[int]:
        """The shape of ``rank``'s shard."""
        shape = list(_local(self.param).shape)
        shape[self.sharding.dim] = self.sharding.sizes[rank]
        return shape

    def whole_shape(self) -> list[int]:
        """The shape of the gathered matrices: whole along the sharded dim, local along batch dims."""
        shape = list(_local(self.param).shape)
        shape[self.sharding.dim] = sum(self.sharding.sizes)
        return shape


@dataclass(frozen=True)
class _Bucket:
    """Sharded matrices that move together. Shards are laid out by owner rank on the sender's side (``sent``)
    and by sender rank on the owner's side (``owned``), the same way in both directions."""

    group: ProcessGroup
    sent: list[_Matrix]
    owned: list[_Matrix]
    send_splits: list[int]
    recv_splits: list[int]


@dataclass(frozen=True)
class _Plan:
    """The work of one step, in units that update together: buckets of sharded matrices, chunks of local
    matrices orthogonalized as one batch, AdamW groups, each one compiled update, and Sinkhorn parameters."""

    buckets: list[_Bucket]
    local: list[list[_Matrix]]
    adamw: list[tuple[int, list[Tensor]]]  # (param group index, params)
    sinkhorn: list[_Sinkhorn]

    @property
    def units(self) -> list[list[Tensor]]:
        return (
            [[m.param for m in bucket.sent] for bucket in self.buckets]
            + [[m.param for m in chunk] for chunk in self.local]
            + [params for _, params in self.adamw]
            + [[s.param] for s in self.sinkhorn]
        )

    def restrict(self, params: Iterable[Tensor]) -> "_Plan":
        """The units of ``params``, a union of whole units."""
        selected = set(map(id, params))
        return _Plan(
            buckets=[bucket for bucket in self.buckets if id(bucket.sent[0].param) in selected],
            local=[chunk for chunk in self.local if id(chunk[0].param) in selected],
            adamw=[(group, params) for group, params in self.adamw if id(params[0]) in selected],
            sinkhorn=[s for s in self.sinkhorn if id(s.param) in selected],
        )


@dataclass
class _Step:
    """One Muon step as partial updates (``SubsetStep``): the hyperparameters and gradients are captured when
    the step begins, so a unit updated later updates as it would have then."""

    optimizer: "Muon"
    plan: _Plan
    hyperparameters: list[dict]  # per param group
    grads: dict[Tensor, Tensor]

    @property
    def units(self) -> list[list[Tensor]]:
        return self.plan.units

    def step_params(self, params: Collection[Tensor]) -> None:
        self.optimizer._run(self.plan.restrict(params), self.hyperparameters, self.grads)


# The compiled helpers take one tensor each, flattened where the arithmetic allows, so they compile few graphs. They
# have no graph breaks, but are not fullgraph: past dynamo's recompile limit they run eagerly instead of raising.
@torch.compile(dynamic=True)
def _momentum_(M: Tensor, G: Tensor, U: Tensor, mu: Tensor, nesterov: bool) -> None:
    """M <- μM + G and U <- μM + G (Nesterov) or M."""
    G = G.to(M.dtype)
    M.mul_(mu).add_(G)
    U.copy_(M * mu + G if nesterov else M)


@torch.compile(dynamic=True)
def _muon_update_(X: Tensor, U: Tensor, lr: Tensor, adjusted_lr: Tensor, weight_decay: Tensor) -> None:
    X.mul_(1 - lr * weight_decay)
    X.sub_(U * adjusted_lr)


@torch.compile(dynamic=True)
def _adamw_(
    X: list[Tensor],
    G: list[Tensor],
    M: list[Tensor],
    V: list[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: Tensor,
    epsilon: Tensor,
) -> None:
    """dion's AdamW, op for op: X <- (1 - lr·λ)X - lr/(1 - β1ᵗ) · M / (√V / √(1 - β2ᵗ) + ε)."""
    G = [g.to(M[0].dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * len(X))
    torch._foreach_lerp_(V, [g.to(V[0].dtype) for g in torch._foreach_mul(G, G)], [1 - beta2] * len(X))
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, (1 - beta2**step).sqrt())
    torch._foreach_add_(denom, [epsilon] * len(X))
    torch._foreach_mul_(X, 1 - lr * weight_decay)
    update = torch._foreach_div(M, denom)
    torch._foreach_mul_(update, lr / (1 - beta1**step))
    torch._foreach_sub_(X, update)


def _hyperparameters(group: dict) -> dict:
    if group["algorithm"] == "sinkhorn":
        # Sinkhorn runs eagerly
        return {key: group[key] for key in ("lr", "lr_scale", "mu", "epsilon", "tau", "iters")}
    # 0-d CPU tensors, so the compiled updates do not recompile when the scheduler changes the lr
    keys = ("lr", "mu", "beta1", "beta2", "weight_decay", "epsilon", "step")
    return {key: torch.tensor(group[key]) for key in keys} | {"nesterov": group["nesterov"]}


def _warmup_all_to_all(group: ProcessGroup) -> None:
    """Establish NCCL peer connections before the first optimizer step.

    This is a correctness workaround, not an optional performance warm-up: multi-node GLM-Air training can
    deadlock on Muon's first bulk all-to-all without it.
    """
    size = dist.get_world_size(group)
    if size <= 1 or dist.get_backend(group) != "nccl":
        return
    get_logger().info(f"Warming Muon all-to-all connections on group {group.group_name} ({size} ranks)")
    # A bulk transfer, like the step's, rather than a scalar collective that may leave bulk channels unset
    inputs = torch.zeros(size * 3 * 2**20, dtype=torch.bfloat16, device="cuda")
    dist.all_to_all_single(torch.empty_like(inputs), inputs, group=group)
    torch.cuda.synchronize()


class Muon(Optimizer):
    """Muon for the param groups with ``algorithm="muon"`` (matrices, ndim >= 2), AdamW for those with
    ``algorithm="adamw"`` and Sinkhorn for those with ``algorithm="sinkhorn"`` (matrices m x n).

    Muon: M <- μM + G, U <- orthogonalize(μM + G) (Nesterov) or orthogonalize(M), and
    X <- (1 - lr·λ)X - lr·update_rms·√max(m, n)·U for every matrix m x n. ``matrix_partitions`` maps a
    parameter to the row counts of the matrices packed along its dim -2, each orthogonalized and scaled as its
    own matrix.

    Sinkhorn (DeepSeek-V4.1, Algorithm 1): M <- μM + (1 - μ)G and U = μM + (1 - μ)G, with the rows whose norm is
    at most ``tau`` times the mean row norm zeroed, then ``iters`` alternating row/column L2 normalizations, and
    X <- X - lr·lr_scale·update_rms·√n·U, without weight decay.

    ``step()`` runs ``begin_subset_step()`` and updates all of its units, so an offloaded step that updates
    one chunk of units at a time (``SupportsSubsetStep``) is bitwise the same step. Units are identical on
    every rank, and the step never synchronizes with the host.
    """

    def __init__(
        self,
        params: Iterable[dict],
        *,
        lr: float,
        mu: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = True,
        update_rms: float = 0.2,
        newton_schulz: str = "default",
        matrix_partitions: Mapping[Tensor, Sequence[int]] | None = None,
        bucket_bytes: int = BUCKET_BYTES,
    ):
        defaults = dict(
            algorithm="muon",
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov,
            step=0,
        )
        super().__init__(params, defaults)
        self.update_rms = update_rms
        self.schedule = NEWTON_SCHULZ[newton_schulz]
        self.bucket_bytes = bucket_bytes
        self._plans: dict[tuple[int, ...], _Plan] = {}

        partitions = {param: tuple(sizes) for param, sizes in (matrix_partitions or {}).items()}
        self._matrices: dict[Tensor, _Matrix] = {}
        self._sinkhorn: dict[Tensor, _Sinkhorn] = {}
        for index, group in enumerate(self.param_groups):
            if group["algorithm"] not in ("muon", "adamw", "sinkhorn"):
                raise ValueError(f"Unknown algorithm {group['algorithm']}")
            if group["algorithm"] == "muon":
                for param in group["params"]:
                    self._matrices[param] = self._matrix(param, index, partitions.pop(param, (param.size(-2),)))
            if group["algorithm"] == "sinkhorn":
                for param in group["params"]:
                    if param.ndim != 2:
                        raise ValueError(f"Sinkhorn parameters must be matrices, got shape {tuple(param.shape)}")
                    self._sinkhorn[param] = _Sinkhorn(param, index, _shard_groups(param, 0), _shard_groups(param, 1))
        if partitions:
            raise ValueError("matrix_partitions has parameters that are not Muon parameters of this optimizer")

    def _matrix(self, param: Tensor, group: int, partitions: tuple[int, ...]) -> _Matrix:
        if param.ndim < 2:
            raise ValueError(f"Muon parameters must be matrices, got shape {tuple(param.shape)}")
        if sum(partitions) != param.size(-2) or min(partitions) <= 0:
            raise ValueError(f"Partitions {partitions} do not split dim -2 of shape {tuple(param.shape)}")
        sharding = _matrix_sharding(param)
        ratios = [self.update_rms * math.sqrt(max(rows, param.size(-1))) for rows in partitions]
        lr_ratio = ratios[0]
        if len(partitions) > 1:
            # Per row, on the device, so the step scales packed matrices without a host-to-device copy
            rows = torch.tensor(ratios).repeat_interleave(torch.tensor(partitions))
            if sharding is not None and sharding.dim == param.ndim - 2:
                rows = rows.narrow(0, sharding.offsets[sharding.rank], sharding.sizes[sharding.rank])
            lr_ratio = rows.unsqueeze(-1).to(_local(param).device)
        return _Matrix(param, group, partitions, sharding, lr_ratio)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        step = self.begin_subset_step()
        step.step_params([param for unit in step.units for param in unit])
        return loss

    def begin_subset_step(self) -> _Step:
        """Start a step: advance the step counters and capture the hyperparameters and gradients. The returned
        step updates any union of its ``units`` with ``step_params``."""
        for group in self.param_groups:
            group["step"] += 1
        params = [param for group in self.param_groups for param in group["params"] if param.grad is not None]
        key = tuple(map(id, params))
        if key not in self._plans:
            self._plans[key] = self._plan(params)
        hyperparameters = [_hyperparameters(group) for group in self.param_groups]
        return _Step(self, self._plans[key], hyperparameters, {param: _local(param.grad) for param in params})

    def _plan(self, params: list[Tensor]) -> _Plan:
        matrices = [self._matrices[param] for param in params if param in self._matrices]
        by_group = defaultdict(list)
        for matrix in matrices:
            if matrix.sharding is not None:
                by_group[matrix.sharding.group].append(matrix)
        buckets = []
        for group, members in by_group.items():
            world_size, rank = dist.get_world_size(group), dist.get_rank(group)
            costs = [newton_schulz_cost(m.param.shape, m.partitions) for m in members]
            nbytes = [2 * m.param.numel() for m in members]
            for pairs in plan_buckets(costs, nbytes, world_size, self.bucket_bytes):
                # Every owner's matrices grouped by shape, in the same order on the sender's and owner's side
                owned = [
                    sorted((members[i] for i, owner in pairs if owner == r), key=_shape_class)
                    for r in range(world_size)
                ]
                send_splits = [sum(math.prod(m.local_shape(rank)) for m in owned[r]) for r in range(world_size)]
                recv_splits = [sum(math.prod(m.local_shape(q)) for m in owned[rank]) for q in range(world_size)]
                sent = [m for r in range(world_size) for m in owned[r]]
                buckets.append(_Bucket(group, sent, owned[rank], send_splits, recv_splits))

        # Chunked by shard sizes that are the same on every rank, so that every rank has the same units
        local, chunk, chunk_bytes = [], [], 0
        for matrix in (m for m in matrices if m.sharding is None):
            nbytes = 2 * matrix.param.numel() // _num_shards(matrix.param)
            if chunk and chunk_bytes + nbytes > self.bucket_bytes:
                local.append(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append(matrix)
            chunk_bytes += nbytes
        local += [chunk] * bool(chunk)

        selected = set(map(id, params))
        adamw = [
            (index, [param for param in group["params"] if id(param) in selected])
            for index, group in enumerate(self.param_groups)
            if group["algorithm"] == "adamw"
        ]
        sinkhorn = [self._sinkhorn[param] for param in params if param in self._sinkhorn]
        return _Plan(buckets, local, [(group, params) for group, params in adamw if params], sinkhorn)

    @torch.no_grad()
    def _run(self, plan: _Plan, hyperparameters: list[dict], grads: dict[Tensor, Tensor]) -> None:
        # Bucket k+1 gathers and bucket k-1 scatters while bucket k is orthogonalized. Local chunks fill in.
        slots = len(plan.buckets) + 1
        local = [plan.local[i * len(plan.local) // slots : (i + 1) * len(plan.local) // slots] for i in range(slots)]
        gathering = self._gather(plan.buckets[0], hyperparameters, grads) if plan.buckets else None
        scattering = None
        for i, bucket in enumerate(plan.buckets):
            gathered = gathering
            if i + 1 < len(plan.buckets):
                gathering = self._gather(plan.buckets[i + 1], hyperparameters, grads)
            for chunk in local[i]:
                self._update_local(chunk, hyperparameters, grads)
            orthogonalized = self._orthogonalize_and_scatter(bucket, *gathered)
            if scattering is not None:
                self._apply(*scattering, hyperparameters)
            scattering = (bucket, *orthogonalized)
        for chunk in local[-1]:
            self._update_local(chunk, hyperparameters, grads)
        if scattering is not None:
            self._apply(*scattering, hyperparameters)

        for group, params in plan.adamw:
            self._adamw(params, hyperparameters[group], grads)
        for sinkhorn in plan.sinkhorn:
            self._sinkhorn_update(sinkhorn, hyperparameters[sinkhorn.group], grads)

    def _state(self, param: Tensor, *keys: str) -> list[Tensor]:
        """State tensors, read at every use: an offloaded step swaps them for GPU copies between updates."""
        state = self.state[param]
        for key in keys:
            if key not in state:
                state[key] = torch.zeros_like(param)
        return [_local(state[key]) for key in keys]

    def _momentum(
        self, matrices: list[_Matrix], updates: list[Tensor], hyperparameters: list[dict], grads: dict[Tensor, Tensor]
    ) -> None:
        """M <- μM + G, and the input of the orthogonalization into ``updates``."""
        for matrix, U in zip(matrices, updates):
            if U.numel():
                (M,) = self._state(matrix.param, "momentum")
                h = hyperparameters[matrix.group]
                _momentum_(M.view(-1), grads[matrix.param].reshape(-1), U.view(-1), h["mu"], h["nesterov"])

    def _update(self, matrices: list[_Matrix], updates: list[Tensor], hyperparameters: list[dict]) -> None:
        for matrix, U in zip(matrices, updates):
            if U.numel():
                X, h = _local(matrix.param), hyperparameters[matrix.group]
                if isinstance(matrix.lr_ratio, Tensor):  # per row
                    _muon_update_(X, U, h["lr"], matrix.lr_ratio * h["lr"], h["weight_decay"])
                else:
                    _muon_update_(X.view(-1), U.reshape(-1), h["lr"], matrix.lr_ratio * h["lr"], h["weight_decay"])

    def _update_local(self, chunk: list[_Matrix], hyperparameters: list[dict], grads: dict[Tensor, Tensor]) -> None:
        """Orthogonalize matrices that every rank holds whole, as one batch per matrix shape. The chunk's inputs
        share one buffer in which matrices of one shape lie back to back, so each batch is a view of it."""
        matrices = sorted((m for m in chunk if _local(m.param).numel()), key=_shape_class)
        if not matrices:
            return
        device = _local(matrices[0].param).device
        buffer = torch.empty(sum(_local(m.param).numel() for m in matrices), dtype=torch.bfloat16, device=device)
        updates = self._shard_views(buffer, matrices)
        self._momentum(matrices, updates, hyperparameters, grads)
        epsilons = [self.param_groups[m.group]["epsilon"] for m in matrices]
        updates = orthogonalize(updates, [m.partitions for m in matrices], self.schedule, epsilons)
        self._update(matrices, updates, hyperparameters)

    def _gather(
        self, bucket: _Bucket, hyperparameters: list[dict], grads: dict[Tensor, Tensor]
    ) -> tuple[dist.Work, Tensor, Tensor]:
        """Momentum into this rank's shards, sent to their owners."""
        device = _local(bucket.sent[0].param).device
        send = torch.empty(sum(bucket.send_splits), dtype=torch.bfloat16, device=device)
        self._momentum(bucket.sent, self._shard_views(send, bucket.sent), hyperparameters, grads)
        recv = send.new_empty(sum(bucket.recv_splits))
        work = dist.all_to_all_single(
            recv, send, bucket.recv_splits, bucket.send_splits, group=bucket.group, async_op=True
        )
        return work, recv, send

    def _orthogonalize_and_scatter(
        self, bucket: _Bucket, work: dist.Work, recv: Tensor, send: Tensor
    ) -> tuple[dist.Work, Tensor, Tensor]:
        """Orthogonalize the matrices this rank owns and send each rank its shards back."""
        work.wait()
        world_size = len(bucket.recv_splits)
        pieces = iter(recv.split([math.prod(m.local_shape(q)) for q in range(world_size) for m in bucket.owned]))
        shards = [[next(pieces) for _ in bucket.owned] for _ in range(world_size)]
        # Whole matrices back to back, grouped by shape (see _plan), so each Newton-Schulz batch is a view
        shapes = [m.whole_shape() for m in bucket.owned]
        buffer = recv.new_empty(sum(math.prod(shape) for shape in shapes))
        matrices = [piece.view(shape) for piece, shape in zip(buffer.split([math.prod(s) for s in shapes]), shapes)]
        for j, (m, matrix) in enumerate(zip(bucket.owned, matrices)):
            torch.cat([shards[q][j].view(m.local_shape(q)) for q in range(world_size)], dim=m.sharding.dim, out=matrix)
        epsilons = [self.param_groups[m.group]["epsilon"] for m in bucket.owned]
        outputs = orthogonalize(matrices, [m.partitions for m in bucket.owned], self.schedule, epsilons)
        scattered = [
            output.narrow(m.sharding.dim, m.sharding.offsets[q], m.sharding.sizes[q]).reshape(-1)
            for q in range(world_size)
            for output, m in zip(outputs, bucket.owned)
        ]
        send = torch.cat(scattered).to(torch.bfloat16) if scattered else recv.new_empty(0)
        recv = send.new_empty(sum(bucket.send_splits))
        work = dist.all_to_all_single(
            recv, send, bucket.send_splits, bucket.recv_splits, group=bucket.group, async_op=True
        )
        return work, recv, send

    def _apply(self, bucket: _Bucket, work: dist.Work, recv: Tensor, send: Tensor, hyperparameters: list[dict]) -> None:
        work.wait()
        self._update(bucket.sent, self._shard_views(recv, bucket.sent), hyperparameters)

    @staticmethod
    def _shard_views(buffer: Tensor, matrices: list[_Matrix]) -> list[Tensor]:
        shapes = [_local(m.param).shape for m in matrices]
        return [piece.view(shape) for piece, shape in zip(buffer.split([math.prod(s) for s in shapes]), shapes)]

    def _adamw(self, params: list[Tensor], h: dict, grads: dict[Tensor, Tensor]) -> None:
        """One AdamW over the whole group, as dion: its compiled update rounds differently for other lists."""
        M, V = zip(*(self._state(param, "momentum", "variance") for param in params))
        X, G = [_local(param) for param in params], [grads[param] for param in params]
        _adamw_(X, G, list(M), list(V), h["lr"], h["beta1"], h["beta2"], h["weight_decay"], h["step"], h["epsilon"])

    def _sinkhorn_update(self, sinkhorn: _Sinkhorn, h: dict, grads: dict[Tensor, Tensor]) -> None:
        """Sinkhorn on the local shard: row norms all-reduce over the shards of the columns, column norms and the
        mean row norm over the shards of the rows."""
        param, rows, columns = sinkhorn.param, sinkhorn.row_shards, sinkhorn.column_shards
        X, (M,) = _local(param), self._state(param, "momentum")
        G = grads[param].to(M.dtype)
        M.lerp_(G, 1 - h["mu"])
        U = G.lerp(M, h["mu"]).float()
        row_norms = _norms(U, 1, columns)
        mean_row_norm = _all_reduce(row_norms.sum(), rows) / param.shape[0]
        U = U * (row_norms > h["tau"] * mean_row_norm).unsqueeze(1)
        for k in range(h["iters"]):
            if k % 2 == 0:
                U = U / (_norms(U, 1, columns) + h["epsilon"]).unsqueeze(1)
            else:
                U = U / (_norms(U, 0, rows) + h["epsilon"]).unsqueeze(0)
        X.sub_(U.to(X.dtype), alpha=h["lr"] * h["lr_scale"] * self.update_rms * math.sqrt(param.shape[1]))
