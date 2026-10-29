import math
from collections import defaultdict
from itertools import chain

import dion
import torch
import torch.distributed as dist
from dion.opt_utils import AsyncRuntime, AsyncTask, create_param_batches, pad_batch, to_local
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

DEEPSEEK_V4_NEWTON_SCHULZ = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2

# Polar Express (arXiv 2505.16932), divided by the safety factor 1.01 except for the last polynomial, which repeats
POLAR_EXPRESS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for a, b, c in [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    ]
] + [(1.875, -1.25, 0.375)]


def _odd_polynomials(G: Tensor, coefficients: list, dtype: torch.dtype, epsilon: float, safety: float = 1.0) -> Tensor:
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) * safety + epsilon)).to(dtype)
    for a, b, c in coefficients:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def newton_schulz_deepseek_v4(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    """DeepSeek-V4's hybrid Newton-Schulz: 8 fast-converging steps, then 2 steps that pin the singular values at 1."""
    return _odd_polynomials(G, DEEPSEEK_V4_NEWTON_SCHULZ, torch.bfloat16, epsilon)


def polar_express(G: Tensor, epsilon: float = 1e-7, steps: int = 5, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Polar Express: ``steps`` of its per-step optimal polynomials on X / (1.01 ||X||_F), normalized in bf16 and
    iterated in ``dtype``."""
    coefficients = POLAR_EXPRESS[:steps] + POLAR_EXPRESS[-1:] * (steps - len(POLAR_EXPRESS))
    return _odd_polynomials(G, coefficients, dtype, epsilon, safety=1.01)


def polar_express_step3p5(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Step 3.5 Flash: 6 Polar Express steps in fp16 (bf16 spiked)."""
    return polar_express(G, epsilon, steps=6, dtype=torch.float16)


@torch.compile(fullgraph=True)
def _dimuon_polar_express(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    return polar_express(G, 1e-7, steps=5, dtype=torch.float16)


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


def _sinkhorn_task(param: Tensor, state: dict, group: dict, update_rms: float):
    sinkhorn_update(
        to_local(param),
        to_local(param.grad),
        to_local(state["momentum"]),
        lr=group["lr"] * group["lr_scale"],
        momentum=group["mu"],
        update_rms=update_rms,
        tau=group["tau"],
        epsilon=group["epsilon"],
        iters=group["iters"],
        shape=tuple(param.shape),
        row_groups=_shard_groups(param, 1),
        col_groups=_shard_groups(param, 0),
    )
    yield


def _segment_reduce(values: Tensor, sizes: list[int], reduce: str) -> Tensor:
    """Sum or max of ``values`` [B, rows, ...] over consecutive row segments of ``sizes``: [B, len(sizes), ...].
    Splits instead of index_add/scatter_reduce, whose CUDA kernels are not deterministic. The max assumes values
    are nonnegative, so an empty segment is 0."""
    pad = values.new_zeros(values.shape[0], 1, *values.shape[2:])
    chunks = values.split(sizes, dim=1)
    if reduce == "sum":
        return torch.stack([chunk.sum(1) for chunk in chunks], 1)
    return torch.stack([torch.cat([chunk, pad], 1).amax(1) for chunk in chunks], 1)


def _all_reduce(tensor: Tensor, groups: list[ProcessGroup], op=dist.ReduceOp.SUM) -> Tensor:
    for group in groups:
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


def _conditioner(energy: Tensor, peak: Tensor, damping: float) -> Tensor:
    """Dimuon's (energy / peak + damping)^-1/4, the identity where the peak (the previous momentum) is zero."""
    cond = (energy / peak.clamp_min(torch.finfo(energy.dtype).tiny) + damping) ** -0.25
    return torch.where(peak > 0, cond, torch.ones_like(cond))


class _DimuonShard:
    """Dimuon's diagonal conditioning on the local shard of a [..., m, n] parameter, with matrices packed along its
    rows (``partitions``). Leading dims are batches of matrices (experts). Row, column and Frobenius norms
    all-reduce over the mesh dims that shard the matrix rows (``row_groups``) or columns (``col_groups``)."""

    def __init__(self, param: Tensor, partitions: tuple[int, ...] | None):
        m, self.n = param.shape[-2:]
        sizes = partitions or (m,)
        self.row_groups, self.col_groups = _shard_groups(param, param.ndim - 2), _shard_groups(param, param.ndim - 1)
        segments = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        for group in self.row_groups:  # dion supports one matrix shard, in torch.chunk layout
            chunks = segments.chunk(dist.get_world_size(group))
            rank = dist.get_rank(group)
            segments = chunks[rank] if rank < len(chunks) else segments[:0]
        self.segments = segments.to(param.device)
        self.local_sizes = torch.bincount(segments, minlength=len(sizes)).tolist()
        self.rows = torch.tensor(sizes, dtype=torch.float32, device=param.device)

    def conditioners(self, M: Tensor, damping: float) -> tuple[Tensor, Tensor]:
        """Row and column conditioners [B, m_local, 1], [B, m_local, n_local] (each row's own segment)."""
        energy = M.square()
        rows = _all_reduce(energy.sum(-1), self.col_groups)
        row_peak = _all_reduce(_segment_reduce(rows, self.local_sizes, "max"), self.row_groups, dist.ReduceOp.MAX)
        cols = _all_reduce(_segment_reduce(energy, self.local_sizes, "sum"), self.row_groups)
        col_peak = _all_reduce(cols.amax(-1, keepdim=True), self.col_groups, dist.ReduceOp.MAX)
        L = _conditioner(rows, row_peak[:, self.segments], damping).unsqueeze(-1)
        C = _conditioner(cols, col_peak, damping)[:, self.segments]
        return L, C

    def norms(self, D: Tensor) -> Tensor:
        """Frobenius norm of each matrix's update, per row [B, m_local, 1]."""
        squares = _segment_reduce(D.square().sum(-1), self.local_sizes, "sum")
        return _all_reduce(squares, self.row_groups + self.col_groups).sqrt()[:, self.segments].unsqueeze(-1)


class Muon(dion.Muon):
    """dion's Muon with an update RMS of ``update_rms`` and more algorithms. ``variant`` ``dimuon`` or ``muown``
    updates the Muon groups with Dimuon or Muown. ``sinkhorn`` groups take the Sinkhorn-balanced momentum update,
    with their lr times ``lr_scale``. Both scales apply inside the step, so the groups' lr stays the scheduled lr."""

    def __init__(self, *args, update_rms: float, variant: str = "muon", bf16_momentum: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_rms = update_rms
        self.variant = variant
        self.bf16_momentum = bf16_momentum
        self._dimuon_shards = {}

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
        if self.variant == "muon":
            muon_tasks = self._create_muon_tasks([self._dion_group(g) for g in groups["muon"]])
        else:
            update = self._dimuon_batch if self.variant == "dimuon" else self._muown_batch
            muon_tasks = (
                AsyncTask(update(batch, group))
                for group in groups["muon"]
                for batch in create_param_batches(
                    [p for p in group["params"] if p.grad is not None],
                    batch_size=self._get_group_mesh_info(group)[1],
                    matrix_partitions=self._matrix_partitions,
                )
            )
        sinkhorn_tasks = (
            AsyncTask(_sinkhorn_task(p, self._get_or_initialize_state(p, "sinkhorn"), group, self.update_rms))
            for group in groups["sinkhorn"]
            for p in group["params"]
            if p.grad is not None
        )
        tasks = chain(muon_tasks, self._create_adamw_tasks(groups["adamw"]), sinkhorn_tasks)
        AsyncRuntime(tasks, max_concurrent_tasks=3).run()
        return loss

    def _dion_group(self, group: dict) -> dict:
        """dion scales Muon updates to RMS 0.2 and decays weights by the group lr: rescale the lr, keep lr * wd."""
        scale = self.update_rms / 0.2
        return dict(group, lr=group["lr"] * scale, weight_decay=group["weight_decay"] / scale)

    def _dion_muon_update(self, params, X, G, M, group, *, newton_schulz_func):
        """dion's Muon update of the local tensors ``X`` (shards shaped like ``params``) with gradients ``G`` and
        momentum ``M``, with the group's lr, momentum, weight decay and lr adjustment. Matrices are orthogonalized
        where dion gathers them whole."""
        device_rank, world_size, process_group, _ = self._get_group_mesh_info(group)
        shard_dim, is_local = None, False
        if isinstance(params[0], DTensor):
            shard_dim, is_local = self._get_matrix_shard_dim(params[0], group["flatten"], process_group)
            X = [
                DTensor.from_local(x, p.device_mesh, p.placements, shape=p.shape, stride=p.stride())
                for x, p in zip(X, params)
            ]
        if not is_local:
            X, G, M = (pad_batch(list(tensors), world_size) for tensors in (X, G, M))
        yield from dion.muon.muon_update_batch_async(
            X=X,
            G=G,
            M=M,
            lr=torch.tensor(group["lr"]),
            momentum=torch.tensor(group["mu"]),
            weight_decay=torch.tensor(group["weight_decay"]),
            epsilon=torch.tensor(group["epsilon"]),
            nesterov=group["nesterov"],
            flatten=group["flatten"],
            adjust_lr=group["adjust_lr"],
            device_rank=device_rank,
            world_size=world_size,
            shard_dim=shard_dim,
            is_local=is_local,
            process_group=process_group,
            newton_schulz_func=newton_schulz_func,
            partitions=self._matrix_partitions.get(params[0]),
        )

    def _dimuon_batch(self, params, group):
        """Dimuon (github.com/anpaure/dimuon) on a batch of parameters with one shape and sharding: EMA momentum,
        conditioners from the previous momentum around Polar Express, and a fixed update norm, so each matrix's
        update has RMS ``update_rms`` times the lr."""
        beta = group["mu"]
        partitions = self._matrix_partitions.get(params[0])
        shards, conditioners, inputs = [], [], []
        for p in params:
            if p not in self._dimuon_shards:
                self._dimuon_shards[p] = _DimuonShard(p, partitions)
            state = self.state[p]
            if not state:
                state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16 if self.bf16_momentum else None)
            elif self.bf16_momentum and state["momentum"].dtype != torch.bfloat16:
                # Optimizer.load_state_dict casts the loaded state to the parameter dtype
                state["momentum"] = state["momentum"].to(torch.bfloat16)
            M = to_local(state["momentum"])
            G = to_local(p.grad).reshape(-1, *M.shape[-2:]).float()
            M_old = M.reshape(G.shape).float()
            L, C = self._dimuon_shards[p].conditioners(M_old, damping=1e-4)
            M_new = M_old.lerp(G, 1 - beta)
            M.copy_(M_new.reshape(M.shape))
            U = G.lerp(M_new, beta) if group["nesterov"] else M_new
            shards.append(self._dimuon_shards[p])
            conditioners.append((L, C))
            inputs.append((L * U * C).reshape(M.shape))

        # With zero momentum, lr -1, no decay and no lr adjustment, dion writes the orthogonalized inputs to outputs
        outputs = [torch.zeros_like(u) for u in inputs]
        identity = dict(group, lr=-1.0, mu=0.0, weight_decay=0.0, nesterov=False, adjust_lr=None)
        zeros = [torch.zeros_like(u) for u in inputs]
        yield from self._dion_muon_update(
            params, outputs, inputs, zeros, identity, newton_schulz_func=_dimuon_polar_express
        )

        lr = group["lr"]
        for p, shard, (L, C), output in zip(params, shards, conditioners, outputs):
            D = L * output.reshape(L.shape[0], -1, C.shape[-1]) * C
            norm = shard.norms(D)
            target = lr * self.update_rms * (shard.rows[shard.segments].unsqueeze(-1) * shard.n).sqrt()
            X = to_local(p)
            X.mul_(1 - lr * group["weight_decay"])
            X.sub_((torch.where(norm > 0, target / norm, 0.0) * D).reshape(X.shape).to(X.dtype))

    def _muown_batch(self, params, group):
        """Muown (arXiv 2605.10797, Algorithm 1) on a batch of parameters with one shape and sharding: W = diag(g/r) R.
        dion's Muon updates the direction R with the tangential gradient; Adam (the AdamW betas and epsilon, the
        same lr) updates the row norms g. Decoupled weight decay shrinks W, then g is reset to W's row norms."""
        directions, gradients, momentums, states = [], [], [], []
        for p in params:
            W, groups = to_local(p), _shard_groups(p, p.ndim - 1)
            state = self.state[p]
            if not state:
                state["momentum"] = torch.zeros_like(p)
                state["row_norm"] = torch.zeros_like(p[..., :1])
                to_local(state["row_norm"]).copy_(_norms(W, -1, groups).unsqueeze(-1))
                state["direction_norm"] = state["row_norm"].clone()
                state["row_norm_exp_avg"] = torch.zeros_like(state["row_norm"])
                state["row_norm_exp_avg_sq"] = torch.zeros_like(state["row_norm"])
            g, r = to_local(state["row_norm"]), to_local(state["direction_norm"])
            G = to_local(p.grad).to(W.dtype)
            unit_rows = W / g.clamp_min(torch.finfo(g.dtype).tiny)
            grad_g = (G * unit_rows).sum(-1, keepdim=True)
            for shard_group in groups:
                dist.all_reduce(grad_g, group=shard_group)
            directions.append(unit_rows * r)
            gradients.append(g / r.clamp_min(torch.finfo(r.dtype).tiny) * (G - unit_rows * grad_g))
            momentums.append(to_local(state["momentum"]))
            states.append((state, grad_g, groups))

        muon_group = dict(self._dion_group(group), weight_decay=0.0)
        yield from self._dion_muon_update(
            params, directions, gradients, momentums, muon_group, newton_schulz_func=self._newton_schulz_func
        )

        lr, (beta1, beta2), step = group["lr"], (group["beta1"], group["beta2"]), group["step"]
        for p, R, (state, grad_g, groups) in zip(params, directions, states):
            g, r = to_local(state["row_norm"]), to_local(state["direction_norm"])
            exp_avg, exp_avg_sq = to_local(state["row_norm_exp_avg"]), to_local(state["row_norm_exp_avg_sq"])
            exp_avg.lerp_(grad_g, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
            denom = (exp_avg_sq / (1 - beta2**step)).sqrt_().add_(group["epsilon"])
            g.sub_(lr / (1 - beta1**step) * exp_avg / denom)
            r.copy_(_norms(R, -1, groups).unsqueeze(-1))
            W = to_local(p)
            W.mul_(-lr * group["weight_decay"]).add_(g / r.clamp_min(torch.finfo(r.dtype).tiny) * R)
            if group["weight_decay"] > 0:
                g.copy_(_norms(W, -1, groups).unsqueeze(-1))
