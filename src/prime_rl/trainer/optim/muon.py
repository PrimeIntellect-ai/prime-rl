import math

import dion
import torch
import torch.distributed as dist
from dion.opt_utils import AsyncRuntime, AsyncTask, create_param_batches, pad_batch, to_local
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

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


def newton_schulz(coefficients: list[tuple[float, float, float]], safety=1.0, dtype=torch.bfloat16, epsilon=None):
    """Orthogonalization by odd polynomials: X <- aX + (bA + cA²)X with A = XXᵀ, one (a, b, c) per iteration, on
    X = G / (safety·‖G‖_F + epsilon) normalized in bf16 and iterated in ``dtype``. A given ``epsilon`` replaces the
    caller's."""
    fixed_epsilon = epsilon

    @torch.compile(fullgraph=True)
    def orthogonalize(G: Tensor, epsilon: float | Tensor = 1e-7) -> Tensor:
        X = G.to(dtype=torch.bfloat16)
        if G.size(-2) > G.size(-1):
            X = X.mT
        if fixed_epsilon is not None:
            epsilon = fixed_epsilon
        X = (X / (X.norm(dim=(-2, -1), keepdim=True) * safety + epsilon)).to(dtype)
        for a, b, c in coefficients:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    return orthogonalize


NEWTON_SCHULZ = {
    # dion's zeropower_via_newtonschulz5
    "default": newton_schulz(
        [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046)]
        + [(2.8366, -3.0525, 1.2012)]
    ),
    # DeepSeek-V4: 8 fast-converging steps, then 2 that pin the singular values at 1
    "deepseek_v4": newton_schulz([(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2),
    "polar_express": newton_schulz(POLAR_EXPRESS[:5], safety=1.01),
    # Step 3.5 Flash: 6 Polar Express steps in fp16 (bf16 spiked)
    "step3p5": newton_schulz(POLAR_EXPRESS[:6], safety=1.01, dtype=torch.float16),
}
# Dimuon's orthogonalizer
DIMUON_NEWTON_SCHULZ = newton_schulz(POLAR_EXPRESS[:5], safety=1.01, dtype=torch.float16, epsilon=1e-7)


def _shard_groups(param: Tensor, dim: int) -> list[ProcessGroup]:
    """The process groups of the mesh dims that shard ``dim`` of ``param``."""
    if not isinstance(param, DTensor):
        return []
    mesh = param.device_mesh
    return [
        mesh.get_group(mesh_dim)
        for mesh_dim, placement in enumerate(param.placements)
        if getattr(placement, "dim", None) == dim and mesh.size(mesh_dim) > 1
    ]


def _all_reduce(tensor: Tensor, groups: list[ProcessGroup], op=dist.ReduceOp.SUM) -> Tensor:
    for group in groups:
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


def _norms(U: Tensor, dim: int, groups: list[ProcessGroup]) -> Tensor:
    """L2 norms of U along ``dim``, summed over the process groups that shard that dimension."""
    return _all_reduce(U.square().sum(dim=dim), groups).sqrt()


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
    mean_row_norm = _all_reduce(row_norms.sum(), col_groups) / shape[0]
    U = U * (row_norms > tau * mean_row_norm).unsqueeze(1)
    for k in range(iters):
        if k % 2 == 0:
            U = U / (_norms(U, 1, row_groups) + epsilon).unsqueeze(1)
        else:
            U = U / (_norms(U, 0, col_groups) + epsilon).unsqueeze(0)
    X.sub_(U.to(X.dtype), alpha=lr * update_rms * math.sqrt(shape[1]))


def _segment_reduce(values: Tensor, sizes: list[int], reduce: str) -> Tensor:
    """Sum or max of ``values`` [B, rows, ...] over consecutive row segments of ``sizes``: [B, len(sizes), ...].
    Splits instead of index_add/scatter_reduce, whose CUDA kernels are not deterministic. The max assumes values
    are nonnegative, so an empty segment is 0."""
    pad = values.new_zeros(values.shape[0], 1, *values.shape[2:])
    chunks = values.split(sizes, dim=1)
    if reduce == "sum":
        return torch.stack([chunk.sum(1) for chunk in chunks], 1)
    return torch.stack([torch.cat([chunk, pad], 1).amax(1) for chunk in chunks], 1)


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
    """dion's Muon with an update RMS of ``update_rms`` and more update rules, one task per batch: dion's AdamW for
    ``adamw`` groups, Sinkhorn for ``sinkhorn`` groups (lr times the group's ``lr_scale``), and the ``variant`` rule
    (``muon``, ``dimuon``, ``muown``) for ``muon`` groups, all on dion's gathering and orthogonalization. Both scales
    apply inside the step, so the groups' lr stays the scheduled lr."""

    def __init__(self, *args, update_rms: float = 0.2, variant: str = "muon", bf16_momentum: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_rms = update_rms
        self.bf16_momentum = bf16_momentum
        self._muon_rule = {"muon": self._muon, "dimuon": self._dimuon, "muown": self._muown}[variant]
        self._dimuon_shards = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["step"] += 1
        AsyncRuntime(self._tasks(), max_concurrent_tasks=3).run()
        return loss

    def _tasks(self):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if group["algorithm"] == "adamw":
                yield from self._create_adamw_tasks([group])
            elif group["algorithm"] == "sinkhorn":
                yield from (AsyncTask(self._sinkhorn(p, group)) for p in params)
            else:
                batch_size = self._get_group_mesh_info(group)[1]
                for batch in create_param_batches(params, batch_size, matrix_partitions=self._matrix_partitions):
                    yield AsyncTask(self._muon_rule(batch, group))

    def _momentum(self, p: Tensor) -> Tensor:
        state = self.state[p]
        if not state:
            state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16 if self.bf16_momentum else None)
        elif self.bf16_momentum and state["momentum"].dtype != torch.bfloat16:
            # Optimizer.load_state_dict casts the loaded state to the parameter dtype
            state["momentum"] = state["momentum"].to(torch.bfloat16)
        return to_local(state["momentum"])

    def _dion_update(self, params, X, G, M, group, *, weight_decay=0.0, orthogonalize_only=False):
        """dion's Muon update (momentum, orthogonalization, scaled step) of the local tensors ``X`` (shards shaped like
        ``params``) with gradients ``G`` and momentum ``M``, orthogonalizing where dion gathers whole matrices.

        dion has no hook for the update rule, so this is where it is bent: it redoes dion's batch setup (global
        DTensor shapes, padding); it scales the update to RMS ``update_rms`` (dion fixes 0.2 and decays by the lr, so
        the lr is rescaled and lr * weight_decay kept); and with ``orthogonalize_only`` (zero momentum, lr -1, no
        decay or lr adjustment) dion writes the orthogonalized ``G``, by Dimuon's Newton-Schulz, to ``X``."""
        if orthogonalize_only:
            lr, momentum, nesterov, adjust_lr = -1.0, 0.0, False, None
            ns, epsilon = DIMUON_NEWTON_SCHULZ, group["epsilon"]
        else:
            scale = self.update_rms / 0.2
            lr, weight_decay = group["lr"] * scale, weight_decay / scale
            momentum, nesterov, adjust_lr = group["mu"], group["nesterov"], group["adjust_lr"]
            ns, epsilon = self._newton_schulz_func, group["epsilon"]
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
            lr=torch.tensor(lr),
            momentum=torch.tensor(momentum),
            weight_decay=torch.tensor(weight_decay),
            epsilon=torch.tensor(epsilon),
            nesterov=nesterov,
            flatten=group["flatten"],
            adjust_lr=adjust_lr,
            device_rank=device_rank,
            world_size=world_size,
            shard_dim=shard_dim,
            is_local=is_local,
            process_group=process_group,
            newton_schulz_func=ns,
            partitions=self._matrix_partitions.get(params[0]),
        )

    def _muon(self, params, group):
        G = [to_local(p.grad) for p in params]
        M = [self._momentum(p) for p in params]
        yield from self._dion_update(params, to_local(params), G, M, group, weight_decay=group["weight_decay"])

    def _sinkhorn(self, p, group):
        sinkhorn_update(
            to_local(p),
            to_local(p.grad),
            self._momentum(p),
            lr=group["lr"] * group["lr_scale"],
            momentum=group["mu"],
            update_rms=self.update_rms,
            tau=group["tau"],
            epsilon=group["epsilon"],
            iters=group["iters"],
            shape=tuple(p.shape),
            row_groups=_shard_groups(p, 1),
            col_groups=_shard_groups(p, 0),
        )
        yield

    def _dimuon(self, params, group):
        """Dimuon (github.com/anpaure/dimuon) on a batch of parameters with one shape and sharding: EMA momentum,
        conditioners from the previous momentum around Polar Express, and a fixed update norm, so each matrix's
        update has RMS ``update_rms`` times the lr."""
        beta = group["mu"]
        shards, conditioners, inputs = [], [], []
        for p in params:
            if p not in self._dimuon_shards:
                self._dimuon_shards[p] = _DimuonShard(p, self._matrix_partitions.get(p))
            M = self._momentum(p)
            G = to_local(p.grad).reshape(-1, *M.shape[-2:]).float()
            M_old = M.reshape(G.shape).float()
            L, C = self._dimuon_shards[p].conditioners(M_old, damping=1e-4)
            M_new = M_old.lerp(G, 1 - beta)
            M.copy_(M_new.reshape(M.shape))
            U = G.lerp(M_new, beta) if group["nesterov"] else M_new
            shards.append(self._dimuon_shards[p])
            conditioners.append((L, C))
            inputs.append((L * U * C).reshape(M.shape))

        outputs = [torch.zeros_like(u) for u in inputs]
        zeros = [torch.zeros_like(u) for u in inputs]
        yield from self._dion_update(params, outputs, inputs, zeros, group, orthogonalize_only=True)

        lr = group["lr"]
        for p, shard, (L, C), output in zip(params, shards, conditioners, outputs):
            D = L * output.reshape(L.shape[0], -1, C.shape[-1]) * C
            norm = shard.norms(D)
            target = lr * self.update_rms * (shard.rows[shard.segments].unsqueeze(-1) * shard.n).sqrt()
            X = to_local(p)
            X.mul_(1 - lr * group["weight_decay"])
            X.sub_((torch.where(norm > 0, target / norm, 0.0) * D).reshape(X.shape).to(X.dtype))

    def _muown(self, params, group):
        """Muown (arXiv 2605.10797, Algorithm 1) on a batch of parameters with one shape and sharding: W = diag(g/r) R.
        dion's Muon updates the direction R with the tangential gradient; Adam (the AdamW betas and epsilon, the
        same lr) updates the row norms g. Decoupled weight decay shrinks W, then g is reset to W's row norms."""
        directions, gradients, momentums, rows = [], [], [], []
        for p in params:
            W, groups = to_local(p), _shard_groups(p, p.ndim - 1)
            state = self.state[p]
            momentums.append(self._momentum(p))
            if "row_norm" not in state:
                state["row_norm"] = torch.zeros_like(p[..., :1])
                to_local(state["row_norm"]).copy_(_norms(W, -1, groups).unsqueeze(-1))
                state["direction_norm"] = state["row_norm"].clone()
                state["row_norm_exp_avg"] = torch.zeros_like(state["row_norm"])
                state["row_norm_exp_avg_sq"] = torch.zeros_like(state["row_norm"])
            g, r = to_local(state["row_norm"]), to_local(state["direction_norm"])
            G = to_local(p.grad).to(W.dtype)
            unit_rows = W / g.clamp_min(torch.finfo(g.dtype).tiny)
            grad_g = _all_reduce((G * unit_rows).sum(-1, keepdim=True), groups)
            directions.append(unit_rows * r)
            gradients.append(g / r.clamp_min(torch.finfo(r.dtype).tiny) * (G - unit_rows * grad_g))
            rows.append((state, grad_g, groups))

        yield from self._dion_update(params, directions, gradients, momentums, group)

        lr, (beta1, beta2), step = group["lr"], (group["beta1"], group["beta2"]), group["step"]
        for p, R, (state, grad_g, groups) in zip(params, directions, rows):
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
