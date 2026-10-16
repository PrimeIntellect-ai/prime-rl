"""Master-versus-cast probe for the FP8 weight-dynamics experiments (int4/experiments/fp8-weight-dynamics in research-prod)."""

import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.trainer.model import get_load_balance_stats, reshard_module
from prime_rl.trainer.models.fp8_rounding import FROZEN_STEP, seed_for_step
from prime_rl.trainer.models.kernels.fp8_utils import grouped_per_block_cast_to_fp8_triton, weight_ue8m0_for_device
from prime_rl.trainer.models.layers.expert_compute import DeepGemmFP8ExpertCompute
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear, bf16_compute
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.sft.data import Batch, CatDataset, StatefulIterableDataset, cat_collate
from prime_rl.trainer.sft.weight_dynamics import _canonical_params, _full, _local
from prime_rl.utils.logger import get_logger

FP8_COVERED_PARAM = re.compile(
    r"\.(self_attn\.(q|k|v|o|qkv)_proj\.weight|mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj))$"
)
BLOCK = 128
PREFIX = "master_cast_probe/"
OWN_KEYS = (
    "loss_cast",
    "gap",
    "grad_norm_cast",
    "cast_changed_fraction",
    "descent_cast",
    "forward_visible_fraction",
    "grad_change_cos_displacement",
)


def ue8m0_cast_fp8(
    weight: torch.Tensor,
    seed: torch.Tensor | None = None,
    salt: torch.Tensor | None = None,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """UE8M0 128x128 block cast of a 2D or 3D weight as FP8 payload and scales, rounded stochastically when seeded."""
    blocks = weight.unsqueeze(0) if weight.dim() == 2 else weight
    assert blocks.shape[-2] % BLOCK == 0 and blocks.shape[-1] % BLOCK == 0, f"unaligned weight {tuple(weight.shape)}"
    return grouped_per_block_cast_to_fp8_triton(
        blocks.bfloat16().contiguous(), True, rounding_seed=seed, rounding_salt=salt, rounding_offset=offset
    )


def dequantize(fp8: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """bf16 values of a block cast, exact because FP8 payloads times power-of-two scales fit in bf16."""
    dequantized = fp8.float() * scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    return dequantized.to(torch.bfloat16).view(shape)


def ue8m0_cast(
    weight: torch.Tensor,
    seed: torch.Tensor | None = None,
    salt: torch.Tensor | None = None,
    offset: int = 0,
) -> torch.Tensor:
    """Dequantized UE8M0 128x128 block cast of a 2D or 3D weight, round-to-nearest unless seeded."""
    return dequantize(*ue8m0_cast_fp8(weight, seed, salt, offset), weight.shape)


def ue8m0_rtn(weight: torch.Tensor) -> torch.Tensor:
    """Dequantized UE8M0 round-to-nearest 128x128 block cast of a 2D or 3D weight, exact in bf16."""
    return ue8m0_cast(weight)


def load_probe_batches(dataset: StatefulIterableDataset, seq_len: int, cp: int, count: int) -> list[Batch]:
    """The first ``count`` packed micro-batches of this rank's share of ``dataset``, as the training loader packs them."""
    packed = iter(CatDataset(dataset, seq_len, pad_to_multiple_of=cp))
    return [cat_collate([next(packed)]) for _ in range(count)]


@dataclass
class CastOwner:
    """Rounding salt and RNG index offset with which a weight's local shard is cast in the training forward."""

    salt: torch.Tensor | None
    offset: int


def _local_row_start(weight: torch.Tensor) -> int:
    if not isinstance(weight, DTensor):
        return 0
    _, global_offset = compute_local_shape_and_global_offset(weight.shape, weight.device_mesh, weight.placements)
    return global_offset[0]


def _cast_owners(model: nn.Module) -> dict[int, CastOwner]:
    owners = {}
    for module in model.modules():
        if isinstance(module, Float8BlockwiseLinear) and module.weight is not None:
            weight = module.weight
            owners[id(weight)] = CastOwner(module.rounding_salt, _local_row_start(weight) * weight.shape[1])
        elif isinstance(module, GroupedExperts) and isinstance(module.compute, DeepGemmFP8ExpertCompute):
            salts = getattr(module, "rounding_salts", {})
            for name, param in module.named_parameters(recurse=False):
                local = _local(param)
                offset = getattr(module, "fp8_expert_offset", 0) * local.shape[1] * local.shape[2]
                owners[id(param)] = CastOwner(salts.get(name), offset)
    return owners


class MasterCastProbe:
    """Loss and gradient on fixed batches at the bf16 master ``m``, at its UE8M0 RTN cast ``Q(m)`` and at the arm's
    own training cast ``Q_own(m)``, all in bf16 compute, projected on the displacement ``D = bf16(m) - m_0`` of the
    FP8-covered weights. Leaves weights, gradients and optimizer state unchanged."""

    def __init__(
        self,
        model: nn.Module,
        batches: list[Batch],
        compute_loss: Callable[[Batch], tuple[torch.Tensor, torch.Tensor]],
        loss_group: dist.ProcessGroup | None,
        grad_divide_factor: int,
    ):
        self.model = model
        self.batches = batches
        self.compute_loss = compute_loss
        self.loss_group = loss_group
        self.grad_divide_factor = grad_divide_factor
        params = _canonical_params(model)
        self.names = names = sorted(name for name in params if FP8_COVERED_PARAM.search(name))
        assert any("self_attn" in name for name in names) and any("experts" in name for name in names), names
        self.params = [params[name] for name in names]
        owners = _cast_owners(model)
        if owners:
            covered_ids = {id(param) for param in self.params}
            assert set(owners) == covered_ids, (
                f"FP8 weights {len(owners)} differ from the probed weights {len(covered_ids)}: "
                f"{sorted(name for name, param in params.items() if id(param) in set(owners) ^ covered_ids)}"
            )
            assert weight_ue8m0_for_device(), (
                "the probe's casts are UE8M0, so the FP8 forward must cast weights with UE8M0"
            )
            self.owners = [owners[id(param)] for param in self.params]
        self.rounding = getattr(model, "fp8_rounding", None)
        if not owners:
            self.kind = "bf16"
        elif self.rounding is None:
            self.kind = "rtn"
        else:
            self.kind = "fixed" if self.rounding.frozen else "sr"
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for name, param in zip(names, self.params):
            # Global sums add local shards over all ranks, so every element must live on exactly one rank.
            if isinstance(param, DTensor):
                assert not any(p.is_replicate() or p.is_partial() for p in param.placements), (name, param.placements)
                assert param.device_mesh.size() == world_size, (name, param.device_mesh)
            else:
                assert world_size == 1, name
        if self.kind == "fixed":
            assert self.rounding.seed.item() == seed_for_step(self.rounding.base_seed, FROZEN_STEP)
        self.start = [_local(param).detach().to(torch.bfloat16, copy=True) for param in self.params]
        self._check_local_casts()
        start_bytes = sum(s.numel() * s.element_size() for s in self.start)
        get_logger().info(
            f"Master-cast probe ({self.kind} training cast) on {len(self.params)} weights "
            f"({start_bytes / 1024**3:.1f} GiB of local bf16 starts), {len(batches)} micro-batches per rank"
        )

    def _seed_tensor(self, step: int) -> torch.Tensor:
        return torch.tensor(seed_for_step(self.rounding.base_seed, step), device="cuda", dtype=torch.int64)

    def forward_seed(self, step: int) -> torch.Tensor | None:
        """The rounding seed of the training forward of ``step``, or None for round-to-nearest."""
        if self.kind == "fixed":
            return self._seed_tensor(FROZEN_STEP)
        if self.kind == "sr":
            return self._seed_tensor(step)
        return None

    def own_cast_fp8(self, index: int, weight: torch.Tensor, seed: torch.Tensor | None):
        owner = self.owners[index]
        if owner.salt is None:
            return ue8m0_cast_fp8(weight)
        return ue8m0_cast_fp8(weight, seed, owner.salt, owner.offset)

    def own_cast(self, index: int, weight: torch.Tensor, seed: torch.Tensor | None) -> torch.Tensor:
        return dequantize(*self.own_cast_fp8(index, weight, seed), weight.shape)

    def _check_local_casts(self) -> None:
        dense = next(
            (i for i, name in enumerate(self.names) if "qkv_proj" in name),
            next(i for i, param in enumerate(self.params) if param.dim() == 2),
        )
        expert = next(i for i, param in enumerate(self.params) if param.dim() == 3)
        seed = self.forward_seed(FROZEN_STEP)
        for i in (dense, expert):
            param = self.params[i].detach()
            checks = [("RTN", ue8m0_rtn(_local(param)), lambda full: ue8m0_rtn(full))]
            if seed is not None:
                salt = self.owners[i].salt
                checks.append(("own", self.own_cast(i, _local(param), seed), lambda full: ue8m0_cast(full, seed, salt)))
            for label, local_cast, full_cast in checks:
                if isinstance(param, DTensor):
                    local_cast = DTensor.from_local(
                        local_cast, param.device_mesh, param.placements, shape=param.shape, stride=param.stride()
                    ).full_tensor()
                assert torch.equal(local_cast, full_cast(_full(param).bfloat16())), (
                    f"local {label} cast of {self.names[i]} differs from the cast of the full tensor"
                )

    def _loss_and_backward(self) -> tuple[torch.Tensor, torch.Tensor]:
        loss_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        token_count = torch.zeros((), dtype=torch.float64, device="cuda")
        for batch in self.batches:
            local_loss_sum, local_token_count = self.compute_loss(batch)
            local_loss_sum.backward()
            loss_sum += local_loss_sum.detach()
            token_count += local_token_count
        totals = torch.stack([loss_sum, token_count])
        if dist.is_initialized():
            dist.all_reduce(totals, group=self.loss_group)
        return totals[0] / totals[1], totals[1]

    def _local_grads(self) -> list[torch.Tensor]:
        return [_local(param.grad).float() for param in self.params]

    @torch.no_grad()
    def _write(self, values: Iterable[torch.Tensor]) -> None:
        for param, value in zip(self.params, values):
            _local(param).copy_(value)

    @torch.no_grad()
    def _cast_sums(
        self,
        masters: list[torch.Tensor],
        grads_master: list[torch.Tensor],
        cast: Callable[[int, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Global sums over the covered weights for one cast, with the gradient at the cast in ``param.grad``."""
        sums = torch.zeros(9, dtype=torch.float64, device="cuda")
        for i, (master, start, grad_master, grad_cast) in enumerate(
            zip(masters, self.start, grads_master, self._local_grads())
        ):
            master_bf16 = master.bfloat16()
            cast_master = cast(i, master_bf16)
            displacement = master_bf16.float() - start.float()
            grad_change = grad_cast - grad_master
            terms = [
                displacement.square().sum(),
                (grad_master * displacement).sum(),
                grad_master.square().sum(),
                (grad_cast * displacement).sum(),
                grad_change.square().sum(),
                grad_cast.square().sum(),
                ((cast_master.float() - cast(i, start).float()) * displacement).sum(),
                (cast_master != master_bf16).sum(),
                torch.tensor(master.numel()),
            ]
            sums += torch.stack([term.to(sums) for term in terms])
        if dist.is_initialized():
            dist.all_reduce(sums)
        return sums

    def _cast_metrics(self, prefix: str, loss_master: float, loss_cast: float, sums: list[float], scale: float):
        d_sq, grad_master_d, _, grad_cast_d, grad_change_sq, _, visible, cast_changed, numel = sums
        metrics = {
            f"{prefix}loss_cast": loss_cast,
            f"{prefix}gap": loss_cast - loss_master,
            f"{prefix}grad_norm_cast": scale * math.sqrt(sums[5]),
            f"{prefix}cast_changed_fraction": cast_changed / numel,
        }
        if d_sq > 0:
            d_norm = math.sqrt(d_sq)
            metrics[f"{prefix}descent_cast"] = -scale * grad_cast_d / d_norm
            metrics[f"{prefix}forward_visible_fraction"] = visible / d_sq
            if grad_change_sq > 0:
                metrics[f"{prefix}grad_change_cos_displacement"] = (grad_cast_d - grad_master_d) / (
                    math.sqrt(grad_change_sq) * d_norm
                )
        return metrics

    def measure(self, step: int) -> dict[str, float]:
        """Probe after the optimizer step of ``step`` (0 before training). The own cast is the next forward's."""
        start_time = time.perf_counter()
        assert all(param.grad is None for param in self.model.parameters()), "run the probe after zero_grad"
        casts = {"": lambda i, weight: ue8m0_rtn(weight)}
        own_seed = self.forward_seed(step + 1)
        if own_seed is not None:
            casts["own/"] = lambda i, weight: self.own_cast(i, weight, own_seed)
        losses, sums = {}, {}
        with torch.compiler.set_stance("force_eager"), bf16_compute():
            loss_master, token_count = self._loss_and_backward()
            grads_master = [grad.clone() for grad in self._local_grads()]
            self.model.zero_grad(set_to_none=True)
            masters = [_local(param).detach().clone() for param in self.params]
            for prefix, cast in casts.items():
                self._write(cast(i, master.bfloat16()) for i, master in enumerate(masters))
                losses[prefix], _ = self._loss_and_backward()
                sums[prefix] = self._cast_sums(masters, grads_master, cast).tolist()
                self.model.zero_grad(set_to_none=True)
            self._write(masters)
        del masters, grads_master
        # Keep the probe's routing out of the next training step's statistics.
        get_load_balance_stats(self.model)
        reshard_module(self.model)

        # Gradients of the token-mean loss: FSDP divides the reduced sum by grad_divide_factor.
        scale = self.grad_divide_factor / token_count.item()
        d_sq, grad_master_d, grad_master_sq = sums[""][:3]
        loss_master = loss_master.item()
        metrics = {
            f"{PREFIX}loss_master": loss_master,
            f"{PREFIX}displacement_norm": math.sqrt(d_sq),
            f"{PREFIX}grad_norm_master": scale * math.sqrt(grad_master_sq),
            f"{PREFIX}num_tokens": token_count.item(),
        }
        if d_sq > 0:
            metrics[f"{PREFIX}descent_master"] = -scale * grad_master_d / math.sqrt(d_sq)
        for prefix in casts:
            metrics |= self._cast_metrics(PREFIX + prefix, loss_master, losses[prefix].item(), sums[prefix], scale)
        own = PREFIX + "own/"
        if self.kind == "rtn":
            metrics |= {
                own + key.removeprefix(PREFIX): value
                for key, value in list(metrics.items())
                if key.removeprefix(PREFIX) in OWN_KEYS
            }
        elif self.kind == "bf16":
            metrics |= {own + "loss_cast": loss_master, own + "gap": 0.0, own + "cast_changed_fraction": 0.0}
            metrics[own + "grad_norm_cast"] = metrics[f"{PREFIX}grad_norm_master"]
            if d_sq > 0:
                metrics[own + "descent_cast"] = metrics[f"{PREFIX}descent_master"]
                metrics[own + "forward_visible_fraction"] = 1.0
        elif self.kind == "sr" and d_sq > 0:
            # One stochastic realization; the forward is unbiased, E[Q(x)] = x, so its expected visible fraction is 1.
            metrics[own + "forward_visible_fraction_realized"] = metrics[own + "forward_visible_fraction"]
            metrics[own + "forward_visible_fraction"] = 1.0
        metrics[f"{PREFIX}time"] = time.perf_counter() - start_time
        return metrics


class FP8FlipRate:
    """Per-step share of FP8-covered weight elements whose training-forward value or bf16 master changed since the
    previous step and since the first training step. Shares the probe's weights, casts and start."""

    def __init__(self, probe: MasterCastProbe):
        self.probe = probe
        self.previous_forward: list | None = None
        self.first_forward: list | None = None
        self.previous_master: list[torch.Tensor] | None = None

    def _forward_value(self, index: int, master_bf16: torch.Tensor, seed: torch.Tensor | None):
        if self.probe.kind == "bf16":
            return master_bf16
        return self.probe.own_cast_fp8(index, master_bf16, seed)

    @staticmethod
    def _values(forward, shape: torch.Size) -> torch.Tensor:
        return forward if isinstance(forward, torch.Tensor) else dequantize(*forward, shape)

    @torch.no_grad()
    def step(self, step: int) -> dict[str, float]:
        """Call after ``set_fp8_rounding_step(model, step)`` and before the forward of ``step``."""
        seed = self.probe.forward_seed(step)
        first = self.first_forward is None
        if first:
            self.first_forward, self.previous_forward, self.previous_master = [], [], []
        counts = torch.zeros(5, dtype=torch.float64, device="cuda")
        for i, param in enumerate(self.probe.params):
            master = _local(param).detach().to(torch.bfloat16, copy=True)
            forward = self._forward_value(i, master, seed)
            values = self._values(forward, master.shape)
            if first:
                self.first_forward.append(forward)
                self.previous_forward.append(forward)
                self.previous_master.append(master)
            counts[0] += (values != self._values(self.previous_forward[i], master.shape)).sum()
            counts[1] += (values != self._values(self.first_forward[i], master.shape)).sum()
            counts[2] += (master != self.previous_master[i]).sum()
            counts[3] += (master != self.probe.start[i]).sum()
            counts[4] += master.numel()
            self.previous_forward[i] = forward
            self.previous_master[i] = master
        if dist.is_initialized():
            dist.all_reduce(counts)
        changed_prev, changed_first, master_prev, master_start, numel = counts.tolist()
        metrics = {
            "fp8_flip/forward_changed_vs_first": changed_first / numel,
            "fp8_flip/master_changed_vs_start": master_start / numel,
        }
        if not first:
            metrics["fp8_flip/forward_changed_vs_prev"] = changed_prev / numel
            metrics["fp8_flip/master_changed_vs_prev"] = master_prev / numel
        return metrics
