"""Master-versus-cast probe for the FP8 weight-dynamics experiments (int4/experiments/fp8-weight-dynamics in research-prod)."""

import math
import re
import time
from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.model import get_load_balance_stats, reshard_module
from prime_rl.trainer.models.kernels.fp8_utils import grouped_per_block_cast_to_fp8_triton
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


def ue8m0_rtn(weight: torch.Tensor) -> torch.Tensor:
    """Dequantized UE8M0 round-to-nearest 128x128 block cast of a 2D or 3D weight, exact in bf16."""
    blocks = weight.unsqueeze(0) if weight.dim() == 2 else weight
    assert blocks.shape[-2] % BLOCK == 0 and blocks.shape[-1] % BLOCK == 0, f"unaligned weight {tuple(weight.shape)}"
    fp8, scale = grouped_per_block_cast_to_fp8_triton(blocks.bfloat16().contiguous(), use_ue8m0=True)
    dequantized = fp8.float() * scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    return dequantized.to(torch.bfloat16).view(weight.shape)


def load_probe_batches(dataset: StatefulIterableDataset, seq_len: int, cp: int, count: int) -> list[Batch]:
    """The first ``count`` packed micro-batches of this rank's share of ``dataset``, as the training loader packs them."""
    packed = iter(CatDataset(dataset, seq_len, pad_to_multiple_of=cp))
    return [cat_collate([next(packed)]) for _ in range(count)]


def _fp8_param_ids(model: nn.Module) -> set[int]:
    ids = set()
    for module in model.modules():
        if isinstance(module, Float8BlockwiseLinear) and module.weight is not None:
            ids.add(id(module.weight))
        elif isinstance(module, GroupedExperts) and isinstance(module.compute, DeepGemmFP8ExpertCompute):
            ids.update(id(param) for param in module.parameters(recurse=False))
    return ids


class MasterCastProbe:
    """Loss and gradient on fixed batches at the bf16 master ``m`` and at its UE8M0 RTN cast ``Q(m)``, both in bf16
    compute, projected on the displacement ``D = bf16(m) - m_0`` of the FP8-covered weights. Leaves weights,
    gradients and optimizer state unchanged."""

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
        names = sorted(name for name in params if FP8_COVERED_PARAM.search(name))
        assert any("self_attn" in name for name in names) and any("experts" in name for name in names), names
        self.params = [params[name] for name in names]
        fp8_ids = _fp8_param_ids(model)
        if fp8_ids:
            covered_ids = {id(param) for param in self.params}
            assert fp8_ids == covered_ids, (
                f"FP8 weights {len(fp8_ids)} differ from the probed weights {len(covered_ids)}: "
                f"{sorted(name for name, param in params.items() if id(param) in fp8_ids ^ covered_ids)}"
            )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for name, param in zip(names, self.params):
            # Global sums add local shards over all ranks, so every element must live on exactly one rank.
            if isinstance(param, DTensor):
                assert not any(p.is_replicate() or p.is_partial() for p in param.placements), (name, param.placements)
                assert param.device_mesh.size() == world_size, (name, param.device_mesh)
            else:
                assert world_size == 1, name
        self.start = [_local(param).detach().to(torch.bfloat16, copy=True) for param in self.params]
        self._check_local_casts(names)
        start_bytes = sum(s.numel() * s.element_size() for s in self.start)
        get_logger().info(
            f"Master-cast probe on {len(self.params)} weights ({start_bytes / 1024**3:.1f} GiB of local bf16 starts), "
            f"{len(batches)} micro-batches per rank"
        )

    def _check_local_casts(self, names: list[str]) -> None:
        dense = next(i for i, param in enumerate(self.params) if param.dim() == 2)
        expert = next(i for i, param in enumerate(self.params) if param.dim() == 3)
        for i in (dense, expert):
            param = self.params[i].detach()
            local_cast = ue8m0_rtn(_local(param))
            if isinstance(param, DTensor):
                local_cast = DTensor.from_local(
                    local_cast, param.device_mesh, param.placements, shape=param.shape, stride=param.stride()
                ).full_tensor()
            assert torch.equal(local_cast, ue8m0_rtn(_full(param))), f"local cast of {names[i]} differs from global"

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

    def measure(self) -> dict[str, float]:
        start_time = time.perf_counter()
        assert all(param.grad is None for param in self.model.parameters()), "run the probe after zero_grad"
        with torch.compiler.set_stance("force_eager"), bf16_compute():
            loss_master, token_count = self._loss_and_backward()
            grads_master = [grad.clone() for grad in self._local_grads()]
            self.model.zero_grad(set_to_none=True)
            masters = [_local(param).detach().clone() for param in self.params]
            self._write(ue8m0_rtn(master) for master in masters)
            loss_cast, _ = self._loss_and_backward()
            self._write(masters)
        with torch.no_grad():
            sums = torch.zeros(9, dtype=torch.float64, device="cuda")
            for master, start, grad_master, grad_cast in zip(masters, self.start, grads_master, self._local_grads()):
                master_bf16 = master.bfloat16()
                cast = ue8m0_rtn(master_bf16)
                cast_displacement = cast.float() - ue8m0_rtn(start).float()
                displacement = master_bf16.float() - start.float()
                grad_change = grad_cast - grad_master
                terms = [
                    displacement.square().sum(),
                    (grad_master * displacement).sum(),
                    (grad_cast * displacement).sum(),
                    grad_change.square().sum(),
                    grad_master.square().sum(),
                    grad_cast.square().sum(),
                    (cast_displacement * displacement).sum(),
                    (cast != master_bf16).sum(),
                    torch.tensor(master.numel()),
                ]
                sums += torch.stack([term.to(sums) for term in terms])
            del masters, grads_master
            self.model.zero_grad(set_to_none=True)
            if dist.is_initialized():
                dist.all_reduce(sums)
        # Keep the probe's routing out of the next training step's statistics.
        get_load_balance_stats(self.model)
        reshard_module(self.model)

        # Gradients of the token-mean loss: FSDP divides the reduced sum by grad_divide_factor.
        grad_scale = self.grad_divide_factor / token_count.item()
        d_sq, grad_master_d, grad_cast_d, grad_change_sq, grad_master_sq, grad_cast_sq, visible, cast_changed, numel = (
            sums.tolist()
        )
        metrics = {
            "master_cast_probe/loss_master": loss_master.item(),
            "master_cast_probe/loss_cast": loss_cast.item(),
            "master_cast_probe/gap": (loss_cast - loss_master).item(),
            "master_cast_probe/displacement_norm": math.sqrt(d_sq),
            "master_cast_probe/grad_norm_master": grad_scale * math.sqrt(grad_master_sq),
            "master_cast_probe/grad_norm_cast": grad_scale * math.sqrt(grad_cast_sq),
            "master_cast_probe/num_tokens": token_count.item(),
            "master_cast_probe/cast_changed_fraction": cast_changed / numel,
        }
        if d_sq > 0:
            d_norm = math.sqrt(d_sq)
            metrics |= {
                "master_cast_probe/descent_master": -grad_scale * grad_master_d / d_norm,
                "master_cast_probe/descent_cast": -grad_scale * grad_cast_d / d_norm,
                "master_cast_probe/forward_visible_fraction": visible / d_sq,
            }
            if grad_change_sq > 0:
                metrics["master_cast_probe/grad_change_cos_displacement"] = (grad_cast_d - grad_master_d) / (
                    math.sqrt(grad_change_sq) * d_norm
                )
        metrics["master_cast_probe/time"] = time.perf_counter() - start_time
        return metrics
