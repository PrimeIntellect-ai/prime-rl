import math
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Partial, Replicate

from prime_rl.trainer.model import iter_moe_blocks


def reduce_stats(sums: dict[str, Tensor], maxes: dict[str, Tensor], group: dist.ProcessGroup) -> dict[str, Tensor]:
    """Sum or max per-rank statistics over ``group`` with one all-reduce per kind, and copy them to the CPU once."""
    packed = {}
    for values, op in ((sums, dist.ReduceOp.SUM), (maxes, dist.ReduceOp.MAX)):
        if values:
            flat = torch.cat([value.reshape(-1).to(torch.float64) for value in values.values()])
            dist.all_reduce(flat, op=op, group=group)
            packed[op] = flat
    flat = torch.cat(list(packed.values())).cpu()
    reduced, offset = {}, 0
    for values in (sums, maxes):
        for name, value in values.items():
            reduced[name] = flat[offset : offset + value.numel()].reshape(value.shape)
            offset += value.numel()
    return reduced


def take_routing_counts(model: nn.Module) -> tuple[Tensor, Tensor]:
    """This rank's `[num_moe_layers, num_experts]` expert counts and `[num_moe_layers]` routing-confidence sums
    since the last call, and reset them."""
    moes = list(iter_moe_blocks(model))
    counts = torch.stack([moe.tokens_per_expert for moe in moes])
    confidence = torch.stack([moe.routing_confidence_sum for moe in moes])
    for moe in moes:
        moe.tokens_per_expert.zero_()
        moe.routing_confidence_sum.zero_()
    return counts, confidence


def expert_load_metrics(counts: Tensor) -> dict[str, float]:
    """Balance of the `[num_moe_layers, num_experts]` expert counts of a whole step, per layer, summarized as the
    mean and the worst layer. `max_mean` / `min_mean`: busiest / least-loaded expert over the mean load; `cv`:
    std / mean; `cold_frac`: experts under 0.1x the mean load; `entropy`: entropy of the experts' shares of the
    assignments over log(num_experts) (1 = uniform)."""
    mean = counts.mean(dim=1)
    share = counts / counts.sum(dim=1, keepdim=True)
    per_layer = {
        "max_mean": (counts.amax(dim=1) / mean, "max"),
        "min_mean": (counts.amin(dim=1) / mean, "min"),
        "cv": (counts.std(dim=1, correction=0) / mean, "max"),
        "cold_frac": ((counts < 0.1 * mean[:, None]).double().mean(dim=1), "max"),
        "entropy": (-(share * share.clamp_min(1e-30).log()).sum(dim=1) / math.log(counts.shape[1]), "min"),
    }
    metrics = {}
    for name, (values, worst) in per_layer.items():
        metrics[f"expert_load/{name}/mean"] = values.mean().item()
        metrics[f"expert_load/{name}/{worst}"] = (values.amax() if worst == "max" else values.amin()).item()
    return metrics


class DeadExperts:
    """Experts that received no tokens over the last ``window`` steps."""

    def __init__(self, window: int = 100):
        self.counts: deque[Tensor] = deque(maxlen=window)

    def update(self, counts: Tensor) -> int:
        self.counts.append(counts)
        return int((torch.stack(list(self.counts)).sum(dim=0) == 0).sum())


class SpikeDetector:
    """Flags a value more than ``threshold`` standard deviations above the mean of the previous ``window`` values."""

    def __init__(self, window: int = 128, threshold: float = 6.0):
        self.values: deque[float] = deque(maxlen=window)
        self.threshold = threshold

    def update(self, value: float) -> float:
        spike = 0.0
        if len(self.values) >= 32:
            history = torch.tensor(list(self.values), dtype=torch.float64)
            spike = float(value > history.mean().item() + self.threshold * history.std().item())
        self.values.append(value)
        return spike


def param_group(name: str) -> str:
    """The parameter group a parameter's gradient norm is logged under."""
    parts = name.split(".")
    if "embed_tokens" in parts:
        return "embed"
    if "lm_head" in parts:
        return "lm_head"
    if any("norm" in part for part in parts):
        return "norm"
    if "router" in parts:
        return "router"
    if "shared_expert" in parts:
        return "shared_experts"
    if "experts" in parts:
        return "experts"
    if "self_attn" in parts:
        return "attn"
    return "mlp"


def grad_norms_by_group(model: nn.Module) -> dict[str, Tensor]:
    """Gradient L2 norm per parameter group (see `param_group`). The squared norms of the local shards are summed
    over the sharded mesh dimensions only (replicated dimensions hold the same gradient), one collective per mesh
    and placement."""
    by_layout: dict = defaultdict(lambda: defaultdict(list))
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        layout = (grad.device_mesh, tuple(grad.placements)) if isinstance(grad, DTensor) else None
        by_layout[layout][param_group(name)].append(grad.to_local() if isinstance(grad, DTensor) else grad)
    squares: dict[str, Tensor] = {}
    for layout, groups in by_layout.items():
        local = torch.stack(
            [torch.stack(torch._foreach_norm(grads)).double().square().sum() for grads in groups.values()]
        )
        if layout is not None:
            mesh, placements = layout
            local = DTensor.from_local(
                local, mesh, [Partial() if placement.is_shard() else Replicate() for placement in placements]
            ).full_tensor()
        for group, square in zip(groups, local):
            squares[group] = squares.get(group, 0) + square
    return {group: square.sqrt() for group, square in squares.items()}
