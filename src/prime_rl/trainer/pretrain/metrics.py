import math
from collections import defaultdict, deque
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Partial, Replicate

from prime_rl.trainer.model import compute_expert_load_stats, iter_moe_blocks
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.utils.vlm import get_language_model


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


def routed_k_metrics(model: nn.Module, group: dist.ProcessGroup) -> dict[str, float]:
    """Routed experts per training token since the last call, for variable-k routing, over all ranks: mean, std,
    share of tokens with none and with the maximum (the mean over MoE layers), the worst layer's std, and the
    busiest rank's routed pairs per token over the mean."""
    moes = [moe for moe in iter_moe_blocks(model) if moe.routed_k_counts is not None]
    if not moes:
        return {}
    counts = torch.stack([moe.routed_k_counts for moe in moes])
    for moe in moes:
        moe.routed_k_counts.zero_()
    k = torch.arange(counts.shape[1], device=counts.device)
    # This rank's routed pairs per token, over the global mean: the load the data puts on one rank.
    rank_mean = (counts * k).sum() / counts.sum()
    rank_max = rank_mean.clone()
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(counts, group=group)
    tokens = counts.sum(dim=1)
    mean = (counts * k).sum(dim=1) / tokens
    std = ((counts * (k - mean[:, None]) ** 2).sum(dim=1) / tokens).sqrt()
    return {
        "router/k/mean": mean.mean().item(),
        "router/k/std": std.mean().item(),
        "router/k/std/max": std.max().item(),
        "router/k/zero_frac": (counts[:, 0] / tokens).mean().item(),
        "router/k/max_frac": (counts[:, -1] / tokens).mean().item(),
        "router/k/rank_max_mean": (rank_max / mean.mean()).item(),
    }


def moe_layer_indices(model: nn.Module) -> list[int]:
    """The decoder-layer index of every MoE layer, in the order of `iter_moe_blocks`."""
    return [
        index
        for index, layer in enumerate(get_language_model(model).layers)
        if isinstance(getattr(layer, "mlp", None), MoE)
    ]


def expert_load_metrics(counts: Tensor, layer_indices: list[int]) -> dict[str, float]:
    """Balance of the `[num_moe_layers, num_experts]` expert counts of a whole step, per layer, summarized as the
    mean and the worst layer: `cv`, `max_mean` and `cold_frac` as in `compute_expert_load_stats`, plus `min_mean`
    (least-loaded expert over the mean load) and `entropy` (entropy of the experts' shares of the assignments over
    log(num_experts), 1 = uniform). Also per decoder layer (``layer_indices``): `max_mean` and `top_share`, the
    busiest expert's share of the layer's assignments."""
    metrics = {name: value.item() for name, value in compute_expert_load_stats(counts).items()}
    mean = counts.mean(dim=1)
    share = counts / counts.sum(dim=1, keepdim=True)
    min_mean = counts.amin(dim=1) / mean
    entropy = -(share * share.clamp_min(1e-30).log()).sum(dim=1) / math.log(counts.shape[1])
    for name, values in (("min_mean", min_mean), ("entropy", entropy)):
        metrics[f"expert_load/{name}/mean"] = values.mean().item()
        metrics[f"expert_load/{name}/min"] = values.amin().item()
    for layer, max_mean, top_share in zip(layer_indices, counts.amax(dim=1) / mean, share.amax(dim=1)):
        metrics[f"expert_load/layer_{layer}/max_mean"] = max_mean.item()
        metrics[f"expert_load/layer_{layer}/top_share"] = top_share.item()
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
    if any("hyper_connection" in part for part in parts):
        return "hyper_connection"
    if "router" in parts:
        return "router"
    if "shared_expert" in parts:
        return "shared_experts"
    if "experts" in parts:
        return "experts"
    if "self_attn" in parts:
        return "attn"
    return "mlp"


def group_norms(named_tensors: Iterable[tuple[str, Tensor]], group_of=param_group) -> dict[str, Tensor]:
    """L2 norm of (possibly sharded) tensors per group. The squared norms of the local shards are summed over the
    sharded mesh dimensions only (replicated dimensions hold the same values), one collective per mesh and
    placement."""
    by_layout: dict = defaultdict(lambda: defaultdict(list))
    for name, tensor in named_tensors:
        layout = (tensor.device_mesh, tuple(tensor.placements)) if isinstance(tensor, DTensor) else None
        by_layout[layout][group_of(name)].append(tensor.to_local() if isinstance(tensor, DTensor) else tensor)
    squares: dict[str, Tensor] = {}
    for layout, groups in by_layout.items():
        local = torch.stack(
            [torch.stack(torch._foreach_norm(tensors)).double().square().sum() for tensors in groups.values()]
        )
        if layout is not None:
            mesh, placements = layout
            local = DTensor.from_local(
                local, mesh, [Partial() if placement.is_shard() else Replicate() for placement in placements]
            ).full_tensor()
        for group, square in zip(groups, local):
            squares[group] = squares.get(group, 0) + square
    return {group: square.sqrt() for group, square in squares.items()}


def grad_norms_by_group(model: nn.Module) -> dict[str, Tensor]:
    """Gradient L2 norm per parameter group (see `param_group`)."""
    return group_norms((name, param.grad) for name, param in model.named_parameters() if param.grad is not None)
