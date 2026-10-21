import torch
import torch.distributed as dist
from torch import Tensor, nn

from prime_rl.configs.pretrain import MoEBalanceConfig
from prime_rl.trainer.model import is_tt_moe_model, iter_moe_blocks
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.prime_moe.moe import histogram_quantile_biases


def setup_moe_balance(model: nn.Module, balance: MoEBalanceConfig, micro_batch_size: int) -> list[MoE]:
    """Prepare the MoE layers' load-balancing state for training. Returns the layers with the sequence-wise balance
    loss."""
    if is_tt_moe_model(model) and model.config.model_type != "prime_moe":
        raise ValueError(f"Pretraining MoE models use model_type 'prime_moe', got {model.config.model_type!r}")
    for moe in iter_moe_blocks(model):
        # Allocated uninitialized with the rest of the model.
        if moe.routed_k_counts is not None:
            moe.routed_k_counts.zero_()
        moe.router.quantile_sum.zero_()
        moe.router.quantile_count.zero_()
        moe.router.quantile_balance = balance.bias_update == "quantile"
        if moe.router.quantile_balance and balance.quantile_estimator == "histogram":
            device = moe.router.selection_bias.device
            moe.router.quantile_hist = torch.zeros(moe.router.num_experts, balance.quantile_bins, device=device)
            # The first step's grid; later steps use the previous step's global range.
            moe.router.quantile_grid = torch.tensor([-1.0, 1.0], device=device)
            moe.router.quantile_range = torch.tensor([float("inf"), float("-inf")], device=device)
    moes_with_seq_balance = list(iter_moe_blocks(model)) if balance.seq_aux_coeff > 0 else []
    for moe in moes_with_seq_balance:
        # A micro-batch is `micro_batch_size` packed rows concatenated into one sample; the sequences are the rows.
        moe.balance_rows = micro_batch_size
    return moes_with_seq_balance


@torch.no_grad()
def update_selection_biases(model: nn.Module, tokens_per_expert: Tensor, rate: float) -> None:
    """Aux-loss-free load balancing (https://arxiv.org/abs/2408.15664): move each layer's selection bias by
    ``rate`` towards underloaded experts, given the step's global ``[num_moe_layers, num_experts]`` counts.
    The update is mean-centred (as in Moonlight and torchtitan), which leaves the selection unchanged but keeps
    the biases from drifting together."""
    for moe, load in zip(iter_moe_blocks(model), tokens_per_expert):
        if moe.router.selection_bias is not None:
            delta = torch.sign(load.mean() - load)
            moe.router.selection_bias += rate * (delta - delta.mean())


@torch.no_grad()
def quantile_update_selection_biases(model: nn.Module, group: dist.ProcessGroup) -> None:
    """Quantile balancing: set each layer's selection bias to minus the step's balancing thresholds of its experts
    (``PrimeRouter.record_quantiles``), averaged over the micro-batches and ranks and mean-centred."""
    routers = [moe.router for moe in iter_moe_blocks(model)]
    sums = torch.stack([torch.cat([router.quantile_sum, router.quantile_count[None]]) for router in routers])
    dist.all_reduce(sums, group=group)
    for router, row in zip(routers, sums):
        thresholds = row[:-1] / row[-1]
        router.selection_bias.copy_(thresholds.mean() - thresholds)
        router.quantile_sum.zero_()
        router.quantile_count.zero_()


@torch.no_grad()
def histogram_quantile_update_selection_biases(model: nn.Module, group: dist.ProcessGroup) -> dict[str, float]:
    """Quantile balancing with the histogram estimator: sum the step's histograms over the ranks, set each layer's
    selection bias from them (``histogram_quantile_biases``) and use the step's global range of the required biases
    as the next step's grid. Returns the grid's mean bin width and the share of values in its edge bins, which also
    hold the values outside it."""
    routers = [moe.router for moe in iter_moe_blocks(model)]
    hists = torch.stack([router.quantile_hist for router in routers])
    dist.all_reduce(hists, group=group)
    ranges = torch.stack([torch.stack([-router.quantile_range[0], router.quantile_range[1]]) for router in routers])
    dist.all_reduce(ranges, op=dist.ReduceOp.MAX, group=group)
    bin_width = clamped = 0.0
    for router, hist, (neg_lo, hi) in zip(routers, hists, ranges):
        router.selection_bias.copy_(histogram_quantile_biases(hist, router.quantile_grid, router.top_k))
        bin_width += ((router.quantile_grid[1] - router.quantile_grid[0]) / hist.shape[1]).item()
        clamped += ((hist[:, 0].sum() + hist[:, -1].sum()) / hist.sum()).item()
        router.quantile_grid = torch.stack([-neg_lo, torch.maximum(hi, -neg_lo + 1e-6)])
        router.quantile_hist.zero_()
        router.quantile_range.copy_(torch.tensor([float("inf"), float("-inf")]))
    return {"router/qb/bin_width": bin_width / len(routers), "router/qb/edge_share": clamped / len(routers)}
