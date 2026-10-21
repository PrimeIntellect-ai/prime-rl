from torch import nn

from prime_rl.configs.trainer import AdamWConfig, MuonConfig
from prime_rl.trainer.optim import _default_muon_param_groups
from prime_rl.trainer.parallel_dims import ParallelDims


def param_groups(model: nn.Module, optim: AdamWConfig | MuonConfig, parallel_dims: ParallelDims) -> list[dict] | None:
    """The default Muon groups, with the Gated Residual projections (``hyper_connection``) moved to the AdamW group.
    None (the default groups) without Muon or Gated Residual.

    These projections produce sigmoid gate logits (the write projection only one scalar per stream), not hidden
    activations. Muon's orthogonalized update moves every direction by the same amount, which makes the gates noisy:
    on Muon they cost +0.028 held-out loss at d384 (4 seeds)."""
    named_params = list(model.named_parameters())
    hyper_connection = [param for name, param in named_params if "hyper_connection" in name and param.requires_grad]
    if optim.type != "muon" or not hyper_connection:
        return None
    rest = [(name, param) for name, param in named_params if "hyper_connection" not in name]
    groups = _default_muon_param_groups(optim, rest, parallel_dims, optim.lr)
    next(group for group in groups if group["algorithm"] == "adamw")["params"].extend(hyper_connection)
    return groups
