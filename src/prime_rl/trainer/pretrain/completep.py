from torch import nn

from prime_rl.configs.trainer import AdamWConfig, InitConfig
from prime_rl.trainer.init import completep_multipliers, param_role
from prime_rl.utils.weights import resolve_fqn


def completep_param_groups(model: nn.Module, init: InitConfig, optim: AdamWConfig) -> list[dict]:
    """AdamW parameter groups for CompleteP with alpha = 1 (Dey et al. 2025). ``optim.lr``,
    ``optim.eps`` and ``optim.weight_decay`` are the values for the base shape; each role scales
    them by the width and depth multipliers. The residual output projections and the LM head
    carry CompleteP's 1/m_L and 1/m_N forward multipliers folded into their weights (see
    ``role_std``), which turns their rules into the ones below."""
    m_n, m_l = completep_multipliers(init, model.config.hidden_size, model.config.num_hidden_layers)
    # (lr, eps, weight decay) scale per role. PyTorch AdamW decays by lr * weight_decay.
    scales = {
        "embedding": (1.0, 1 / m_n, 1.0),
        "hidden": (1 / m_n, 1 / (m_n * m_l), m_n),
        "residual_out": (1 / (m_n * m_l), 1 / m_n, m_n * m_l),
        "output": (1 / m_n, 1.0, m_n),
        "norm": (1.0, 1 / m_n, 0.0),
        "bias": (1.0, 1 / m_n, 0.0),
    }
    params = {role: [] for role in scales}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[param_role(resolve_fqn(model, name), param)].append(param)
    return [
        dict(params=params[role], lr=optim.lr * lr, eps=optim.eps * eps, weight_decay=optim.weight_decay * wd)
        for role, (lr, eps, wd) in scales.items()
        if params[role]
    ]
