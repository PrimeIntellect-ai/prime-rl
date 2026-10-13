from torch import nn

from prime_rl.configs.trainer import AdamWConfig, InitConfig, MuonConfig
from prime_rl.trainer.init import completep_multipliers, param_role
from prime_rl.utils.weights import resolve_fqn


def completep_param_groups(model: nn.Module, init: InitConfig, optim: AdamWConfig | MuonConfig) -> list[dict]:
    """Optimizer parameter groups for CompleteP with alpha = 1 (Dey et al. 2025). ``optim.lr``,
    ``optim.eps`` and ``optim.weight_decay`` are the values for the base shape; each role scales
    them by the width and depth multipliers. The residual output projections and the LM head
    carry CompleteP's 1/m_L and 1/m_N forward multipliers folded into their weights (see
    ``role_std``), which turns their rules into the ones below.

    With Muon, the hidden matrices and residual output projections are Muon groups with
    ``rms_norm`` learning-rate adjustment, whose update size grows as sqrt(m_N); their learning
    rate is scaled by 1/sqrt(m_N) to compensate (and weight decay by sqrt(m_N), keeping
    lr * weight_decay fixed). Everything else is trained with AdamW, as with ``optim.type = "adamw"``.
    """
    if any("mlp.experts" in name for name, _ in model.named_parameters()):
        raise ValueError("CompleteP is only implemented for dense models")
    m_n, m_l = completep_multipliers(init, model.config.hidden_size, model.config.num_hidden_layers)
    # (lr, eps, weight decay) scale per role for AdamW. PyTorch AdamW and Dion decay by lr * weight_decay.
    adam_scales = {
        "embedding": (1.0, 1 / m_n, 1.0),
        "hidden": (1 / m_n, 1 / (m_n * m_l), m_n),
        "residual_out": (1 / (m_n * m_l), 1 / m_n, m_n * m_l),
        "output": (1 / m_n, 1.0, m_n),
        "norm": (1.0, 1 / m_n, 0.0),
        "bias": (1.0, 1 / m_n, 0.0),
    }
    # (lr, weight decay) scale per role for Muon with rms_norm adjustment.
    muon_scales = {"hidden": (m_n**-0.5, m_n**0.5), "residual_out": (m_n**-0.5 / m_l, m_n**0.5 * m_l)}

    params = {role: [] for role in adam_scales}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[param_role(resolve_fqn(model, name), param)].append(param)

    groups = []
    for role, role_params in params.items():
        if not role_params:
            continue
        if optim.type == "muon" and role in muon_scales:
            lr, wd = muon_scales[role]
            groups.append(
                dict(
                    params=role_params,
                    algorithm="muon",
                    adjust_lr="rms_norm",
                    lr=optim.lr * lr,
                    weight_decay=optim.weight_decay * wd,
                )
            )
            continue
        lr, eps, wd = adam_scales[role]
        group = dict(params=role_params, lr=optim.lr * lr, weight_decay=optim.weight_decay * wd)
        if optim.type == "muon":
            group.update(algorithm="adamw", epsilon=optim.eps * eps)
        else:
            group["eps"] = optim.eps * eps
        groups.append(group)
    return groups
