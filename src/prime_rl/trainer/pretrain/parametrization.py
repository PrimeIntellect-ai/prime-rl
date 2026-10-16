from collections import defaultdict

from torch import nn

from prime_rl.configs.pretrain import PretrainAdamWConfig, PretrainMuonConfig
from prime_rl.trainer.models.init import param_role
from prime_rl.trainer.models.layers.attn import FlashAttention
from prime_rl.utils.weights import resolve_fqn


def param_groups(
    model: nn.Module, optim: PretrainAdamWConfig | PretrainMuonConfig, m_n: float, m_l: float, ep_enabled: bool
) -> list[dict]:
    """Optimizer parameter groups per role for CompleteP with alpha = 1 (Dey et al. 2025). ``optim.lr``,
    ``optim.eps`` and ``optim.weight_decay`` are the values for the base shape; each role scales
    them by the width and depth multipliers. With both multipliers at 1 this is the standard
    parametrization, where only norms and biases differ (no weight decay). The residual output projections and the LM head
    carry CompleteP's 1/m_L and 1/m_N forward multipliers folded into their weights (see
    ``role_std``), which turns their rules into the ones below.

    With Muon, the hidden matrices and residual output projections are Muon groups with
    ``rms_norm`` learning-rate adjustment, whose update size grows as sqrt(m_N); their learning
    rate is scaled by 1/sqrt(m_N) to compensate (and weight decay by sqrt(m_N), keeping
    lr * weight_decay fixed). Everything else is trained with AdamW, as with ``optim.type = "adamw"``.

    MoE experts follow the same roles per expert matrix (``rms_norm`` adjusts each expert by its
    own shape). The router maps width to a fixed number of experts, so its spectral-norm target
    sqrt(num_experts / d) shrinks as m_N^-1/2: with Muon its learning rate is scaled by 1/m_N;
    with AdamW it follows the hidden rule.
    """
    # (lr, eps, weight decay) scale per role for AdamW. PyTorch AdamW and Dion decay by lr * weight_decay.
    adam_scales = {
        "embedding": (1.0, 1 / m_n, 1.0),
        "hidden": (1 / m_n, 1 / (m_n * m_l), m_n),
        "router": (1 / m_n, 1 / (m_n * m_l), m_n),
        "residual_out": (1 / (m_n * m_l), 1 / m_n, m_n * m_l),
        "output": (1 / m_n, 1.0, m_n),
        "norm": (1.0, 1 / m_n, 0.0),
        "bias": (1.0, 1 / m_n, 0.0),
    }
    # (lr, weight decay) scale per role for Muon with rms_norm adjustment.
    muon_scales = {
        "hidden": (m_n**-0.5, m_n**0.5),
        "router": (1 / m_n, m_n),
        "residual_out": (m_n**-0.5 / m_l, m_n**0.5 * m_l),
    }

    params = defaultdict(list)
    for name, param in model.named_parameters():
        if param.requires_grad:
            name = resolve_fqn(model, name)
            role = "router" if ".mlp.router." in name and param.ndim == 2 else param_role(name, param)
            params[role, ".mlp.experts." in name].append(param)

    groups = []
    for (role, experts), role_params in params.items():
        if optim.type == "muon" and role in muon_scales:
            lr, wd = muon_scales[role]
            group = dict(
                params=role_params,
                algorithm="muon",
                adjust_lr="rms_norm",
                lr=optim.lr * lr,
                weight_decay=optim.weight_decay * wd,
            )
            if experts and ep_enabled:
                group["distributed_mesh_name"] = "dp_shard_mod_ep"
            groups.append(group)
            continue
        lr, eps, wd = adam_scales[role]
        group = dict(params=role_params, lr=optim.lr * lr, weight_decay=optim.weight_decay * wd)
        if optim.type == "muon":
            group.update(algorithm="adamw", epsilon=optim.eps * eps)
        else:
            group["eps"] = optim.eps * eps
        groups.append(group)
    return groups


def head_block_partitions(model: nn.Module, heads_per_block: int) -> dict[nn.Parameter, tuple[int, ...]]:
    """Muon row partitions that split each attention query, key and value projection into blocks of
    ``heads_per_block`` heads, so that Muon orthogonalizes every block on its own."""
    partitions = {}
    for module in model.modules():
        if not isinstance(module, FlashAttention):
            continue
        block_rows = heads_per_block * module.head_dim
        blocks = [(block_rows,) * (size // block_rows) for size in module.qkv_sizes]
        if module.qkv_proj is None:
            for projection, projection_blocks in zip((module.q_proj, module.k_proj, module.v_proj), blocks):
                partitions[projection.weight] = projection_blocks
        else:
            partitions[module.qkv_proj.weight] = sum(blocks, ())
    return partitions
