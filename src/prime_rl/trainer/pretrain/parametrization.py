from collections import defaultdict

from torch import nn

from prime_rl.configs.trainer import AdamWConfig, MuonConfig
from prime_rl.trainer.models.init import param_role
from prime_rl.utils.weights import resolve_fqn

# Roles trained by Muon (``rms_norm`` learning-rate adjustment) when ``optim.type = "muon"``; everything else, and
# every role with ``optim.type = "adamw"``, is trained by AdamW.
MUON_ROLES = ("hidden", "router", "residual_out")


def param_groups(model: nn.Module, optim: AdamWConfig | MuonConfig, ep_enabled: bool) -> list[dict]:
    """Optimizer parameter groups per role, all with ``optim.lr`` and ``optim.eps``, and ``optim.weight_decay``
    except for norms and biases (no weight decay). With Muon, the hidden matrices (expert matrices per expert), the
    residual output projections and the MoE router are Muon groups; the embedding, the LM head, norms, biases and the
    Gated Residual projections (``hyper_connection``, as in Qwen3.8-Next) train with AdamW."""
    params = defaultdict(list)
    for name, param in model.named_parameters():
        if param.requires_grad:
            name = resolve_fqn(model, name)
            role = param_role(name, param)
            if role == "hidden" and ".mlp.router." in name:
                role = "router"
            elif role == "hidden" and "hyper_connection" in name:
                role = "hyper_connection"
            params[role, ".mlp.experts." in name].append(param)

    groups = []
    for (role, experts), role_params in params.items():
        weight_decay = 0.0 if role in ("norm", "bias") else optim.weight_decay
        if optim.type == "muon" and role in MUON_ROLES:
            group = dict(
                params=role_params, algorithm="muon", adjust_lr="rms_norm", lr=optim.lr, weight_decay=weight_decay
            )
            if experts and ep_enabled:
                group["distributed_mesh_name"] = "dp_shard_mod_ep"
            groups.append(group)
            continue
        group = dict(params=role_params, lr=optim.lr, weight_decay=weight_decay)
        if optim.type == "muon":
            group.update(algorithm="adamw", epsilon=optim.eps)
        else:
            group["eps"] = optim.eps
        groups.append(group)
    return groups
