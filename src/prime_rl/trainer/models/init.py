import hashlib
import math
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import InitConfig
from prime_rl.trainer.models.layers.hyper_connection import ExpandedRMSNorm
from prime_rl.trainer.models.layers.moe import TokenChoiceTopKRouter
from prime_rl.utils.weights import resolve_fqn

ParamRole = Literal["embedding", "output", "residual_out", "hidden", "norm", "bias"]

# Elements per random block. Blocks are seeded independently, so each rank draws only the
# blocks that overlap its shard and the values do not depend on how the tensor is sharded.
BLOCK_NUMEL = 1 << 22


def param_role(name: str, param: torch.Tensor) -> ParamRole:
    """Classify a parameter by its canonical name: the initialization and the pretraining optimizer groups are defined
    over these roles."""
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "output"
    if param.ndim == 1:
        if name.endswith("bias"):
            return "bias"
        if "norm" in name:
            return "norm"
        raise ValueError(f"No init role for 1-D parameter {name!r}")
    if name.endswith(("o_proj.weight", "down_proj.weight", "down_proj")):
        return "residual_out"
    return "hidden"


def block_seed(seed: int, name: str, block: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}:{block}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def normal_shard(
    name: str,
    global_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """The ``local_shape`` slice at ``global_offset`` of a standard-normal tensor of
    ``global_shape`` whose values depend only on ``(seed, name)``. The tensor is drawn in
    blocks of rows along dim 0; only the blocks overlapping the slice are generated."""
    row_numel = math.prod(global_shape[1:])
    rows_per_block = max(1, BLOCK_NUMEL // max(row_numel, 1))
    start, stop = global_offset[0], global_offset[0] + local_shape[0]
    pieces = []
    for block in range(start // rows_per_block, math.ceil(stop / rows_per_block)):
        block_start = block * rows_per_block
        block_rows = min(rows_per_block, global_shape[0] - block_start)
        generator = torch.Generator(device=device).manual_seed(block_seed(seed, name, block))
        values = torch.randn((block_rows, *global_shape[1:]), generator=generator, device=device)
        lo, hi = max(start, block_start) - block_start, min(stop, block_start + block_rows) - block_start
        pieces.append(values[lo:hi])
    out = torch.cat(pieces) if pieces else torch.empty((0, *global_shape[1:]), device=device)
    for dim in range(1, len(global_shape)):
        out = out.narrow(dim, global_offset[dim], local_shape[dim])
    return out


@torch.no_grad()
def init_weights(model: nn.Module, config: InitConfig) -> None:
    """Initialize every parameter and the MoE selection biases in place: weight matrices and embeddings from
    N(0, ``config.std``), norm weights at 1, biases at 0. Each rank fills only its local shard, without communication,
    and the values do not depend on the world size or the sharding."""
    for name, param in model.named_parameters():
        # Canonical names, so wrappers such as activation checkpointing don't change the values.
        name = resolve_fqn(model, name)
        role = param_role(name, param)
        local = param.to_local() if isinstance(param, DTensor) else param
        if role == "norm":
            local.fill_(1.0)
            continue
        if role == "bias":
            local.zero_()
            continue
        if isinstance(param, DTensor):
            local_shape, global_offset = compute_local_shape_and_global_offset(
                param.shape, param.device_mesh, param.placements
            )
        else:
            local_shape, global_offset = tuple(param.shape), (0,) * param.ndim
        values = normal_shard(
            name, tuple(param.shape), tuple(local_shape), tuple(global_offset), config.seed, local.device
        )
        local.copy_(values.mul_(config.std))
    # MoE selection biases are buffers: they start at 0, so routing starts from the router scores alone.
    # Gated Residual norm gains are zero-centred (scale 1 + w), so they start at 0.
    for module in model.modules():
        if isinstance(module, TokenChoiceTopKRouter) and module.selection_bias is not None:
            module.selection_bias.zero_()
        if isinstance(module, ExpandedRMSNorm):
            module.weight.zero_()
