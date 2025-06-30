import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from zeroband.utils.models import ModelType


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=layer_reshard_after_forward,
        )
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
