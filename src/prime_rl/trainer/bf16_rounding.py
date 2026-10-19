"""Stochastic rounding of FSDP2's fp32 -> bf16 parameter cast (FP8 weight-dynamics experiments in research-prod).

FSDP2 casts each rank's fp32 shard to bf16 before the all-gather, round-to-nearest. This module attaches FSDP2's
all-gather extension (``fsdp_pre_all_gather`` / ``fsdp_post_all_gather``) to every fp32 parameter that FSDP casts to
bf16, so the cast rounds stochastically instead. It is attached after the weights are loaded, by giving each local
shard a subclass identity over the same storage, so loading, checkpoints and the optimizer still see a plain tensor.
The extension is used by every FSDP group, including the 1-rank groups of experts under full expert parallelism, which
skip the multi-rank all-gather path.

The rounding is a pure function of (step seed, parameter name, global rank), so the forward, the activation-checkpoint
recompute and the backward all-gathers of one step see one realization, and a resumed run reproduces it.
"""

from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param import ExtensionsData, FSDPParam, ShardedState

from prime_rl.trainer.models.fp8_rounding import _stream_id
from prime_rl.utils.logger import get_logger

SUPPORTED_TORCH = "2.13"


class _State:
    base_seed = 0
    step_seed = 0
    enabled = True
    cast_elements: dict[str, int] = defaultdict(int)


def sr_to_bf16(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Round fp32 to bf16 up with probability equal to the position in the bf16 bin (unbiased, exact bf16).

    NaNs stay NaN: adding noise to a NaN bit pattern can wrap it to -0.0 or infinity.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    bits = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, device=x.device, generator=generator)
    bits.add_(x.contiguous().view(torch.int32)).bitwise_and_(-65536)
    rounded = bits.view(torch.float32).to(torch.bfloat16)
    return torch.where(x.isnan(), x.to(torch.bfloat16), rounded)


class BF16SRShard(torch.Tensor):
    """An fp32 local shard whose FSDP all-gather input is its stochastically rounded bf16 cast.

    Every op on it behaves as on a plain tensor (``__torch_function__`` is disabled), so only FSDP's extension hooks
    see the subclass.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __reduce_ex__(self, protocol):
        return self.as_subclass(torch.Tensor).__reduce_ex__(protocol)

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        fsdp_param: FSDPParam = self._fsdp_param
        padded = fsdp_param._sharded_param_data.view(fsdp_param.padded_sharded_param_size)
        if _State.enabled and mp_policy.param_dtype == torch.bfloat16:
            seed = _stream_id(f"{_State.step_seed}:{self._salt}:{self._rank}")
            _State.cast_elements[self._kind] += padded.numel()
            return (sr_to_bf16(padded, seed),), None
        return (padded.to(mp_policy.param_dtype),), None

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        (data,) = all_gather_outputs
        if out is not None:
            return
        return data, (data,)


def _eligible(fsdp_param: FSDPParam) -> bool:
    return (
        fsdp_param.orig_dtype == torch.float32
        and fsdp_param.param_dtype == torch.bfloat16
        and fsdp_param.sharded_state == ShardedState.SHARDED
        and not fsdp_param.offload_to_cpu
        and fsdp_param.fsdp_placement.dim == 0
        and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
    )


def install_bf16_stochastic_rounding(model: nn.Module, base_seed: int) -> dict[str, int]:
    """Attach the SR extension to every eligible FSDP parameter of ``model``; returns element counts by kind."""
    assert torch.__version__.startswith(SUPPORTED_TORCH), (
        f"written against torch {SUPPORTED_TORCH}, got {torch.__version__}"
    )
    assert isinstance(model, FSDPModule), "install on the root FSDP module"
    # Lazy init sets the dtype attributes and re-pads shards; attaching after it keeps the extension in place.
    model._get_fsdp_state()._lazy_init()
    _State.base_seed = base_seed
    rank = dist.get_rank() if dist.is_initialized() else 0
    names = {id(param): name for name, param in model.named_parameters()}
    patched, skipped = defaultdict(int), defaultdict(int)
    skipped_names = []
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        fsdp_params = [p for group in module._get_fsdp_state()._fsdp_param_groups for p in group.fsdp_params]
        for fsdp_param in fsdp_params:
            name = names[id(fsdp_param.sharded_param)]
            kind = "expert" if "experts." in name else "dense"
            local = fsdp_param._sharded_local_tensor
            if not _eligible(fsdp_param):
                skipped[kind] += local.numel()
                skipped_names.append(name)
                continue
            shard = local.as_subclass(BF16SRShard)
            shard._fsdp_param = fsdp_param
            shard._salt = _stream_id(name)
            shard._rank = rank
            shard._kind = kind
            fsdp_param.sharded_param._local_tensor = shard
            fsdp_param._extensions_data = ExtensionsData()
            fsdp_param._unsharded_inner_tensors = []
            patched[kind] += local.numel()
    counts = {f"{kind}_patched": n for kind, n in patched.items()} | {
        f"{kind}_skipped": n for kind, n in skipped.items()
    }
    get_logger().info(f"bf16 stochastic rounding (seed {base_seed}) on local elements: {counts}")
    if skipped_names:
        get_logger().info(f"bf16 stochastic rounding skips (round-to-nearest or not cast): {skipped_names}")
    return counts


def set_bf16_rounding_step(step: int) -> None:
    _State.step_seed = _stream_id(f"bf16-sr:{_State.base_seed}:{step}")


@contextmanager
def bf16_rounding_disabled():
    """Cast round-to-nearest inside the block, e.g. for a probe that must match the other arms."""
    previous, _State.enabled = _State.enabled, False
    try:
        yield
    finally:
        _State.enabled = previous


def pop_cast_elements() -> dict[str, int]:
    """Elements SR-cast since the last call, by kind (counting every all-gather)."""
    counts = dict(_State.cast_elements)
    _State.cast_elements.clear()
    return counts
