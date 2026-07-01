"""Optimizer hooks for capturing sparse weight diffs during training.

Registers pre/post-step hooks on the optimizer that snapshot parameters before
the step and compute a boolean diff (changed / unchanged) after the step. The
diff is stored in optimizer state as a DTensor, so it is automatically offloaded
by ``CPUOffloadOptimizer`` when CPU optimizer offload is enabled.
"""

import copy
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from torch.nn import Module

logger = get_logger()

SPARSE_DIFF_STATE_KEY = "sparse_diff"


def _get_base_optimizer(optimizer) -> Optimizer:
    """Return the underlying torch.optim.Optimizer, unwrapping CPUOffloadOptimizer."""
    from prime_rl.trainer.optim import CPUOffloadOptimizer

    if isinstance(optimizer, CPUOffloadOptimizer):
        return optimizer.base_optimizer
    return optimizer


def _get_state_dict(optimizer) -> dict:
    """Return the optimizer state dict, unwrapping CPUOffloadOptimizer."""
    from prime_rl.trainer.optim import CPUOffloadOptimizer

    if isinstance(optimizer, CPUOffloadOptimizer):
        return optimizer.state
    return optimizer.state


def _param_to_names(model: "Module") -> dict[torch.nn.Parameter, list[str]]:
    """Build a mapping from parameter objects to their names in the model.

    Tied weights (e.g. lm_head.weight sharing embed_tokens.weight) produce
    multiple names for the same parameter.
    """
    param_to_names: dict[torch.nn.Parameter, list[str]] = {}
    for name, param in model.named_parameters():
        param_to_names.setdefault(param, []).append(name)
    return param_to_names


def setup_sparse_diff_hook(optimizer, model: "Module") -> None:
    """Register pre/post-step hooks to capture boolean weight diffs.

    Must be called after optimizer creation but before the first ``optimizer.step()``.
    The diff is computed as ``param.data.ne(snapshot)`` — a boolean tensor with the
    same sharding as the parameter. It is stored in ``optimizer.state[param]`` under
    the ``"sparse_diff"`` key, so ``CPUOffloadOptimizer`` offloads it automatically.

    Handles ``CPUOffloadOptimizer`` by registering hooks on the inner optimizer;
    the hooks fire between ``_move_states("cuda")`` and ``_move_states("cpu")``,
    so snapshots and diffs are always computed on GPU.
    """
    base_opt = _get_base_optimizer(optimizer)
    param_to_names = _param_to_names(model)
    snapshots: dict[torch.nn.Parameter, torch.Tensor] = {}

    trainable_params = {param: names for param, names in param_to_names.items() if param.requires_grad}

    def pre_step_hook(opt: Optimizer, args, kwargs):
        snapshots.clear()
        for param in trainable_params:
            snapshots[param] = param.data.clone()

    def post_step_hook(opt: Optimizer, args, kwargs):
        for param, snapshot in snapshots.items():
            if isinstance(param.data, DTensor):
                # DTensor doesn't support aten.ne — compute on local tensors
                diff_local = param.data._local_tensor.ne(snapshot._local_tensor)
                diff = DTensor.from_local(diff_local, param.data.device_mesh, param.data.placements)
            else:
                diff = param.data.ne(snapshot)
            state = opt.state.setdefault(param, {})
            state[SPARSE_DIFF_STATE_KEY] = diff
        snapshots.clear()

    base_opt.register_step_pre_hook(pre_step_hook)
    base_opt.register_step_post_hook(post_step_hook)
    logger.debug(f"Registered sparse diff hook on optimizer ({len(trainable_params)} trainable params)")


def get_sparse_diffs(optimizer) -> dict[torch.nn.Parameter, torch.Tensor]:
    """Extract sparse diff tensors from optimizer state.

    Returns a mapping from parameter objects to their boolean diff tensors.
    Tensors may be on CPU (if optimizer state was offloaded) or GPU.
    """
    state = _get_state_dict(optimizer)
    diffs: dict[torch.nn.Parameter, torch.Tensor] = {}
    for param, s in state.items():
        diff = s.get(SPARSE_DIFF_STATE_KEY)
        if diff is not None:
            diffs[param] = diff
    return diffs


def clear_sparse_diffs(optimizer) -> None:
    """Remove sparse diff tensors from optimizer state to free memory."""
    state = _get_state_dict(optimizer)
    for param, s in state.items():
        s.pop(SPARSE_DIFF_STATE_KEY, None)


def ensure_diffs_on_device(optimizer, device: str | torch.device) -> None:
    """Move sparse diff tensors to the target device if they are on CPU.

    Used by the broadcast path to ensure diffs are on GPU for DTensor all-gather.
    Handles both DTensor and regular tensor states.
    """
    state = _get_state_dict(optimizer)
    target = torch.device(device)
    for param, s in state.items():
        diff = s.get(SPARSE_DIFF_STATE_KEY)
        if diff is None:
            continue
        if isinstance(diff, DTensor):
            if diff._local_tensor.device.type == target.type:
                continue
            new_local = diff._local_tensor.to(target, non_blocking=True)
            new_dtensor = copy.copy(diff)
            new_dtensor._local_tensor = new_local
            s[SPARSE_DIFF_STATE_KEY] = new_dtensor
        elif isinstance(diff, torch.Tensor):
            if diff.device.type == target.type:
                continue
            s[SPARSE_DIFF_STATE_KEY] = diff.to(target, non_blocking=True)
