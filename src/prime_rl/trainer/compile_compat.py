"""PyTorch 2.13 compatibility for runtime autotuning under checkpointing.

Modes opting out of compiler internals must not observe runtime
autotuning allocations or benchmark executions during checkpoint recomputation.
"""

from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack, _pop_mode, _push_mode


@contextmanager
def _disable_modes_for_compile_internals():
    modes = _get_current_dispatch_mode_stack()
    retained = [mode for mode in modes if not mode.ignore_compile_internals()]
    if len(retained) == len(modes):
        yield
        return

    for _ in modes:
        _pop_mode()
    for mode in retained:
        _push_mode(mode)
    try:
        yield
    finally:
        for _ in retained:
            _pop_mode()
        for mode in modes:
            _push_mode(mode)


def apply_autotune_checkpoint_patch() -> None:
    """Apply the tested runtime backport once; refuse unvalidated Torch versions."""
    if torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError(f"Autotuning checkpoint backport requires Torch 2.13.0; got {torch.__version__}")

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    for name in ("bench", "autotune_to_one_config", "coordinate_descent_tuning"):
        original = getattr(CachingAutotuner, name)
        if getattr(original, "_prime_rl_checkpoint_backport", False):
            continue
        wrapped = _disable_modes_for_compile_internals()(original)
        wrapped._prime_rl_checkpoint_backport = True
        setattr(CachingAutotuner, name, wrapped)
