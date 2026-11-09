"""Patch RL trainer module functions right after their modules load, so gn_probes observes training without edits to src/."""

import importlib.abc
import sys


def _probes():
    from gn_probes import STATE

    return STATE


def _patch_model(module):
    original = module.setup_model

    def setup_model(*args, **kwargs):
        model = original(*args, **kwargs)
        _probes().attach_model(model)
        return model

    module.setup_model = setup_model


def _patch_utils(module):
    original = module.clip_grad_norm_

    def clip_grad_norm_(manager, model, max_norm):
        _probes().on_grads_ready(manager, model)
        grad_norm = original(manager, model, max_norm)
        _probes().on_grad_norm(grad_norm)
        return grad_norm

    module.clip_grad_norm_ = clip_grad_norm_


def _patch_loss(module):
    original = module.compute_loss

    def compute_loss(*args, **kwargs):
        _probes().on_compute_loss(kwargs)
        return original(*args, **kwargs)

    module.compute_loss = compute_loss


def _patch_optim(module):
    original = module.setup_optimizer

    def setup_optimizer(*args, **kwargs):
        optimizer, manager = original(*args, **kwargs)
        _probes().attach_optimizer(optimizer, manager)
        return optimizer, manager

    module.setup_optimizer = setup_optimizer


def _patch_parallel_dims(module):
    original = module.get_parallel_dims

    def get_parallel_dims(*args, **kwargs):
        parallel_dims = original(*args, **kwargs)
        _probes().parallel_dims = parallel_dims
        return parallel_dims

    module.get_parallel_dims = get_parallel_dims


def _patch_gradient_normalization(module):
    cls = module.GradientNormalization
    original = cls.from_counts

    def from_counts(klass, *args, **kwargs):
        norm = original(*args, **kwargs)
        _probes().on_normalization(norm, args, kwargs)
        return norm

    cls.from_counts = classmethod(from_counts)


PATCHES = {
    "prime_rl.trainer.model": _patch_model,
    "prime_rl.trainer.utils": _patch_utils,
    "prime_rl.trainer.rl.loss": _patch_loss,
    "prime_rl.trainer.optim": _patch_optim,
    "prime_rl.trainer.parallel_dims": _patch_parallel_dims,
    "prime_rl.trainer.gradient_normalization": _patch_gradient_normalization,
}


class PostImportPatcher(importlib.abc.MetaPathFinder):
    """Find each patched module with the remaining finders and run its patch right after the module executes."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            PATCHES[fullname](module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install():
    """Insert the patcher ahead of the default finders."""
    sys.meta_path.insert(0, PostImportPatcher())
