"""Patch RL trainer module functions right after their modules load, so gn_probes observes training without edits to src/."""

import importlib.abc
import os
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

    rescale = float(os.environ.get("GN_LOSS_RESCALE", "1"))
    if rescale != 1.0:
        original_scale = module.scale_gradients_

        def scale_gradients_(manager, model, factor):
            return original_scale(manager, model, factor / rescale)

        module.scale_gradients_ = scale_gradients_


def _patch_loss(module):
    original = module.compute_loss

    rescale = float(os.environ.get("GN_LOSS_RESCALE", "1"))

    def compute_loss(*args, **kwargs):
        _probes().on_compute_loss(kwargs)
        loss, metrics = original(*args, **kwargs)
        return loss * rescale if rescale != 1.0 else loss, metrics

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


def _patch_rl_data(module):
    """Stamp rl and ce weight streams on fake samples, chosen by GN_MC_MODE (none leaves them unchanged)."""
    original = module.FakeDataLoader._get_sample_micro_batch

    def get_sample_micro_batch(self, generator):
        import torch

        micro_batch = original(self, generator)
        input_ids = micro_batch["input_ids"]
        position = torch.arange(input_ids.shape[-1]).unsqueeze(0)
        mode = os.environ.get("GN_MC_MODE", "none")
        if mode == "mod":
            micro_batch["rl_weights"] = (input_ids % 3 != 0).float()
            micro_batch["ce_weights"] = (input_ids % 4 != 0).float() * 0.5
        elif mode == "pow2":
            micro_batch["rl_weights"] = (position < 1024).float()
            micro_batch["ce_weights"] = torch.full(input_ids.shape, 0.5)
        elif mode == "none":
            pass
        elif mode == "rl1000":
            micro_batch["rl_weights"] = (position < 1000).float()
        elif mode == "ce1000":
            micro_batch["rl_weights"] = torch.zeros(input_ids.shape)
            micro_batch["ce_weights"] = (position < 1000).float() * 0.5
        else:
            raise ValueError(f"unknown GN_MC_MODE {mode}")
        return micro_batch

    module.FakeDataLoader._get_sample_micro_batch = get_sample_micro_batch


PATCHES = {
    "prime_rl.trainer.rl.data": _patch_rl_data,
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
