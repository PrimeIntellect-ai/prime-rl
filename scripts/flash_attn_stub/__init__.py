"""Stub flash_attn package for ARM64 GB200 (no compiled kernels).

Installs an import hook that synthesizes any missing submodule of
flash_attn (e.g. flash_attn.ops, flash_attn.ops.triton.rotary) so
imports succeed at module-load time. The actual kernel functions
raise NotImplementedError if called — callers should use SDPA.
"""
__version__ = "2.7.3"

import sys
import types
import importlib.abc
import importlib.machinery


def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn is stubbed on ARM64 GB200 — use attn='sdpa'")


def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn is stubbed on ARM64 GB200 — use attn='sdpa'")


def flash_attn_supports_top_left_mask():
    return False


def _stub_callable(name):
    def _f(*args, **kwargs):
        raise NotImplementedError(f"flash_attn stub: {name} not implemented on ARM64 GB200")
    _f.__name__ = name
    return _f


class _FlashAttnSubmoduleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Synthesize any flash_attn.* submodule on demand.

    Returns an empty module with a __getattr__ that lazily produces stub
    callables for any attribute access, so imports like
    `from flash_attn.ops.triton.rotary import apply_rotary` succeed
    and `apply_rotary(...)` raises NotImplementedError.
    """

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("flash_attn."):
            return None
        if fullname in sys.modules:
            return None
        # Don't shadow our own real submodules
        if fullname == "flash_attn.flash_attn_interface":
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__file__ = "<flash_attn stub>"
        # __getattr__ returns a stub callable for any name
        def __getattr__(name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _stub_callable(f"{spec.name}.{name}")
        mod.__getattr__ = __getattr__
        return mod

    def exec_module(self, module):
        pass


sys.meta_path.append(_FlashAttnSubmoduleFinder())
