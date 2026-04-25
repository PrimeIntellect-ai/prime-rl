"""Stub flash_attn for ARM64 GB200 builds (no CUDA compile on QEMU).

Uses a meta-path import hook so that ANY `flash_attn.<x>` or
`flash_attn.<x>.<y>...` import resolves to a stub module. Attribute
access on those modules returns a stub that raises NotImplementedError
on call — so module-load-time imports succeed but runtime use surfaces a
clear error.

Supports scenarios where vLLM + ring_flash_attn + transformers perform
eager imports across many submodules (flash_attn.ops, flash_attn.layers,
flash_attn.modules, flash_attn.bert_padding, flash_attn.flash_attn_interface, etc).
"""

__version__ = "2.8.3"  # matches declared pip version


def _stub(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn is stubbed on ARM64 GB200 (no from-source CUDA compile). "
        "Real use of flash_attn kernels requires rebuilding with "
        "docker-arm64-post-install.sh enabled."
    )


# Make any attribute access on `flash_attn` return the stub callable.
def __getattr__(name):  # PEP 562
    # Pretend all top-level names exist and are callable stubs.
    return _stub


import sys as _sys
import types as _types
import importlib.abc as _abc
import importlib.util as _util


class _StubModule(_types.ModuleType):
    """Module whose every attribute is a stub callable."""

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _stub


class _StubFinder(_abc.MetaPathFinder, _abc.Loader):
    """Meta-path finder that synthesizes any `flash_attn.<x>...` submodule.

    The physical `flash_attn/__init__.py` (this file) and
    `flash_attn/flash_attn_interface.py` are resolved by the normal
    filesystem finder first (they have real names that ring_flash_attn
    imports). This finder catches everything else.
    """

    PREFIX = "flash_attn."

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self.PREFIX):
            return None
        # Avoid re-synthesizing names that filesystem packages provide.
        # (e.g. flash_attn.flash_attn_interface is a real file.)
        # A module already being imported will already be in sys.modules
        # at the point this is reached — the finder isn't consulted.
        return _util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        mod = _StubModule(spec.name)
        mod.__path__ = []  # mark as package so deeper submodule imports work
        return mod

    def exec_module(self, module):
        return None


_sys.meta_path.append(_StubFinder())


# Common symbols that some users import directly from `flash_attn`.
flash_attn_func = _stub
flash_attn_varlen_func = _stub
flash_attn_qkvpacked_func = _stub
flash_attn_kvpacked_func = _stub
flash_attn_with_kvcache = _stub
