"""Shim re-exporting nixl_cu12 under another name.

PI's prime-rl imports `from nixl_cu13._api import ...` but our build of
NIXL 0.10.1 from source produces a `nixl_cu12` package name. Both refer
to the same NIXL source tree — just different packaging metadata. This
shim file is installed as both `nixl/__init__.py` and
`nixl_cu13/__init__.py` so existing import sites resolve without
patching upstream code.
"""

from nixl_cu12 import *  # noqa: F401, F403
from nixl_cu12 import _api, _bindings, _utils, logging  # noqa: F401
import nixl_cu12 as _impl
import sys as _sys

# Ensure `from <self>._api import x` resolves to nixl_cu12._api.x
_sys.modules[__name__ + "._api"] = _impl._api
_sys.modules[__name__ + "._bindings"] = _impl._bindings
_sys.modules[__name__ + "._utils"] = _impl._utils
_sys.modules[__name__ + ".logging"] = _impl.logging
