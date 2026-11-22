"""Optional full-game Craftax environment.

The regular ``glyphbench/craftax-*`` tasks are self-contained.  The archival
full-game wrapper is registered only when the pinned Craftax fork is available,
so importing the base package never requires an optional submodule.
"""
from __future__ import annotations

from glyphbench.core.registry import register_env

REGISTRY: dict[str, type] = {}

try:
    # Probe the one extension API the wrapper needs.  PyPI Craftax does not
    # provide it; the repository's pinned public fork does.
    from craftax.craftax import glyphbench_api as _glyphbench_api  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pass
else:
    from glyphbench.envs.craftaxfull.fork import CraftaxForkEnv

    REGISTRY["glyphbench/craftaxfull-v0"] = CraftaxForkEnv

for env_id, cls in REGISTRY.items():
    register_env(env_id, cls, default_use_memory=True)
