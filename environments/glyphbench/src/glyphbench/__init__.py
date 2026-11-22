"""GlyphBench: unified benchmark of public text-rendered RL environments."""

import os

__version__ = "0.1.1"

from glyphbench.core import (
    REGISTRY,
    ActionSpec,
    BaseGlyphEnv,
    GridObservation,
    all_glyphbench_env_ids,
    all_task_memory_defaults,
    default_use_memory,
    make_env,
    register_env,
)

# Importing any suite module populates REGISTRY via register_env side-effects.
from glyphbench.envs import _import_all_suites as _load_suites

if not os.environ.get("GLYPHBENCH_SKIP_SUITE_AUTOLOAD"):
    _load_suites()
del _load_suites

_LAZY_EXPORTS = {
    "GlyphBenchEnv": "glyphbench.verifiers_v1",
    "GlyphBenchHarness": "glyphbench.verifiers_harness",
    "GlyphBenchTaskset": "glyphbench.verifiers_v1",
    "GlyphbenchMultiTurnEnv": "glyphbench.verifiers_integration",
    "GlyphbenchXMLParser": "glyphbench.verifiers_integration",
    "EpisodicReturnRubric": "glyphbench.verifiers_integration",
    "load_environment": "glyphbench.verifiers_integration",
}


def __getattr__(name: str):
    if module_name := _LAZY_EXPORTS.get(name):
        from importlib import import_module

        value = getattr(import_module(module_name), name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "ActionSpec",
    "BaseGlyphEnv",
    "GridObservation",
    "GlyphBenchEnv",
    "GlyphBenchHarness",
    "GlyphBenchTaskset",
    "REGISTRY",
    "all_glyphbench_env_ids",
    "all_task_memory_defaults",
    "default_use_memory",
    "make_env",
    "register_env",
]
