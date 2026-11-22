"""Legacy Verifiers integration.

The exports remain available for existing evaluation workflows, but are loaded
lazily so importing :mod:`glyphbench` does not initialize Verifiers' legacy
runtime.  Prime-RL training uses GlyphBench's v1 taskset/environment API.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "GlyphbenchMultiTurnEnv": "glyphbench.verifiers_integration.env",
    "load_environment": "glyphbench.verifiers_integration.env",
    "GlyphbenchXMLParser": "glyphbench.verifiers_integration.parser",
    "build_system_prompt": "glyphbench.verifiers_integration.prompting",
    "render_user_turn": "glyphbench.verifiers_integration.prompting",
    "EpisodicReturnRubric": "glyphbench.verifiers_integration.rubric",
}


def __getattr__(name: str):
    if module_name := _EXPORT_MODULES.get(name):
        value = getattr(import_module(module_name), name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "GlyphbenchMultiTurnEnv",
    "GlyphbenchXMLParser",
    "EpisodicReturnRubric",
    "build_system_prompt",
    "render_user_turn",
    "load_environment",
]
