"""Articraft verifiers environment package.

The ``load_environment`` factory is the entry-point used by
``verifiers.load_environment("articraft")`` and prime-rl's orchestrator.

Imports are lazy (PEP 562) to keep startup fast.
"""

__all__ = [
    "ArticraftArtifactManager",
    "ArticraftEnv",
    "ArticraftRubric",
    "ArtifactPolicy",
    "build_dataset",
    "load_environment",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ArticraftArtifactManager": (".artifact_manager", "ArticraftArtifactManager"),
    "ArtifactPolicy": (".artifact_manager", "ArtifactPolicy"),
    "build_dataset": (".dataset", "build_dataset"),
    "ArticraftEnv": (".env", "ArticraftEnv"),
    "load_environment": (".env", "load_environment"),
    "ArticraftRubric": (".rubric", "ArticraftRubric"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
