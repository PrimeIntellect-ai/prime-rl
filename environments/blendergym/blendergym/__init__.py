"""BlenderGym verifiers environment package.

The ``load_environment`` factory is the entry-point used by
``verifiers.load_environment("blendergym")`` and prime-rl's orchestrator.

For the Blender subprocess wrapper, import the submodule directly:

    from blendergym.render import run_blender, RenderResult

Imports are lazy (PEP 562) so that submodules like
``blendergym.assets.pipeline_render_script`` can be imported inside
Blender's embedded Python without pulling in heavy dependencies
(datasets, httpx, …) that are unavailable there.
"""

__all__ = [
    "ArtifactManager",
    "ArtifactPolicy",
    "BlenderGymEnv",
    "BlenderGymRubric",
    "RolloutPaths",
    "TurnPaths",
    "build_dataset",
    "load_environment",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ArtifactManager": (".artifact_manager", "ArtifactManager"),
    "ArtifactPolicy": (".artifact_manager", "ArtifactPolicy"),
    "RolloutPaths": (".artifact_manager", "RolloutPaths"),
    "TurnPaths": (".artifact_manager", "TurnPaths"),
    "build_dataset": (".dataset", "build_dataset"),
    "BlenderGymEnv": (".env", "BlenderGymEnv"),
    "load_environment": (".env", "load_environment"),
    "BlenderGymRubric": (".rubric", "BlenderGymRubric"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val  # cache for subsequent accesses
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
