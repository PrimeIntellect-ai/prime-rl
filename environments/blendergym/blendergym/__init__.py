"""BlenderGym verifiers environment package.

The ``load_environment`` factory is the entry-point used by
``verifiers.load_environment("blendergym")`` and prime-rl's orchestrator.

For the Blender subprocess wrapper, import the submodule directly:

    from blendergym.render import run_blender, RenderResult
"""

from .artifact_manager import ArtifactManager, ArtifactPolicy, RolloutPaths, TurnPaths
from .dataset import build_dataset
from .env import BlenderGymEnv, load_environment
from .rubric import BlenderGymRubric

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
