"""Environment implementations. Importing a subpackage registers its envs."""

from __future__ import annotations

ALL_SUITES: tuple[str, ...] = (
    "dummy",
    "minigrid",
    "minihack",
    "atari",
    "miniatari",
    "craftax",
    "craftaxfull",
    "nethack",
    "agentick",
    "procgen",
    "classics",
)


def _import_all_suites() -> None:
    """Force-import every suite so side-effect registration populates REGISTRY."""
    for suite in ALL_SUITES:
        __import__(f"glyphbench.envs.{suite}")
