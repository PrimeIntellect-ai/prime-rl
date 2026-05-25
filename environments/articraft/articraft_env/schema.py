"""Articraft RL environment runtime schema.

All environment-owned state lives under ``state["rollout"]`` (a :class:`Rollout`).
Freshness tracking replaces ``CompileFeedbackLoop`` with direct dataclass fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.workspace_docs import VirtualWorkspace

MUTATING_TOOL_NAMES = frozenset({"apply_patch", "replace", "write_file"})

SCHEMA_VERSION = "articraft-trajectory-v1"


@dataclass(frozen=True)
class Task:
    """Immutable view of one articraft task (one dataset row)."""

    record_id: str
    prompt_text: str
    category_slug: str | None = None
    sdk_package: str = "sdk"

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> Task:
        return cls(
            record_id=info["record_id"],
            prompt_text=info["prompt_text"],
            category_slug=info.get("category_slug"),
            sdk_package=info.get("sdk_package", "sdk"),
        )


@dataclass
class TurnRecord:
    """One turn (one model response + tool execution results)."""

    turn: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    compile_attempted: bool = False
    compile_success: bool | None = None
    compile_signals: dict[str, Any] | None = None


@dataclass
class Rollout:
    """Mutable runtime model of one articraft RL rollout.

    Freshness tracking (``edit_revision`` / ``last_compile_revision``) mirrors
    ``CompileFeedbackLoop`` private attrs but as public, serializable fields.
    """

    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    script_path: Path
    virtual_workspace: VirtualWorkspace

    # -- progress --
    turns: list[TurnRecord] = field(default_factory=list)

    # -- evaluation --
    final_reward: float | None = None

    # -- artifact metadata --
    metadata: dict[str, object] | None = None

    # -- freshness state (replaces CompileFeedbackLoop internals) --
    edit_revision: int = 0
    last_compile_revision: int = -1
    last_compile_bundle_dict: dict[str, Any] | None = None
    last_compile_attempt_dict: dict[str, Any] | None = None

    # -- termination control --
    compile_required_count: int = 0

    # -- observability --
    last_compile_latency_ms: float | None = None

    def code_is_fresh(self) -> bool:
        """Source: harness_compile.py L85-89 latest_code_is_fresh()"""
        return (
            self.last_compile_revision == self.edit_revision
            and self.last_compile_revision >= 0
        )

    def mark_code_mutated(self, tool_name: str) -> None:
        """Source: harness_compile.py L91-94 + harness.py L864-865"""
        if tool_name not in MUTATING_TOOL_NAMES:
            return
        self.edit_revision += 1

    def mark_compile_attempt(self, bundle: Any) -> None:
        """Store every compile attempt (success or failure) for reward."""
        self.last_compile_attempt_dict = bundle.to_dict()

    def mark_compile_success(self, bundle: Any) -> None:
        """Source: harness_compile.py L191-192"""
        self.last_compile_revision = self.edit_revision
        self.last_compile_bundle_dict = bundle.to_dict()
        self.compile_required_count = 0

    @property
    def trajectory_short_id(self) -> str:
        return self.trajectory_id[:12]


def require_rollout(state: dict[str, Any]) -> Rollout:
    rollout = state.get("rollout")
    if not isinstance(rollout, Rollout):
        raise RuntimeError(
            "Articraft rollout state missing; setup_state likely failed"
        )
    return rollout
