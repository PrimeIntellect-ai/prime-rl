"""BlenderGym internal entity model (runtime schema).

This module defines the *BlenderGym-owned* structures that live inside the
verifiers state dict under a single key: ``state["rollout"]``.

Public, durable artifacts are emitted by :mod:`blendergym.trajectory_writer`
as ``meta.json`` / ``trajectory.json`` / ``trajectory.md``. The artifact schema
version is tracked by :data:`SCHEMA_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .render import RenderResult


SCHEMA_VERSION = "blendergym-trajectory-v2"


ExitStatus = Literal["ok", "xml_parse_failed", "render_failed", "timeout"]


@dataclass
class TurnRecord:
    """One turn (one model response + optional Blender render)."""

    # ---- identity ----
    turn: int

    # ---- outcome (highest signal) ----
    exit_status: ExitStatus | None = None
    error_hint: str | None = None

    # ---- action ----
    action: str | None = None

    # ---- artifact paths (relative to rollout.work_dir) ----
    render_path: str | None = None
    code_path: str | None = None
    response_path: str = ""
    log_path: str | None = None

    # ---- metrics (lower signal) ----
    duration_s: float | None = None

    # ---- GPU tracking (server-side per-turn assignment) ----
    render_gpu_id: int | None = None

    @property
    def xml_parsed(self) -> bool:
        return self.exit_status is not None and self.exit_status != "xml_parse_failed"

    @property
    def render_success(self) -> bool:
        return self.exit_status == "ok"

    @property
    def timed_out(self) -> bool:
        return self.exit_status == "timeout"

    @classmethod
    def for_turn(cls, turn: int, *, response_path: str | None = None) -> "TurnRecord":
        if response_path is None:
            response_path = f"turn_{turn}/response.txt"
        return cls(turn=turn, response_path=response_path)

    def fill_xml_parse_failure(self) -> None:
        self.exit_status = "xml_parse_failed"
        self.error_hint = "XMLParser could not find <code>...</code>"

    def fill_from_render(
        self,
        result: "RenderResult",
        *,
        rel_code: str | None = None,
        rel_log: str | None = None,
        rel_render: str | None = None,
    ) -> None:
        """Populate fields from one Blender subprocess run.

        When called via ArtifactManager, relative paths are supplied
        explicitly. When called directly (tests, scripts), falls back
        to the original hardcoded paths.
        """
        t = self.turn
        self.action = "execute_blender_code"
        self.code_path = rel_code if rel_code is not None else f"turn_{t}/code.py"
        self.log_path = rel_log if rel_log is not None else f"turn_{t}/blender.log"
        self.duration_s = round(result.duration_s, 3)

        if result.image_paths:
            self.render_path = (
                rel_render if rel_render is not None else f"turn_{t}/render1.png"
            )

        if result.timed_out:
            self.exit_status = "timeout"
            self.error_hint = f"TIMEOUT after {self.duration_s}s"
            return

        if not result.success:
            self.exit_status = "render_failed"
            self.error_hint = self.extract_error_hint(result.stderr)
            return

        self.exit_status = "ok"

    @staticmethod
    def extract_error_hint(stderr: str) -> str | None:
        if not stderr:
            return None
        for line in reversed(stderr.splitlines()):
            if any(k in line for k in ("Error", "Exception", "Traceback")):
                return line.strip() or None
        return None

    def to_timeline_row(self) -> dict[str, str]:
        return {
            "turn": str(self.turn),
            "action": self.action or "-",
            "exit_status": self.exit_status or "-",
            "duration_s": f"{self.duration_s:.2f}" if self.duration_s is not None else "-",
            "error_hint": self.error_hint or "-",
        }


@dataclass(frozen=True)
class Task:
    """Immutable view of one BlenderGym task (one dataset row).

    Built from the ``info`` dict emitted by ``dataset.py``. If dataset.py renames
    ``blend_file_path / goal_image_path / init_image_path / task_dir``, update
    :meth:`from_info` in the same change.

    This is the runtime task view, not a 1:1 copy of ``trajectory.json["task"]``:
    JSON v2 only exposes task_id/task_type/blend_file and relies on the schema
    convention for inputs/{goal,init}.png + inputs/start.py.
    """

    task_id: str
    task_type: str
    blend_file: Path
    goal_image: Path
    init_image: Path
    start_code_path: Path

    @classmethod
    def from_info(cls, info: dict) -> "Task":
        return cls(
            task_id=info["task_id"],
            task_type=info["task_type"],
            blend_file=Path(info["blend_file_path"]),
            goal_image=Path(info["goal_image_path"]),
            init_image=Path(info["init_image_path"]),
            start_code_path=Path(info["task_dir"]) / "start.py",
        )


@dataclass
class Rollout:
    """Mutable runtime model of one BlenderGym rollout.

    This is a BlenderGym runtime object, not a public JSON state. If verifiers
    ever JSON-serializes the full state, exclude ``rollout`` via state_columns or
    add a custom encoder. The durable public contract is ``trajectory.json``
    (schema_version=SCHEMA_VERSION).

    Note: ``rollout.task`` (Task object) is unrelated to verifiers' ``state["task"]``
    (usually a short string like "default").
    """

    # ---- task reference (immutable) ----
    task: Task

    # ---- execution context ----
    trajectory_id: str
    work_dir: Path
    max_turns: int

    # ---- progress ----
    turns: list[TurnRecord] = field(default_factory=list)

    # ---- evaluation ----
    final_reward: float | None = None

    # ---- artifact metadata ----
    # Fixed-schema dict that is mirrored into ``meta.json`` / ``trajectory.json``
    # / ``trajectory.html`` and is intentionally aligned with the columns
    # used by ``log_eval_samples`` in ``src/prime_rl/utils/monitor/wandb.py``,
    # so a wandb sample row can be matched 1:1 to a local artifact directory:
    #
    #   {
    #     "env": str,            # mirrors wandb sample table column ``env``
    #     "split": "train" | "eval",
    #     "example_id": int | None,
    #     "task_id": str,
    #     "task_type": str,
    #     "trajectory_id": str,  # extra (not in wandb table) — needed because
    #                            # one (env, example_id) can produce multiple
    #                            # rollouts (rollouts_per_example > 1).
    #   }
    #
    # Do not stuff debug-only fields, absolute paths, or per-step state in here;
    # path/state is already captured by ``work_dir`` and ``turns``. Keeping the
    # schema stable is what makes the wandb<->local mapping useful.
    metadata: dict[str, object] | None = None

    # ---- prompt caches (not part of JSON artifacts) ----
    start_code_text: str | None = None
    goal_image_data_url: str | None = None
    init_image_data_url: str | None = None

    @property
    def render_count(self) -> int:
        return len(self.turns)

    @property
    def last_turn(self) -> TurnRecord | None:
        return self.turns[-1] if self.turns else None

    @property
    def last_render_path(self) -> Path | None:
        t = self.last_turn
        if t is None or not t.render_path:
            return None
        return self.work_dir / t.render_path

    @property
    def trajectory_short_id(self) -> str:
        return self.trajectory_id[:8]

    @property
    def xml_parsed(self) -> bool:
        t = self.last_turn
        return bool(t and t.xml_parsed)

    @property
    def render_success(self) -> bool:
        t = self.last_turn
        return bool(t and t.render_success)


def require_rollout(state: dict) -> Rollout:
    rollout = state.get("rollout")
    if not isinstance(rollout, Rollout):
        raise RuntimeError("BlenderGym rollout state missing; setup_state likely failed")
    return rollout
