"""BlenderGym trajectory persistence.

Lays down three artifacts per rollout, one directory:

* ``meta.json`` — flat summary for dashboards / grep.
* ``trajectory.json`` — schema-versioned full trajectory (steps + metrics).
* ``trajectory.md`` — human-readable timeline + per-turn details (Cursor preview
  uses the inputs/turn_N symlinks as image hosts).

v2 schema convention:
``inputs/{goal,init}.png`` + ``inputs/start.py`` +
``turn_N/{response.txt,code.py,blender.log,render1.png}``.

Migration v1 → v2:

* ``session_id`` → ``trajectory_id``
* ``final_metrics`` → ``metrics`` + top-level ``final_reward``
* removed: ``agents``, ``paths``, ``render_success_per_turn``,
  ``trajectory_short_id``, ``task.{goal_image,init_image,start_code}``
* ``TurnRecord`` removed stored ``xml_parsed/render_success/timed_out`` (now
  properties), plus ``observation``, ``thought``, and ``extras``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import Rollout, SCHEMA_VERSION, TurnRecord

logger = logging.getLogger(__name__)


def completion_to_text(completion: list[dict] | str) -> str:
    """Concatenate text blocks from a verifiers chat completion.

    Lives at module scope (not on TurnRecord) because it processes verifiers'
    chat schema and runs *before* the TurnRecord is constructed (to write
    ``response.txt``). Pairs with TurnRecord; not part of it.
    """
    if isinstance(completion, str):
        return completion
    parts: list[str] = []
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def write_trajectory_artifacts(
    rollout: Rollout, *, metrics: dict[str, Any] | None = None
) -> None:
    """Emit ``meta.json`` + ``trajectory.json`` + ``trajectory.md`` from a Rollout."""
    work_dir = rollout.work_dir
    if not work_dir.is_dir():
        logger.warning("write_trajectory_artifacts: work_dir missing %s", work_dir)
        return

    turns = rollout.turns
    final_reward = rollout.final_reward

    meta = {
        "task_id": rollout.task.task_id,
        "task_type": rollout.task.task_type,
        "trajectory_id": rollout.trajectory_id,
        "final_reward": final_reward,
        "exit_statuses": [r.exit_status for r in turns],
        "first_error_hint": next((r.error_hint for r in turns if r.error_hint), None),
        "num_turns": len(turns),
        "max_turns": rollout.max_turns,
    }

    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": rollout.trajectory_id,
        "task": {
            "task_id": rollout.task.task_id,
            "task_type": rollout.task.task_type,
            "blend_file": str(rollout.task.blend_file),
        },
        "final_reward": final_reward,
        "metrics": metrics or {},
        "num_turns": len(turns),
        "max_turns": rollout.max_turns,
        "steps": [asdict(r) for r in turns],
        "runtime": {"gpu_id": rollout.gpu_id},
    }

    (work_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    (work_dir / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, default=str), encoding="utf-8"
    )
    (work_dir / "trajectory.md").write_text(
        _render_markdown(rollout), encoding="utf-8"
    )


def _render_markdown(rollout: Rollout) -> str:
    work_dir = rollout.work_dir
    turns = rollout.turns
    task_id = rollout.task.task_id
    final_reward = rollout.final_reward
    first_err = next((r.error_hint for r in turns if r.error_hint), None)

    parts: list[str] = []
    parts.append(f"# {task_id}__{rollout.trajectory_short_id}\n")
    parts.append(
        f"- final_reward: {final_reward}\n"
        f"- exit_statuses: {[r.exit_status for r in turns]}\n"
        f"- first_error_hint: {first_err}\n"
    )

    parts.append("## Images\n")
    parts.append(_render_image_table(rollout))

    parts.append("## Timeline\n")
    parts.append(_render_timeline_table([r.to_timeline_row() for r in turns]))

    for r in turns:
        parts.append(_render_turn_section(work_dir, r))

    return "\n\n".join(parts)


def _render_image_table(rollout: Rollout) -> str:
    """Two-row markdown table: GOAL / INIT + each turn's render."""
    turns = rollout.turns
    headers = ["GOAL", "INIT"] + [f"Turn {r.turn}" for r in turns]
    cells = ["![](./inputs/goal.png)", "![](./inputs/init.png)"]
    for r in turns:
        if r.render_path:
            cells.append(f"![](./{r.render_path})")
        else:
            cells.append("_(no render)_")
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_row = "| " + " | ".join(cells) + " |"
    return "\n".join([header_row, sep_row, body_row]) + "\n"


def _render_timeline_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No turns recorded._\n"
    columns = ["turn", "action", "exit_status", "duration_s", "error_hint"]
    header_row = "| " + " | ".join(columns) + " |"
    sep_row = "| " + " | ".join("---" for _ in columns) + " |"
    body_rows = [
        "| " + " | ".join(row.get(c, "-") for c in columns) + " |" for row in rows
    ]
    return "\n".join([header_row, sep_row, *body_rows]) + "\n"


def _render_turn_section(work_dir: Path, r: TurnRecord) -> str:
    """Emit the ``## Turn N`` md section.

    Reads ``response.txt`` / ``code.py`` / ``blender.log`` if present, so this
    function does IO; that's why it lives here, not on TurnRecord.
    """
    lines: list[str] = [f"## Turn {r.turn}", ""]
    lines.append(f"### Action\n\n`{r.action or '-'}`\n")

    lines.append("### Observation\n")
    if r.render_path:
        lines.append(f"![](./{r.render_path})\n")
    else:
        lines.append("_No render image produced._\n")
        if r.error_hint:
            lines.append(f"`error_hint: {r.error_hint}`\n")

    lines.append(
        f"### State After Turn\n\n"
        f"- xml_parsed: {r.xml_parsed}\n"
        f"- render_success: {r.render_success}\n"
        f"- exit_status: {r.exit_status}\n"
        f"- duration_s: {r.duration_s}\n"
    )

    for path_attr, label, lang in [
        ("response_path", "response.txt", "text"),
        ("code_path", "code.py", "python"),
        ("log_path", "blender.log", "text"),
    ]:
        rel = getattr(r, path_attr)
        if not rel:
            continue
        full = work_dir / rel
        if not full.is_file():
            continue
        content = full.read_text(encoding="utf-8", errors="replace")
        lines.append(
            f"<details><summary>{label}</summary>\n\n```{lang}\n{content}\n```\n\n</details>\n"
        )
    return "\n".join(lines)
