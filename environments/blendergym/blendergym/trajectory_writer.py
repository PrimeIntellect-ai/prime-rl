"""BlenderGym trajectory persistence.

Lays down three artifacts per rollout, one directory:

* ``meta.json`` — flat summary for dashboards / grep.
* ``trajectory.json`` — schema-versioned full trajectory (steps + metrics).
* ``trajectory.html`` — human-readable timeline + per-turn details. Pure
  inline CSS, **relative** ``./inputs/...`` / ``./turn_N/...`` image refs so
  ``file://`` browser open works without a server (sidesteps Cursor /
  VSCode markdown preview's CSP/symlink limits we hit with the previous
  ``trajectory.md`` + base64 approach).

v2 schema convention (artifact layout, unrelated to the HTML viewer):
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

import html
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, Rollout, TurnRecord

logger = logging.getLogger(__name__)


_HTML_GENERATOR = "blendergym-trajectory-html-v1"
_MAX_DETAILS_CHARS = 80_000

_INLINE_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 24px 32px 64px;
  max-width: 1200px;
  line-height: 1.5;
  color: #1f2328;
  background: #ffffff;
}
@media (prefers-color-scheme: dark) {
  body { color: #e6edf3; background: #0d1117; }
  a { color: #4493f8; }
  pre, code { background: #161b22 !important; color: #e6edf3; }
  th, td { border-color: #30363d !important; }
  thead th { background: #161b22 !important; }
  details { border-color: #30363d !important; }
  summary { background: #161b22 !important; }
  .pill { background: #161b22; border-color: #30363d; }
  .pill.error { background: #4c1219; border-color: #6e2532; color: #ffa198; }
  .pill.ok { background: #0f3a1d; border-color: #195a31; color: #56d364; }
}
h1 { margin: 0 0 8px; font-size: 22px; }
h2 { margin: 28px 0 10px; font-size: 18px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }
h3 { margin: 18px 0 6px; font-size: 15px; }
nav.top { margin: 4px 0 16px; font-size: 13px; }
nav.top a { margin-right: 12px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 13px; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; vertical-align: top; }
thead th { background: #f6f8fa; }
.image-strip { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 16px; }
.image-strip figure { margin: 0; flex: 0 0 auto; max-width: 240px; }
.image-strip img { display: block; width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 4px; background: #f6f8fa; }
.image-strip figcaption { font-size: 11px; text-align: center; color: #656d76; margin-top: 4px; }
.turn-render { max-width: 480px; }
.turn-render img { width: 100%; height: auto; display: block; border: 1px solid #d0d7de; border-radius: 4px; }
pre {
  background: #f6f8fa;
  padding: 10px 12px;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
code { font-family: "SF Mono", Menlo, Consolas, monospace; }
details { border: 1px solid #d0d7de; border-radius: 4px; margin: 8px 0; }
summary { cursor: pointer; padding: 6px 10px; background: #f6f8fa; font-size: 13px; user-select: none; }
details[open] summary { border-bottom: 1px solid #d0d7de; }
details > pre { margin: 0; border: none; border-radius: 0; }
.pill {
  display: inline-block;
  padding: 2px 8px;
  border: 1px solid #d0d7de;
  background: #f6f8fa;
  border-radius: 999px;
  font-size: 12px;
  margin-right: 6px;
}
.pill.ok { background: #dafbe1; border-color: #aceebb; color: #1a7f37; }
.pill.error { background: #ffebe9; border-color: #ffaba8; color: #cf222e; }
.muted { color: #656d76; font-style: italic; }
.kv { font-size: 13px; margin: 4px 0; }
.kv strong { display: inline-block; min-width: 130px; color: #656d76; font-weight: 500; }
"""


def completion_to_text(completion: object) -> str:
    """Concatenate text blocks from a verifiers chat completion.

    Lives at module scope (not on TurnRecord) because it processes verifiers'
    chat schema and runs *before* the TurnRecord is constructed (to write
    ``response.txt``). Pairs with TurnRecord; not part of it.

    Handles three shapes:

    * raw string (``vf-eval`` / fake harnesses)
    * list of dicts (test fixtures, OpenAI-style)
    * list of Pydantic ``AssistantMessage`` (verifiers runtime, see
      ``verifiers.utils.response_utils.parse_response_message``); each msg
      has ``.content: str | list[ContentPart] | None`` where each
      ``ContentPart`` is either a Pydantic model with ``.type`` / ``.text``
      or a dict — we normalize via ``getattr`` + ``.get`` fallback.
    """
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""

    parts: list[str] = []
    for msg in completion:
        content = _msg_content(msg)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                text = _content_block_text(block)
                if text:
                    parts.append(text)
    return "\n".join(p for p in parts if p)


def _msg_content(msg: object) -> object:
    """Return ``msg.content`` regardless of dict / Pydantic shape."""
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def _content_block_text(block: object) -> str | None:
    """Return the text of a content part if it's a text block, else None."""
    if isinstance(block, dict):
        if block.get("type") == "text":
            return block.get("text")
        return None
    if getattr(block, "type", None) == "text":
        return getattr(block, "text", None)
    return None


def write_trajectory_artifacts(
    rollout: Rollout, *, metrics: dict[str, Any] | None = None
) -> None:
    """Emit ``meta.json`` + ``trajectory.json`` + ``trajectory.html`` from a Rollout.

    Each file is written via ``.tmp`` + :func:`os.replace` so an interrupted
    write never leaves a half-truncated artifact behind. Also unlinks any
    legacy ``trajectory.md`` so old-format files don't linger after a
    re-run.
    """
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
        "metadata": rollout.metadata,
    }

    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": rollout.trajectory_id,
        "metadata": rollout.metadata,
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

    _atomic_write_text(
        work_dir / "meta.json", json.dumps(meta, indent=2, default=str)
    )
    _atomic_write_text(
        work_dir / "trajectory.json", json.dumps(trajectory, indent=2, default=str)
    )
    _atomic_write_text(work_dir / "trajectory.html", _render_html(rollout))

    (work_dir / "trajectory.md").unlink(missing_ok=True)


# ---- helpers --------------------------------------------------------------


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (``.tmp`` + ``os.replace``)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _html_escape(value: object) -> str:
    """Escape ``value`` as inline HTML text (no quote escaping).

    ``None`` collapses to ``"-"`` so kv rows like ``duration_s: None`` (which
    happen on the XML-parse-failure path where no render runs) render as
    ``duration_s: -`` instead of leaking the Python literal.
    """
    if value is None:
        return "-"
    return html.escape(str(value), quote=False)


def _html_attr(value: object) -> str:
    """Escape ``value`` for use inside an HTML attribute."""
    return html.escape(str(value), quote=True)


def _read_optional_text(
    work_dir: Path, rel: str | None, *, max_chars: int | None = None
) -> str | None:
    """Return UTF-8 text under ``work_dir/rel`` or ``None`` if missing.

    When ``max_chars`` is set and the file is larger, truncates from the
    *front* and prefixes a notice — Blender logs are noisiest at the tail
    (Python tracebacks come last) so the last N characters are most useful.
    """
    if not rel:
        return None
    full = work_dir / rel
    if not full.is_file():
        return None
    text = full.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and len(text) > max_chars:
        text = (
            f"... truncated, showing last {max_chars} chars of {len(text)} ...\n"
            + text[-max_chars:]
        )
    return text


def _rel_img_src(rel: str | None) -> str | None:
    """Return ``./<rel>`` (forward-slashed) or ``None`` if ``rel`` is falsy."""
    if not rel:
        return None
    return "./" + rel.replace("\\", "/")


# ---- HTML rendering -------------------------------------------------------


def _render_html(rollout: Rollout) -> str:
    body_parts = [
        _render_html_header(rollout),
        _render_html_nav(rollout),
        _render_html_image_strip(rollout),
        _render_html_timeline(rollout.turns),
    ]
    for r in rollout.turns:
        body_parts.append(_render_html_turn(rollout.work_dir, r))
    return _wrap_html(rollout, "\n".join(body_parts))


def _wrap_html(rollout: Rollout, body: str) -> str:
    title = _html_attr(f"{rollout.task.task_id}__{rollout.trajectory_short_id}")
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>\n'
        '<meta charset="utf-8">\n'
        f'<meta name="generator" content="{_html_attr(_HTML_GENERATOR)}">\n'
        f"<title>{title}</title>\n"
        f"<style>{_INLINE_CSS}</style>\n"
        f"</head><body>\n{body}\n</body></html>\n"
    )


def _render_html_header(rollout: Rollout) -> str:
    task_id = _html_escape(rollout.task.task_id)
    short = _html_escape(rollout.trajectory_short_id)
    final_reward = rollout.final_reward
    reward_txt = (
        f"{final_reward:.4f}" if isinstance(final_reward, (int, float)) else "-"
    )
    statuses = [r.exit_status or "-" for r in rollout.turns]
    first_err = next((r.error_hint for r in rollout.turns if r.error_hint), None)

    pills = []
    for s in statuses:
        cls = "pill ok" if s == "ok" else ("pill error" if s and s != "-" else "pill")
        pills.append(f'<span class="{cls}">{_html_escape(s)}</span>')
    pills_html = " ".join(pills) if pills else '<span class="muted">no turns</span>'

    err_block = (
        f'<div class="kv"><strong>first_error_hint:</strong> '
        f'<code>{_html_escape(first_err)}</code></div>'
        if first_err
        else ""
    )
    meta_block = ""
    metadata = rollout.metadata
    if metadata:
        meta_block = (
            f'<div class="kv"><strong>env:</strong> {_html_escape(metadata.get("env"))}</div>\n'
            f'<div class="kv"><strong>split:</strong> {_html_escape(metadata.get("split"))}</div>\n'
            f'<div class="kv"><strong>example_id:</strong> {_html_escape(metadata.get("example_id"))}</div>\n'
            f'<div class="kv"><strong>task_id:</strong> {_html_escape(metadata.get("task_id"))}</div>\n'
            f'<div class="kv"><strong>trajectory_id:</strong> '
            f'<code>{_html_escape(metadata.get("trajectory_id"))}</code></div>\n'
        )

    return (
        f"<h1>{task_id}__{short}</h1>\n"
        f"{meta_block}"
        f'<div class="kv"><strong>final_reward:</strong> {_html_escape(reward_txt)}</div>\n'
        f'<div class="kv"><strong>exit_statuses:</strong> {pills_html}</div>\n'
        f"{err_block}"
    )


def _render_html_nav(rollout: Rollout) -> str:
    """Top-of-page nav links to sibling artifacts.

    All links are relative paths next to the HTML file so ``file://`` opens
    keep working regardless of the absolute path the user is viewing from.
    """
    work_dir = rollout.work_dir
    items: list[tuple[str, str]] = []
    for rel, label in [
        ("meta.json", "meta.json"),
        ("trajectory.json", "trajectory.json"),
        ("inputs/start.py", "inputs/start.py"),
    ]:
        if (work_dir / rel).exists():
            items.append((rel, label))
    if not items:
        return ""
    links = " ".join(
        f'<a href="./{_html_attr(rel)}">{_html_escape(label)}</a>'
        for rel, label in items
    )
    return f'<nav class="top">{links}</nav>'


def _render_html_image_strip(rollout: Rollout) -> str:
    """Side-by-side GOAL / INIT / per-turn render thumbnails."""
    work_dir = rollout.work_dir
    figures: list[str] = []

    for rel, label in [("inputs/goal.png", "GOAL"), ("inputs/init.png", "INIT")]:
        if (work_dir / rel).is_file():
            figures.append(_image_figure(rel, label))
        else:
            figures.append(_missing_figure(label))

    for r in rollout.turns:
        label = f"Turn {r.turn}"
        if r.render_path and (work_dir / r.render_path).is_file():
            figures.append(_image_figure(r.render_path, label))
        else:
            figures.append(_missing_figure(label))

    return (
        "<h2>Images</h2>\n"
        f'<div class="image-strip">{"".join(figures)}</div>'
    )


def _image_figure(rel: str, label: str) -> str:
    src = _html_attr(_rel_img_src(rel) or "")
    return (
        f"<figure>"
        f'<a href="{src}"><img src="{src}" alt="{_html_attr(label)}" loading="lazy"></a>'
        f"<figcaption>{_html_escape(label)}</figcaption>"
        f"</figure>"
    )


def _missing_figure(label: str) -> str:
    return (
        f"<figure>"
        f'<div class="muted" style="padding:24px;border:1px dashed #d0d7de;border-radius:4px;">no image</div>'
        f"<figcaption>{_html_escape(label)}</figcaption>"
        f"</figure>"
    )


def _render_html_timeline(turns: list[TurnRecord]) -> str:
    if not turns:
        return '<h2>Timeline</h2>\n<p class="muted">_No turns recorded._</p>'
    columns = ["turn", "action", "exit_status", "duration_s", "error_hint"]
    head = "".join(f"<th>{_html_escape(c)}</th>" for c in columns)
    body_rows: list[str] = []
    for r in turns:
        row = r.to_timeline_row()
        cells = "".join(
            f"<td>{_html_escape(row.get(c, '-'))}</td>" for c in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<h2>Timeline</h2>\n"
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _render_html_turn(work_dir: Path, r: TurnRecord) -> str:
    """One ``<section>`` per turn: render image + state + collapsible files."""
    parts: list[str] = [f"<h2>Turn {r.turn}</h2>"]

    parts.append(
        f'<div class="kv"><strong>action:</strong> '
        f'<code>{_html_escape(r.action or "-")}</code></div>'
    )

    parts.append("<h3>Observation</h3>")
    if r.render_path and (work_dir / r.render_path).is_file():
        src = _html_attr(_rel_img_src(r.render_path) or "")
        parts.append(
            f'<div class="turn-render">'
            f'<a href="{src}"><img src="{src}" alt="Turn {_html_attr(r.turn)} render" loading="lazy"></a>'
            f"</div>"
        )
    else:
        parts.append('<p class="muted">_No render image produced._</p>')
        if r.error_hint:
            parts.append(
                f'<span class="pill error">error_hint: '
                f"{_html_escape(r.error_hint)}</span>"
            )

    parts.append("<h3>State After Turn</h3>")
    state_rows = [
        ("xml_parsed", r.xml_parsed),
        ("render_success", r.render_success),
        ("exit_status", r.exit_status),
        ("duration_s", r.duration_s),
    ]
    for k, v in state_rows:
        parts.append(
            f'<div class="kv"><strong>{_html_escape(k)}:</strong> '
            f"{_html_escape(v)}</div>"
        )

    for path_attr, label, max_chars in [
        ("response_path", "response.txt", None),
        ("code_path", "code.py", None),
        ("log_path", "blender.log", _MAX_DETAILS_CHARS),
    ]:
        rel = getattr(r, path_attr)
        text = _read_optional_text(work_dir, rel, max_chars=max_chars)
        if text is None:
            continue
        parts.append(
            f"<details><summary>{_html_escape(label)}</summary>"
            f"<pre><code>{_html_escape(text)}</code></pre>"
            f"</details>"
        )

    return "\n".join(parts)
