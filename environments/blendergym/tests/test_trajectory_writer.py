"""Unit tests for :mod:`blendergym.trajectory_writer`.

Coverage:

* ``completion_to_text`` handles both raw-string and chat-block completions.
* ``TurnRecord.extract_error_hint`` extracts the right line / returns None.
* ``TurnRecord.fill_from_render`` covers success / failure / timeout.
* ``TurnRecord.to_timeline_row`` emits the expected flat dict.
* ``write_trajectory_artifacts`` lays down meta / json / html and uses the
  per-turn fields correctly (uses a synthetic Rollout, no Blender). Image
  files only need to *exist* on disk — HTML refs them by relative path,
  so a 0-byte ``Path.touch()`` is enough fixture.

Tests deliberately avoid importing the env / rubric, so they run without
torch / open_clip / verifiers fakes (the existing test suite covers those).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from blendergym.render import RenderResult
from blendergym.schema import (
    SCHEMA_VERSION,
    Rollout,
    Task,
    TurnRecord,
    require_rollout,
)
from blendergym.trajectory_writer import (
    completion_to_text,
    write_trajectory_artifacts,
)


def _make_render_result(
    *,
    success: bool,
    image_paths: list[Path] | None = None,
    stderr: str = "",
    duration_s: float = 1.0,
    returncode: int | None = 0,
    timed_out: bool = False,
) -> RenderResult:
    return RenderResult(
        success=success,
        image_paths=list(image_paths or []),
        stderr=stderr,
        duration_s=duration_s,
        returncode=returncode,
        timed_out=timed_out,
    )


def _make_task() -> Task:
    return Task(
        task_id="placement1",
        task_type="placement",
        blend_file=Path("/data/blendergym/placement1/blender_file.blend"),
        goal_image=Path("/data/blendergym/placement1/renders/goal/render1.png"),
        init_image=Path("/data/blendergym/placement1/renders/start/render1.png"),
        start_code_path=Path("/data/blendergym/placement1/start.py"),
    )


def _make_rollout(
    *,
    work_dir: Path,
    turns: list[TurnRecord] | None = None,
    final_reward: float | None = None,
    trajectory_id: str = "abcd1234deadbeef",
    metadata: dict[str, object] | None = None,
) -> Rollout:
    return Rollout(
        task=_make_task(),
        trajectory_id=trajectory_id,
        work_dir=work_dir,
        gpu_id=6,
        max_turns=3,
        turns=list(turns or []),
        final_reward=final_reward,
        metadata=metadata,
    )


def _seed_inputs(work_dir: Path) -> Path:
    inputs = work_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "goal.png").touch()
    (inputs / "init.png").touch()
    (inputs / "start.py").write_text("# start.py", encoding="utf-8")
    return inputs


# ---- completion_to_text ----------------------------------------------------


def test_completion_to_text_handles_str_completion():
    assert completion_to_text("hello") == "hello"


def test_completion_to_text_handles_chat_blocks():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will move the chair to (0.5, 0)."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                {"type": "text", "text": "<code>...</code>"},
            ],
        }
    ]
    out = completion_to_text(msgs)
    assert "I will move" in out
    assert "<code>...</code>" in out
    assert "data:image" not in out


def test_completion_to_text_handles_str_content_messages():
    msgs = [
        {"role": "assistant", "content": "raw string content"},
        {"role": "assistant", "content": [{"type": "text", "text": "blocked"}]},
    ]
    assert "raw string content" in completion_to_text(msgs)
    assert "blocked" in completion_to_text(msgs)


def test_completion_to_text_skips_non_dict_messages():
    assert completion_to_text([None, {"role": "assistant", "content": "ok"}]) == "ok"


# ---- TurnRecord.extract_error_hint -----------------------------------------


def test_extract_error_hint_returns_attribute_error_line():
    stderr = (
        "Saved: '/tmp/render1.png' (took 0:00:02.50)\n"
        "Error: Python: AttributeError: 'Object' object has no attribute 'move_to'\n"
        "[blendergym] some trailing log\n"
    )
    hint = TurnRecord.extract_error_hint(stderr)
    assert hint is not None
    assert "AttributeError" in hint


def test_extract_error_hint_blank_stderr_returns_none():
    assert TurnRecord.extract_error_hint("") is None
    assert TurnRecord.extract_error_hint("Saved: render1.png\nBlender quit\n") is None


def test_extract_error_hint_picks_last_match_when_multiple():
    stderr = "Error: A\nrandom line\nError: B\n"
    assert TurnRecord.extract_error_hint(stderr) == "Error: B"


# ---- TurnRecord.fill_xml_parse_failure -------------------------------------


def test_fill_xml_parse_failure_sets_status():
    r = TurnRecord.for_turn(0)
    r.fill_xml_parse_failure()
    assert r.exit_status == "xml_parse_failed"
    assert r.error_hint is not None
    assert "XMLParser" in r.error_hint
    assert r.xml_parsed is False
    assert r.render_success is False


# ---- TurnRecord.fill_from_render -------------------------------------------


def test_fill_from_render_success(tmp_path):
    render_path = tmp_path / "render1.png"
    render_path.touch()
    result = _make_render_result(
        success=True,
        image_paths=[render_path],
        stderr="",
        duration_s=2.5,
        returncode=0,
    )
    r = TurnRecord.for_turn(0)
    r.fill_from_render(result)
    assert r.exit_status == "ok"
    assert r.action == "execute_blender_code"
    assert r.xml_parsed is True
    assert r.render_success is True
    assert r.timed_out is False
    assert r.render_path == "turn_0/render1.png"
    assert r.error_hint is None
    assert r.duration_s == 2.5


def test_fill_from_render_failed_render():
    result = _make_render_result(
        success=False,
        stderr="AttributeError: 'Object' has no attribute 'move_to'\n",
        duration_s=4.5,
        returncode=1,
    )
    r = TurnRecord.for_turn(1)
    r.fill_from_render(result)
    assert r.exit_status == "render_failed"
    assert r.error_hint is not None
    assert "AttributeError" in r.error_hint
    assert r.render_path is None
    assert r.render_success is False
    assert r.xml_parsed is True
    assert r.timed_out is False


def test_fill_from_render_timeout_takes_precedence_over_no_image():
    result = _make_render_result(
        success=False,
        stderr="...\n[blendergym] TIMEOUT after 120s\n",
        duration_s=120.0,
        returncode=None,
        timed_out=True,
    )
    r = TurnRecord.for_turn(2)
    r.fill_from_render(result)
    assert r.exit_status == "timeout"
    assert r.error_hint is not None
    assert r.error_hint.startswith("TIMEOUT")
    assert r.timed_out is True


def test_turn_record_property_derivation():
    r = TurnRecord.for_turn(0)
    assert r.xml_parsed is False
    assert r.render_success is False
    assert r.timed_out is False

    r.exit_status = "xml_parse_failed"
    assert r.xml_parsed is False
    assert r.render_success is False
    assert r.timed_out is False

    r.exit_status = "render_failed"
    assert r.xml_parsed is True
    assert r.render_success is False
    assert r.timed_out is False

    r.exit_status = "timeout"
    assert r.xml_parsed is True
    assert r.render_success is False
    assert r.timed_out is True

    r.exit_status = "ok"
    assert r.xml_parsed is True
    assert r.render_success is True
    assert r.timed_out is False


def test_rollout_property_derivation(tmp_path):
    rollout = _make_rollout(work_dir=tmp_path, trajectory_id="abcd1234deadbeef")
    assert rollout.render_count == 0
    assert rollout.last_turn is None
    assert rollout.last_render_path is None
    assert rollout.trajectory_short_id == "abcd1234"
    assert rollout.xml_parsed is False
    assert rollout.render_success is False

    failed = TurnRecord.for_turn(0)
    failed.fill_xml_parse_failure()
    rollout.turns.append(failed)
    assert rollout.render_count == 1
    assert rollout.last_turn is failed
    assert rollout.last_render_path is None
    assert rollout.xml_parsed is False
    assert rollout.render_success is False

    ok = TurnRecord.for_turn(1)
    ok.fill_from_render(
        _make_render_result(success=True, image_paths=[tmp_path / "render1.png"])
    )
    rollout.turns.append(ok)
    assert rollout.render_count == 2
    assert rollout.last_turn is ok
    assert rollout.last_render_path == tmp_path / "turn_1" / "render1.png"
    assert rollout.xml_parsed is True
    assert rollout.render_success is True


def test_require_rollout_missing_or_wrong_type(tmp_path):
    with pytest.raises(RuntimeError, match="rollout state missing"):
        require_rollout({})
    with pytest.raises(RuntimeError, match="rollout state missing"):
        require_rollout({"rollout": "not-a-rollout"})

    rollout = _make_rollout(work_dir=tmp_path)
    assert require_rollout({"rollout": rollout}) is rollout


# ---- TurnRecord.to_timeline_row --------------------------------------------


def test_to_timeline_row_for_success(tmp_path):
    rp = tmp_path / "render1.png"
    rp.touch()
    r = TurnRecord.for_turn(0)
    r.fill_from_render(_make_render_result(success=True, image_paths=[rp], duration_s=2.5))
    row = r.to_timeline_row()
    assert row == {
        "turn": "0",
        "action": "execute_blender_code",
        "exit_status": "ok",
        "duration_s": "2.50",
        "error_hint": "-",
    }


def test_to_timeline_row_for_xml_parse_failure():
    r = TurnRecord.for_turn(0)
    r.fill_xml_parse_failure()
    row = r.to_timeline_row()
    assert row["exit_status"] == "xml_parse_failed"
    assert row["duration_s"] == "-"
    assert row["action"] == "-"
    assert row["error_hint"].startswith("XMLParser")


# ---- write_trajectory_artifacts (integration on synthetic state) ----------


def test_write_trajectory_artifacts_emits_meta_json_html(tmp_path):
    work_dir = tmp_path / "placement1__abcd1234"
    work_dir.mkdir()
    _seed_inputs(work_dir)
    turn_dir = work_dir / "turn_0"
    turn_dir.mkdir()
    (turn_dir / "render1.png").touch()
    (turn_dir / "response.txt").write_text("I will move...", encoding="utf-8")
    (turn_dir / "code.py").write_text("import bpy", encoding="utf-8")

    record = TurnRecord.for_turn(0)
    record.fill_from_render(
        _make_render_result(
            success=True, image_paths=[turn_dir / "render1.png"], duration_s=2.5
        )
    )

    metadata = {
        "env": "blendergym",
        "split": "train",
        "example_id": 3,
        "task_id": "placement1",
        "task_type": "placement",
        "trajectory_id": "abcd1234deadbeef",
    }
    rollout = _make_rollout(
        work_dir=work_dir,
        turns=[record],
        final_reward=0.7321,
        metadata=metadata,
    )
    metrics = {"xml_parse_success": 1.0, "render_success": 1.0}
    write_trajectory_artifacts(rollout, metrics=metrics)

    meta = json.loads((work_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["task_id"] == "placement1"
    assert meta["final_reward"] == pytest.approx(0.7321)
    assert meta["exit_statuses"] == ["ok"]
    assert meta["first_error_hint"] is None
    assert meta["num_turns"] == 1
    assert meta["max_turns"] == 3
    assert meta["metadata"] == metadata
    assert "render_success_per_turn" not in meta
    assert "paths" not in meta

    traj = json.loads((work_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert traj["schema_version"] == SCHEMA_VERSION
    assert traj["trajectory_id"] == "abcd1234deadbeef"
    assert traj["metadata"] == metadata
    assert traj["task"]["task_id"] == "placement1"
    assert traj["task"]["task_type"] == "placement"
    assert traj["task"]["blend_file"] == "/data/blendergym/placement1/blender_file.blend"
    assert "goal_image" not in traj["task"]
    assert len(traj["steps"]) == 1
    assert traj["steps"][0]["exit_status"] == "ok"
    assert "xml_parsed" not in traj["steps"][0]
    assert "observation" not in traj["steps"][0]
    assert "extras" not in traj["steps"][0]
    assert traj["final_reward"] == pytest.approx(0.7321)
    assert traj["metrics"]["xml_parse_success"] == 1.0
    assert traj["runtime"]["gpu_id"] == 6
    assert "session_id" not in traj
    assert "agents" not in traj

    html_path = work_dir / "trajectory.html"
    assert html_path.is_file()
    page = html_path.read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>")
    assert 'name="generator" content="blendergym-trajectory-html-v1"' in page
    assert "placement1__abcd1234" in page
    assert "<strong>env:</strong> blendergym" in page
    assert "<strong>split:</strong> train" in page
    assert "<strong>example_id:</strong> 3" in page
    assert "<strong>task_id:</strong> placement1" in page
    assert "<code>abcd1234deadbeef</code>" in page
    # Relative image refs (no base64 / data URI inlining).
    assert '<img src="./inputs/goal.png"' in page
    assert '<img src="./inputs/init.png"' in page
    assert '<img src="./turn_0/render1.png"' in page
    assert "data:image/png;base64," not in page
    # Per-file <details> sections collapse response/code/log.
    assert "<details><summary>response.txt</summary>" in page
    assert "<details><summary>code.py</summary>" in page
    # blender.log doesn't exist, so no entry for it.
    assert "<details><summary>blender.log</summary>" not in page
    # Top nav links to sibling artifacts.
    assert 'href="./meta.json"' in page
    assert 'href="./trajectory.json"' in page
    assert 'href="./inputs/start.py"' in page
    # Timeline table is present.
    assert "<h2>Timeline</h2>" in page
    assert "<table>" in page

    # No leftover .tmp files from atomic write.
    assert not list(work_dir.glob("*.tmp"))


def test_write_trajectory_artifacts_handles_missing_work_dir(tmp_path, caplog):
    rollout = _make_rollout(work_dir=tmp_path / "does_not_exist")
    with caplog.at_level("WARNING"):
        write_trajectory_artifacts(rollout)
    assert any("missing" in rec.message for rec in caplog.records)


def test_write_trajectory_artifacts_with_empty_turns(tmp_path):
    work_dir = tmp_path / "placement1__deadbeef"
    work_dir.mkdir()
    rollout = _make_rollout(work_dir=work_dir, trajectory_id="deadbeef" * 4)
    write_trajectory_artifacts(rollout)
    meta = json.loads((work_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_turns"] == 0
    assert meta["exit_statuses"] == []
    page = (work_dir / "trajectory.html").read_text(encoding="utf-8")
    assert "_No turns recorded._" in page
    # No turn images / no per-turn <details>.
    assert "<details>" not in page


def test_write_trajectory_artifacts_render_failed_turn(tmp_path):
    work_dir = tmp_path / "placement1__failbeef"
    work_dir.mkdir()
    _seed_inputs(work_dir)
    turn_dir = work_dir / "turn_0"
    turn_dir.mkdir()
    (turn_dir / "response.txt").write_text("oops", encoding="utf-8")
    (turn_dir / "code.py").write_text("bad code", encoding="utf-8")
    (turn_dir / "blender.log").write_text(
        "AttributeError: missing attr\n", encoding="utf-8"
    )

    record = TurnRecord.for_turn(0)
    record.fill_from_render(
        _make_render_result(
            success=False,
            stderr="AttributeError: missing attr\n",
            duration_s=1.0,
            returncode=1,
        )
    )
    rollout = _make_rollout(work_dir=work_dir, turns=[record], final_reward=0.0)
    write_trajectory_artifacts(rollout)

    page = (work_dir / "trajectory.html").read_text(encoding="utf-8")
    assert "_No render image produced._" in page
    # No <img src="./turn_0/render1.png"> for the failed turn.
    assert '<img src="./turn_0/render1.png"' not in page
    # Error hint surfaced as a pill.
    assert 'class="pill error"' in page
    assert "AttributeError" in page
    # Log file is rendered as <details>.
    assert "<details><summary>blender.log</summary>" in page


def test_write_trajectory_artifacts_escapes_user_content(tmp_path):
    work_dir = tmp_path / "placement1__escbeef"
    work_dir.mkdir()
    _seed_inputs(work_dir)
    turn_dir = work_dir / "turn_0"
    turn_dir.mkdir()
    (turn_dir / "render1.png").touch()
    payload = "before</pre><script>alert(1)</script>after"
    (turn_dir / "response.txt").write_text(payload, encoding="utf-8")
    (turn_dir / "code.py").write_text(
        "x = '<script>alert(2)</script>'", encoding="utf-8"
    )

    record = TurnRecord.for_turn(0)
    record.fill_from_render(
        _make_render_result(
            success=True, image_paths=[turn_dir / "render1.png"], duration_s=1.0
        )
    )
    record.error_hint = "Error <bad>"
    rollout = _make_rollout(work_dir=work_dir, turns=[record])
    write_trajectory_artifacts(rollout)

    page = (work_dir / "trajectory.html").read_text(encoding="utf-8")
    # User content escaped — no raw <script> tag injected anywhere in body.
    assert "<script>alert(1)</script>" not in page
    assert "<script>alert(2)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in page
    # The naive `</pre>` injection that would close our <pre> early is escaped.
    assert "before&lt;/pre&gt;" in page


def test_write_trajectory_artifacts_truncates_large_log(tmp_path):
    work_dir = tmp_path / "placement1__bigbeef"
    work_dir.mkdir()
    _seed_inputs(work_dir)
    turn_dir = work_dir / "turn_0"
    turn_dir.mkdir()
    (turn_dir / "render1.png").touch()
    (turn_dir / "response.txt").write_text("ok", encoding="utf-8")
    (turn_dir / "code.py").write_text("ok", encoding="utf-8")
    huge = ("X" * 1000 + "\n") * 200  # ~200 KB
    (turn_dir / "blender.log").write_text(huge, encoding="utf-8")

    record = TurnRecord.for_turn(0)
    record.fill_from_render(
        _make_render_result(
            success=True, image_paths=[turn_dir / "render1.png"], duration_s=1.0
        )
    )
    rollout = _make_rollout(work_dir=work_dir, turns=[record])
    write_trajectory_artifacts(rollout)

    page = (work_dir / "trajectory.html").read_text(encoding="utf-8")
    assert "truncated, showing last" in page
    # Cap is 80_000 chars + 60-char prefix + boilerplate; should never embed
    # the whole 200 KB log.
    assert len(page) < 200_000


def test_write_trajectory_artifacts_removes_old_markdown(tmp_path):
    work_dir = tmp_path / "placement1__legacybeef"
    work_dir.mkdir()
    _seed_inputs(work_dir)
    (work_dir / "trajectory.md").write_text("legacy markdown", encoding="utf-8")

    rollout = _make_rollout(work_dir=work_dir)
    write_trajectory_artifacts(rollout)

    assert not (work_dir / "trajectory.md").exists()
    assert (work_dir / "trajectory.html").is_file()


def test_write_trajectory_artifacts_xml_parse_failure_renders_dash_not_none(tmp_path):
    """XML-parse-failure leaves duration_s/render_path as None; HTML must show '-'."""
    work_dir = tmp_path / "placement1__xmlbeef"
    work_dir.mkdir()
    _seed_inputs(work_dir)
    turn_dir = work_dir / "turn_0"
    turn_dir.mkdir()
    (turn_dir / "response.txt").write_text("no <code> block here", encoding="utf-8")

    record = TurnRecord.for_turn(0)
    record.fill_xml_parse_failure()
    rollout = _make_rollout(work_dir=work_dir, turns=[record], final_reward=0.0)
    write_trajectory_artifacts(rollout)

    page = (work_dir / "trajectory.html").read_text(encoding="utf-8")
    assert "duration_s:</strong> None" not in page
    assert "duration_s:</strong> -" in page


def test_completion_to_text_handles_pydantic_assistant_message():
    """verifiers runtime wraps the assistant response in a Pydantic AssistantMessage,
    not a dict — completion_to_text must extract .content via attribute access too."""
    import verifiers as vf  # noqa: F401  -- imported only to make the model class available
    from verifiers.types import AssistantMessage, TextContentPart

    msg_str_content = AssistantMessage(content="plain text reply with <code>...</code>")
    assert completion_to_text([msg_str_content]) == (
        "plain text reply with <code>...</code>"
    )

    msg_list_content = AssistantMessage(
        content=[
            TextContentPart(text="part one"),
            TextContentPart(text="part two"),
        ]
    )
    assert completion_to_text([msg_list_content]) == "part one\npart two"

    # AssistantMessage with content=None (e.g. tool-only response) should yield "".
    msg_none = AssistantMessage(content=None)
    assert completion_to_text([msg_none]) == ""
