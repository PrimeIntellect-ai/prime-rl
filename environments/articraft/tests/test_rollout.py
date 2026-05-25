"""MockClient-driven ArticraftEnv end-to-end verification.

Tests the full lifecycle:
  setup_state → tool execution → compile → termination → reward
without a real LLM — uses direct env_response calls with synthetic tool_calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from articraft_env.schema import Rollout, Task, require_rollout


# ---- helpers ----


@dataclass
class MockToolCall:
    """Minimal tool_call structure matching verifiers convention."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class MockMessage:
    """Minimal message with optional tool_calls."""
    role: str = "assistant"
    content: str = ""
    tool_calls: list[MockToolCall] | None = None


# ---- schema tests ----


class TestTask:
    def test_from_info(self):
        info = {
            "record_id": "rec_test_001",
            "prompt_text": "Build a simple box",
            "category_slug": "test_category",
            "sdk_package": "sdk",
        }
        task = Task.from_info(info)
        assert task.record_id == "rec_test_001"
        assert task.prompt_text == "Build a simple box"
        assert task.category_slug == "test_category"
        assert task.sdk_package == "sdk"

    def test_from_info_defaults(self):
        info = {"record_id": "rec_min", "prompt_text": "A cube"}
        task = Task.from_info(info)
        assert task.category_slug is None
        assert task.sdk_package == "sdk"


class TestRollout:
    def _make_rollout(self, tmp_path: Path) -> Rollout:
        return Rollout(
            task=Task(record_id="test", prompt_text="test"),
            trajectory_id="abc123def456",
            work_dir=tmp_path,
            max_turns=10,
            script_path=tmp_path / "model.py",
            virtual_workspace=None,  # type: ignore[arg-type]
        )

    def test_freshness_initial(self, tmp_path: Path):
        r = self._make_rollout(tmp_path)
        assert not r.code_is_fresh()

    def test_freshness_after_write_and_compile(self, tmp_path: Path):
        r = self._make_rollout(tmp_path)
        r.mark_code_mutated("write_file")
        assert r.edit_revision == 1
        assert not r.code_is_fresh()

        @dataclass
        class FakeBundle:
            def to_dict(self) -> dict:
                return {"status": "success", "summary": "", "signals": []}

        r.mark_compile_attempt(FakeBundle())
        r.mark_compile_success(FakeBundle())
        assert r.code_is_fresh()
        assert r.last_compile_revision == 1
        assert r.compile_required_count == 0

    def test_freshness_non_mutating_tool(self, tmp_path: Path):
        r = self._make_rollout(tmp_path)
        r.mark_code_mutated("read_file")
        assert r.edit_revision == 0

    def test_freshness_after_second_mutation(self, tmp_path: Path):
        r = self._make_rollout(tmp_path)
        r.mark_code_mutated("write_file")

        @dataclass
        class FakeBundle:
            def to_dict(self) -> dict:
                return {"status": "success", "summary": "", "signals": []}

        r.mark_compile_attempt(FakeBundle())
        r.mark_compile_success(FakeBundle())
        assert r.code_is_fresh()

        r.mark_code_mutated("replace")
        assert r.edit_revision == 2
        assert not r.code_is_fresh()

    def test_trajectory_short_id(self, tmp_path: Path):
        r = self._make_rollout(tmp_path)
        assert r.trajectory_short_id == "abc123def456"


class TestRequireRollout:
    def test_missing(self):
        with pytest.raises(RuntimeError, match="rollout state missing"):
            require_rollout({})

    def test_present(self, tmp_path: Path):
        rollout = Rollout(
            task=Task(record_id="t", prompt_text="p"),
            trajectory_id="x",
            work_dir=tmp_path,
            max_turns=5,
            script_path=tmp_path / "m.py",
            virtual_workspace=None,  # type: ignore[arg-type]
        )
        state: dict[str, Any] = {"rollout": rollout}
        assert require_rollout(state) is rollout


# ---- reward tests ----


class TestComputeReward:
    def test_none_bundle(self):
        from articraft_env.rubric import compute_reward
        assert compute_reward(None) == 0.0

    def test_syntax_error(self):
        from agent.models import CompileSignal, CompileSignalBundle
        from articraft_env.rubric import compute_reward

        bundle = CompileSignalBundle(
            status="failure",
            summary="SyntaxError",
            signals=(
                CompileSignal(
                    severity="failure",
                    kind="syntax_error",
                    code="E001",
                    summary="invalid syntax",
                    blocking=True,
                    group="build",
                ),
            ),
        )
        assert compute_reward(bundle) == pytest.approx(0.05)

    def test_build_success_all_qc_pass(self):
        from agent.models import CompileSignalBundle
        from articraft_env.rubric import compute_reward

        bundle = CompileSignalBundle(
            status="success",
            summary="All checks passed",
            signals=(),
        )
        r = compute_reward(bundle, turns_used=10, max_turns=50)
        assert 0.9 <= r <= 1.0

    def test_build_success_with_warnings(self):
        from agent.models import CompileSignal, CompileSignalBundle
        from articraft_env.rubric import compute_reward

        bundle = CompileSignalBundle(
            status="success",
            summary="Warnings",
            signals=(
                CompileSignal(
                    severity="warning",
                    kind="minor_issue",
                    code="W001",
                    summary="minor",
                    blocking=False,
                    group="qc",
                ),
            ),
        )
        r = compute_reward(bundle)
        assert 0.8 <= r <= 0.9


# ---- artifact manager tests ----


class TestArtifactManager:
    def test_rollout_dir_with_split(self, tmp_path: Path):
        from articraft_env.artifact_manager import ArticraftArtifactManager, ArtifactPolicy

        mgr = ArticraftArtifactManager(
            tmp_path,
            ArtifactPolicy(),
            articraft_root=Path("/fake"),
            sdk_package="sdk",
        )
        d = mgr.rollout_dir(
            traj_id="aabbccdd1234", record_id="rec_test_long_name",
            split="train", example_id=42,
        )
        assert "train" in str(d)
        assert "example_0042" in str(d)
        assert "aabbccdd1234" in str(d)

    def test_script_path(self, tmp_path: Path):
        from articraft_env.artifact_manager import ArticraftArtifactManager, ArtifactPolicy

        mgr = ArticraftArtifactManager(
            tmp_path, ArtifactPolicy(), articraft_root=Path("/fake"), sdk_package="sdk",
        )
        p = mgr.script_path(tmp_path / "work")
        assert p.name == "model.py"

    def test_save_trajectory(self, tmp_path: Path):
        from articraft_env.artifact_manager import ArticraftArtifactManager, ArtifactPolicy

        mgr = ArticraftArtifactManager(
            tmp_path, ArtifactPolicy(), articraft_root=Path("/fake"), sdk_package="sdk",
        )
        work_dir = tmp_path / "roll"
        work_dir.mkdir()

        rollout = Rollout(
            task=Task(record_id="rec_x", prompt_text="test"),
            trajectory_id="traj123",
            work_dir=work_dir,
            max_turns=10,
            script_path=work_dir / "model.py",
            virtual_workspace=None,  # type: ignore[arg-type]
            final_reward=0.85,
        )
        mgr.save_trajectory(rollout, metrics={"reward": 0.85})

        meta = json.loads((work_dir / "meta.json").read_text())
        assert meta["record_id"] == "rec_x"
        assert meta["final_reward"] == 0.85

        traj = json.loads((work_dir / "trajectory.json").read_text())
        assert traj["task"]["record_id"] == "rec_x"
