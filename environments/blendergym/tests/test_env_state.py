"""Smoke tests for BlenderGymEnv's private rollout state model."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from blendergym.env import BlenderGymEnv
from blendergym.schema import Rollout, require_rollout


def test_setup_state_stores_single_rollout_object(tmp_path):
    task_dir = tmp_path / "placement1"
    task_dir.mkdir()
    (task_dir / "start.py").write_text("import bpy\n", encoding="utf-8")
    goal = task_dir / "goal.png"
    init = task_dir / "init.png"
    blend = task_dir / "blender_file.blend"
    goal.write_bytes(b"goal")
    init.write_bytes(b"init")
    blend.write_bytes(b"blend")

    env = BlenderGymEnv(
        data_root=tmp_path,
        work_root=tmp_path / "work",
        max_turns=2,
        env_name="blendergym-train",
    )
    state = {
        "trajectory_id": "abcdef1234567890abcdef1234567890",
        "example_id": 3,
        "info": {
            "task_id": "placement1",
            "task_type": "placement",
            "task_dir": str(task_dir),
            "blend_file_path": str(blend),
            "goal_image_path": str(goal),
            "init_image_path": str(init),
        },
    }

    out = asyncio.run(env.setup_state(state))
    rollout = require_rollout(out)
    assert isinstance(rollout, Rollout)
    assert rollout.task.task_id == "placement1"
    assert rollout.task.task_type == "placement"
    assert rollout.max_turns == 2
    assert rollout.start_code_text == "import bpy\n"
    assert rollout.goal_image_data_url.startswith("data:image/png;base64,")
    assert rollout.init_image_data_url.startswith("data:image/png;base64,")
    assert rollout.work_dir.relative_to(env.work_root) == Path(
        "train/example_0003__placement1/abcdef12"
    )
    assert rollout.metadata == {
        "env": "blendergym-train",
        "split": "train",
        "example_id": 3,
        "task_id": "placement1",
        "task_type": "placement",
        "trajectory_id": "abcdef1234567890abcdef1234567890",
    }
    assert (rollout.work_dir / "inputs" / "goal.png").is_symlink()
    assert (rollout.work_dir / "inputs" / "init.png").is_symlink()
    assert (rollout.work_dir / "inputs" / "start.py").is_symlink()

    # BlenderGym-owned fields now live only under state["rollout"].
    for old_key in (
        "task_id",
        "task_type",
        "start_code",
        "blend_file",
        "work_dir",
        "gpu_id",
        "max_turns",
        "goal_image_data_url",
        "init_image_data_url",
        "render_count",
        "turns",
        "last_render_path",
        "xml_parsed",
        "render_success",
        "trajectory_short_id",
        "final_reward",
    ):
        assert old_key not in out


@pytest.mark.parametrize(
    "split,example_id",
    [
        (None, 3),         # split missing
        ("train", None),    # example_id missing
        ("train", "3"),     # example_id is str, not int
    ],
)
def test_make_work_dir_falls_back_to_legacy_layout(tmp_path, split, example_id):
    """Without ``split`` + integer ``example_id``, work_dir collapses to the
    legacy ``{work_root}/{task_id}__{traj8}`` layout. Ensures ad-hoc /
    standalone usage (no orchestrator-style metadata) keeps working."""
    env = BlenderGymEnv(
        data_root=tmp_path,
        work_root=tmp_path / "work",
        max_turns=1,
    )
    work_dir = env.artifact_manager.make_rollout_dir(
        traj_id="abcdef1234567890",
        task_id="placement1",
        split=split,
        example_id=example_id,
    )
    assert work_dir.relative_to(env.work_root) == Path("placement1__abcdef12")
    assert work_dir.is_dir()


def test_make_work_dir_uses_structured_layout_when_metadata_present(tmp_path):
    env = BlenderGymEnv(
        data_root=tmp_path,
        work_root=tmp_path / "work",
        max_turns=1,
    )
    work_dir = env.artifact_manager.make_rollout_dir(
        traj_id="deadbeefcafef00d",
        task_id="placement42",
        split="eval",
        example_id=7,
    )
    assert work_dir.relative_to(env.work_root) == Path(
        "eval/example_0007__placement42/deadbeef"
    )
    assert work_dir.is_dir()
