"""Smoke tests for BlenderGymEnv's private rollout state model."""

from __future__ import annotations

import asyncio
from pathlib import Path

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
        gpu_id_pool=(3,),
        max_turns=2,
    )
    state = {
        "trajectory_id": "abcdef1234567890abcdef1234567890",
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
    assert rollout.gpu_id == 3
    assert rollout.max_turns == 2
    assert rollout.start_code_text == "import bpy\n"
    assert rollout.goal_image_data_url.startswith("data:image/png;base64,")
    assert rollout.init_image_data_url.startswith("data:image/png;base64,")
    assert rollout.work_dir.name == "placement1__abcdef12"
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
