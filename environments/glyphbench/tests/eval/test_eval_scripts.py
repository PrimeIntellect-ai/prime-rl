from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from glyphbench import load_environment
from glyphbench.core.task_selection import list_task_ids

ROOT = Path(__file__).resolve().parents[2]


def _capture_prime_args(tmp_path: Path, script: str, overrides: dict[str, str]) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    prime = bin_dir / "prime"
    prime.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARG_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    prime.chmod(0o700)
    capture = tmp_path / "prime_args.json"
    subprocess.run(
        ["/bin/bash", str(ROOT / "eval" / script)],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
            "VIRTUAL_ENV": str(tmp_path),
            "ARG_CAPTURE": str(capture),
            **overrides,
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(capture.read_text(encoding="utf-8"))


def _value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def _assert_all_task_seeds(args: list[str], script: str, episodes: int, rollouts: int) -> None:
    assert args[:3] == ["eval", "run", "glyphbench"]
    config = json.loads(_value(args, "-a"))
    assert config["num_episodes"] == episodes
    env = load_environment(**config)
    inputs = env._get_eval_inputs(
        num_examples=int(_value(args, "-n")),
        rollouts_per_example=int(_value(args, "--rollouts-per-example")),
    )
    if script == "run_full.sh":
        task_ids = list_task_ids(exclude_suites=["atari", "craftaxfull", "nethack", "agentick"])
        assert len(task_ids) == 303
    else:
        task_ids = list_task_ids(include_suites=["atari", "craftaxfull", "nethack"])
        assert len(task_ids) in {58, 59}  # The full Craftax backend is optional.
    routes = [json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
              for row in inputs]
    expected = {
        (task_id, config["seed"] + episode): rollouts
        for task_id in task_ids
        for episode in range(episodes)
    }
    assert Counter((row["env_id"], row["seed"]) for row in routes) == expected
    assert len(inputs) == len(task_ids) * episodes * rollouts


@pytest.mark.parametrize("script,episodes", [("run_full.sh", 3), ("run_archival.sh", 5)])
def test_eval_scripts_evaluate_every_task_and_save_locally(
    tmp_path: Path, script: str, episodes: int,
) -> None:
    args = _capture_prime_args(tmp_path, script, {})

    _assert_all_task_seeds(args, script, episodes=episodes, rollouts=1)
    assert "--skip-upload" in args
    assert "--save-results" in args
    assert "--output-dir" not in args


@pytest.mark.parametrize("script", ["run_full.sh", "run_archival.sh"])
def test_eval_script_overrides_preserve_rollouts_and_output_path(tmp_path: Path, script: str) -> None:
    output = str(tmp_path / "results with spaces")
    args = _capture_prime_args(tmp_path, script, {
        "SKIP_UPLOAD": "0", "SAVE_RESULTS": "0", "EVAL_OUTPUT_DIR": output,
        "EPISODES": "2", "ROLLOUTS_PER_EXAMPLE": "2",
    })

    _assert_all_task_seeds(args, script, episodes=2, rollouts=2)
    assert "--skip-upload" not in args
    assert "--save-results" not in args
    assert _value(args, "--output-dir") == output
