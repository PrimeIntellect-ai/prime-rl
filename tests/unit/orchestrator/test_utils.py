import json

from prime_rl.orchestrator.utils import save_rollouts


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_save_rollouts_excludes_trajectory_when_requested(tmp_path):
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "trajectory": [{"prompt": "p", "completion": "c"}],
    }

    path = tmp_path / "train_rollouts.jsonl"

    save_rollouts([rollout], path, exclude_keys={"trajectory"})

    rows = _read_jsonl(path)
    assert rows == [{"example_id": 1, "reward": 1.0}]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_save_rollouts_preserves_full_trajectory(tmp_path):
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "trajectory": [{"prompt": "p", "completion": "c"}],
    }

    path = tmp_path / "train_rollouts_full.jsonl"

    save_rollouts([rollout], path)

    rows = _read_jsonl(path)
    assert rows == [rollout]
    assert list(tmp_path.glob(".*.tmp")) == []
