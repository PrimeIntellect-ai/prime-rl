from pathlib import Path

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.ckpt import CheckpointManager, Progress
from prime_rl.orchestrator.replay import ReplayBuffer, ReplayGroup, compute_replay_targets


def make_rollout(
    example_id: int,
    *,
    env_name: str = "env",
    policy_step: int = 0,
    advantage: float = 1.0,
    is_filtered: bool = False,
    image_url: str = "file:///tmp/test-image.png",
) -> dict:
    return {
        "example_id": example_id,
        "env_name": env_name,
        "policy_step": policy_step,
        "advantage": advantage,
        "is_filtered": is_filtered,
        "reward": 1.0,
        "trajectory": [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }
                        ],
                    }
                ],
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [False, False],
                    "completion_ids": [3, 4],
                    "completion_mask": [True, True],
                    "completion_logprobs": [-0.1, -0.2],
                },
            }
        ],
        "timing": {"generation_ms": 1.0, "scoring_ms": 1.0},
        "metrics": {},
        "filters": {"zero_advantage": False},
        "is_truncated": False,
    }


def make_group(example_id: int, *, policy_step: int, insert_step: int, env_name: str = "env") -> ReplayGroup:
    return ReplayGroup.from_rollouts(
        [
            make_rollout(example_id * 10, env_name=env_name, policy_step=policy_step),
            make_rollout(example_id * 10 + 1, env_name=env_name, policy_step=policy_step, advantage=2.0),
        ],
        insert_step=insert_step,
    )


def test_compute_replay_targets_for_rollout_batching():
    requested, actual, fresh = compute_replay_targets(
        batch_target=8,
        replay_fraction=0.5,
        available_replay_progress=2,
        rollouts_per_example=2,
        use_token_batching=False,
    )

    assert requested == 4
    assert actual == 2
    assert fresh == 6


def test_compute_replay_targets_for_token_batching():
    requested, actual, fresh = compute_replay_targets(
        batch_target=100,
        replay_fraction=0.25,
        available_replay_progress=18,
        rollouts_per_example=2,
        use_token_batching=True,
    )

    assert requested == 25
    assert actual == 18
    assert fresh == 82


def test_replay_buffer_sampling_is_non_destructive_and_excludes_current_insert_step():
    replay = ReplayBuffer(capacity=4, max_off_policy_steps=8, seed=0)
    replay.add(
        [
            make_group(0, policy_step=1, insert_step=0),
            make_group(1, policy_step=1, insert_step=1),
            make_group(2, policy_step=2, insert_step=2),
        ]
    )

    sampled = replay.sample(
        target_progress=4,
        use_token_batching=False,
        current_step=3,
        exclude_insert_step=2,
    )

    assert len(sampled) == 2
    assert all(group.insert_step != 2 for group in sampled)
    assert len(replay.groups) == 3


def test_replay_buffer_copies_rollouts_on_insert():
    rollout = make_rollout(0, policy_step=1)
    replay = ReplayBuffer(capacity=2, max_off_policy_steps=8, seed=0)
    replay.add([ReplayGroup.from_rollouts([rollout], insert_step=0)])

    rollout["trajectory"][0]["prompt"][0]["content"][0]["image_url"]["url"] = "placeholder"

    sampled = replay.sample(
        target_progress=1,
        use_token_batching=False,
        current_step=1,
    )[0]

    stored_url = sampled.rollouts[0]["trajectory"][0]["prompt"][0]["content"][0]["image_url"]["url"]
    assert stored_url == "file:///tmp/test-image.png"


def test_replay_buffer_returns_deep_copies_for_nested_rollout_data():
    replay = ReplayBuffer(capacity=2, max_off_policy_steps=8, seed=0)
    replay.add([make_group(0, policy_step=1, insert_step=0)])

    sampled = replay.sample(
        target_progress=2,
        use_token_batching=False,
        current_step=2,
    )[0]
    sampled.rollouts[0]["trajectory"][0]["prompt"][0]["content"][0]["image_url"]["url"] = "placeholder"

    sampled_again = replay.sample(
        target_progress=2,
        use_token_batching=False,
        current_step=2,
    )[0]
    stored_url = sampled_again.rollouts[0]["trajectory"][0]["prompt"][0]["content"][0]["image_url"]["url"]
    assert stored_url == "file:///tmp/test-image.png"


def test_replay_buffer_evicts_stale_groups():
    replay = ReplayBuffer(capacity=4, max_off_policy_steps=2, seed=0)
    replay.add(
        [
            make_group(0, policy_step=1, insert_step=0),
            make_group(1, policy_step=4, insert_step=1),
        ]
    )

    replay.evict_stale(current_step=6)

    assert len(replay.groups) == 1
    assert replay.groups[0].policy_step == 4


def test_checkpoint_manager_round_trips_replay(tmp_path: Path):
    class FakeBuffer:
        def __init__(self):
            self.loaded = False

        def save(self, path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "buffer.txt").write_text("buffer")

        def load(self, path: Path) -> None:
            self.loaded = (path / "buffer.txt").read_text() == "buffer"

    progress = Progress(step=3, total_tokens=9, total_samples=4, total_problems=2)
    replay = ReplayBuffer(capacity=4, max_off_policy_steps=8, seed=0)
    replay.add([make_group(0, policy_step=2, insert_step=1)])

    manager = CheckpointManager(tmp_path, CheckpointConfig())
    manager.save(progress, FakeBuffer(), step=3, replay=replay)

    loaded_progress = Progress()
    loaded_replay = ReplayBuffer(capacity=4, max_off_policy_steps=8, seed=0)
    loaded_buffer = FakeBuffer()
    manager.load(loaded_progress, loaded_buffer, step=3, replay=loaded_replay)

    assert loaded_progress.step == 3
    assert loaded_progress.total_tokens == 9
    assert loaded_buffer.loaded
    assert len(loaded_replay.groups) == 1
    assert loaded_replay.groups[0].policy_step == 2
    assert loaded_replay.groups[0].insert_step == 1


def test_checkpoint_manager_skip_replay_leaves_existing_replay_untouched(tmp_path: Path):
    class FakeBuffer:
        def save(self, path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "buffer.txt").write_text("buffer")

        def load(self, path: Path) -> None:
            assert (path / "buffer.txt").read_text() == "buffer"

    progress = Progress(step=3)
    saved_replay = ReplayBuffer(capacity=4, max_off_policy_steps=8, seed=0)
    saved_replay.add([make_group(0, policy_step=2, insert_step=1)])

    manager = CheckpointManager(tmp_path, CheckpointConfig())
    manager.save(progress, FakeBuffer(), step=3, replay=saved_replay)

    preserved_replay = ReplayBuffer(capacity=4, max_off_policy_steps=8, seed=0)
    preserved_replay.add([make_group(9, policy_step=9, insert_step=9)])
    skip_manager = CheckpointManager(tmp_path, CheckpointConfig(skip_replay=True))
    skip_manager.load(Progress(), FakeBuffer(), step=3, replay=preserved_replay)

    assert len(preserved_replay.groups) == 1
    assert preserved_replay.groups[0].policy_step == 9
    assert preserved_replay.groups[0].insert_step == 9
