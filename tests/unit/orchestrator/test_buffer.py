import random

import pytest
from datasets import Dataset

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.vf import Rollout


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"problem": "0"},
            {"problem": "1"},
            {"problem": "2"},
            {"problem": "3"},
            {"problem": "4"},
        ]
    )


@pytest.fixture
def buffer_with_difficulties(dataset: Dataset) -> Buffer:
    buffer = Buffer(dataset, BufferConfig())
    for pid, difficulty in zip(buffer.problem_ids, ["easy", "easy", "normal", "normal", "hard"]):
        buffer.metadata[pid]["difficulty"] = difficulty
    return buffer


@pytest.fixture
def make_rollouts():
    def _make_rollouts(
        dataset: Dataset, rewards: list[float] | None = None, advantages: list[float] | None = None
    ) -> list[Rollout]:
        rollouts = []
        rewards = rewards or [1.0] * len(dataset)
        advantages = advantages or [1.0] * len(dataset)
        for i, (reward, advantage) in enumerate(zip(rewards, advantages)):
            problem_rollouts = [
                Rollout(
                    example_id=i,
                    task="default",
                    prompt_ids=[0],
                    prompt_mask=[1],
                    completion_ids=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    is_truncated=False,
                    reward=reward,
                    advantage=advantage,
                )
            ] * 2
            rollouts.extend(problem_rollouts)
        return rollouts

    return _make_rollouts


def test_buffer_init(dataset):
    """Test basic buffer initialization."""
    buffer = Buffer(dataset, BufferConfig())
    assert len(buffer.metadata) == 5
    assert len(buffer.rollout_buffer) == 0
    assert len(buffer.problem_ids) == 5
    assert all(metadata["difficulty"] == "normal" for metadata in buffer.metadata.values())


def test_buffer_init_without_id_column(dataset):
    """Test that buffer adds 'id' column if missing."""
    # Remove id column if it exists
    dataset_no_id = dataset.remove_columns(["id"]) if "id" in dataset.column_names else dataset
    buffer = Buffer(dataset_no_id, BufferConfig())
    assert "id" in buffer.dataset.column_names
    assert len(buffer.problem_ids) == 5
    assert all(isinstance(pid, int) for pid in buffer.problem_ids)


def test_buffer_sample_problems_default(dataset):
    buffer = Buffer(dataset, BufferConfig())
    sampled = buffer.sample_problems(2)
    assert len(sampled) == 2
    assert all("id" in p for p in sampled)


def test_buffer_sample_problems_mix(buffer_with_difficulties):
    buffer = Buffer(
        buffer_with_difficulties.dataset,
        BufferConfig(easy_fraction=0.5, hard_fraction=0.5, dataset_path=None),
    )
    buffer.metadata = buffer_with_difficulties.metadata.copy()
    sampled = buffer.sample_problems(3)
    assert len(sampled) == 3
    assert len(set(p["id"] for p in sampled)) == 3


def test_buffer_sample_problems_only_easy(buffer_with_difficulties):
    buffer = Buffer(
        buffer_with_difficulties.dataset,
        BufferConfig(easy_fraction=1.0, hard_fraction=0.0, dataset_path=None),
    )
    buffer.metadata = buffer_with_difficulties.metadata.copy()
    sampled = buffer.sample_problems(2)
    assert len(sampled) == 2
    assert all(p["id"] in [0, 1] for p in sampled)


def test_buffer_sample_problems_only_hard(buffer_with_difficulties):
    buffer = Buffer(
        buffer_with_difficulties.dataset,
        BufferConfig(easy_fraction=0.0, hard_fraction=1.0, dataset_path=None),
    )
    buffer.metadata = buffer_with_difficulties.metadata.copy()
    sampled = buffer.sample_problems(2)
    assert len(sampled) == 2
    assert 4 in [p["id"] for p in sampled]
    assert buffer.problem_metrics["hard"] == 1
    assert buffer.problem_metrics["normal"] == 1


def test_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    assert len(buffer.sample_rollouts(10)) == 10
    assert all(m["difficulty"] == "normal" for m in buffer.metadata.values())


def test_buffer_update_zero_advantage_not_stored(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rollouts = make_rollouts(dataset, rewards=[0.0, 0.0, 0.0, 1.0, 1.0], advantages=[0.0] * 5)
    buffer.update(rollouts)
    assert len(buffer.sample_rollouts(10)) == 0
    assert buffer.metadata[0]["difficulty"] == "hard"
    assert buffer.metadata[4]["difficulty"] == "easy"
    assert buffer.rollout_metrics["hard"] == 6
    assert buffer.rollout_metrics["easy"] == 4


def test_buffer_update_partial_storage(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rollouts = make_rollouts(dataset, rewards=[0.2, 0.4, 0.5, 0.6, 0.8], advantages=[0.0, 1.0, 1.0, 1.0, 0.0])
    buffer.update(rollouts)
    assert len(buffer.sample_rollouts(10)) == 6
    assert buffer.metadata[0]["difficulty"] == "hard"
    assert buffer.metadata[4]["difficulty"] == "hard"
    assert all(buffer.metadata[i]["difficulty"] == "normal" for i in [1, 2, 3])


def test_buffer_update_difficulty(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    buffer.update(make_rollouts(dataset, rewards=[1.0, 0.0, 0.0, 0.0, 0.0], advantages=[0.0] * 5))
    assert buffer.metadata[0]["difficulty"] == "easy"
    assert buffer.metadata[1]["difficulty"] == "hard"
    buffer.update(make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5))
    assert buffer.metadata[2]["difficulty"] == "normal"
    assert len(buffer.rollout_buffer) == 10


def test_buffer_sample_rollouts_edge_cases(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    assert len(buffer.sample_rollouts(10)) == 0
    buffer.update(make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5))
    assert len(buffer.sample_rollouts(20)) == 10
    assert len(buffer.rollout_buffer) == 0


def test_buffer_sample_problems_fallback_to_normal(dataset):
    buffer = Buffer(dataset, BufferConfig())
    for pid in buffer.problem_ids:
        buffer.metadata[pid]["difficulty"] = "normal"
    buffer.sample_problems(5)
    assert buffer.problem_metrics["normal"] == 5
    assert buffer.problem_metrics["easy"] == 0
    assert buffer.problem_metrics["hard"] == 0


def test_buffer_get_metrics(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    buffer.sample_problems(3)
    buffer.update(make_rollouts(dataset, rewards=[1.0, 0.0, 0.5, 0.5, 0.5], advantages=[0.0, 0.0, 1.0, 1.0, 1.0]))
    metrics = buffer.get_metrics()
    assert all(k in metrics for k in ["problem_metrics/easy", "rollout_metrics/normal", "data_metrics/hard"])
    assert abs(sum(v for k, v in metrics.items() if k.startswith("rollout_metrics/")) - 1.0) < 0.001
    assert abs(sum(v for k, v in metrics.items() if k.startswith("data_metrics/")) - 1.0) < 0.001


def test_buffer_save_and_load(dataset, make_rollouts, tmp_path):
    buffer1 = Buffer(dataset, BufferConfig())
    buffer1.update(make_rollouts(dataset, rewards=[1.0, 0.0, 0.5, 0.5, 0.5], advantages=[0.0, 0.0, 1.0, 1.0, 1.0]))
    save_path = tmp_path / "buffer"
    buffer1.save(save_path)
    buffer2 = Buffer(dataset, BufferConfig(dataset_path=save_path))
    assert buffer2.metadata[0]["difficulty"] == "easy"
    assert buffer2.metadata[1]["difficulty"] == "hard"
    assert len(buffer2.rollout_buffer) == 6


def test_buffer_save_empty_rollouts(dataset, tmp_path):
    buffer = Buffer(dataset, BufferConfig())
    buffer.metadata[0]["difficulty"] = "easy"
    buffer.save(tmp_path / "buffer")
    assert (tmp_path / "metadata").exists()


def test_buffer_load_with_changed_dataset(dataset, make_rollouts, tmp_path):
    buffer1 = Buffer(dataset, BufferConfig())
    buffer1.update(make_rollouts(dataset, rewards=[1.0, 0.0, 0.5, 0.5, 0.5], advantages=[0.0, 0.0, 1.0, 1.0, 1.0]))
    save_path = tmp_path / "buffer"
    buffer1.save(save_path)
    
    buffer2 = Buffer(Dataset.from_list([{"problem": str(i)} for i in range(4)]), BufferConfig(dataset_path=save_path))
    assert len(buffer2.metadata) == 4
    assert buffer2.metadata[0]["difficulty"] == "easy"
    assert 4 not in buffer2.metadata
    
    buffer3 = Buffer(Dataset.from_list([{"problem": str(i)} for i in range(6)]), BufferConfig(dataset_path=save_path))
    assert len(buffer3.metadata) == 6
    assert buffer3.metadata[5]["difficulty"] == "normal"


def test_buffer_replay_buffer_basic(dataset, make_rollouts):
    """Test that rollouts removed from rollout_buffer are added to replay_buffer."""
    buffer = Buffer(dataset, BufferConfig(use_replay_buffer=True))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # Sample all rollouts - they should go to replay_buffer
    sampled = buffer.sample_rollouts(10)
    assert len(sampled) == 10
    assert len(buffer.rollout_buffer) == 0
    assert len(buffer.replay_buffer) == 10


def test_buffer_replay_buffer_fill_from_replay(dataset, make_rollouts):
    """Test that when rollout_buffer is empty, we sample from replay_buffer."""
    buffer = Buffer(dataset, BufferConfig(use_replay_buffer=True))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # Sample all rollouts to move them to replay_buffer
    buffer.sample_rollouts(10)
    assert len(buffer.rollout_buffer) == 0
    assert len(buffer.replay_buffer) == 10
    
    # Now sample again - should get from replay_buffer (from the end)
    sampled = buffer.sample_rollouts(5)
    assert len(sampled) == 5
    assert len(buffer.replay_buffer) == 10  # Still in replay buffer
    # Check that sampled items were moved to front of replay_buffer
    # The first 5 items should be the ones that were sampled
    assert len(buffer.replay_buffer[:5]) == 5


def test_buffer_replay_buffer_prioritization(dataset, make_rollouts):
    """Test that replay buffer samples from the end and moves items to front."""
    buffer = Buffer(dataset, BufferConfig(use_replay_buffer=True))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # Move all to replay buffer
    buffer.sample_rollouts(10)
    
    # Sample 3 rollouts (should be the latest 3: last 3 items in list)
    sampled1 = buffer.sample_rollouts(3)
    assert len(sampled1) == 3
    # Check that sampled items moved to front
    assert len(buffer.replay_buffer[:3]) == 3
    
    # Sample 3 more (should sample from the end again)
    sampled2 = buffer.sample_rollouts(3)
    assert len(sampled2) == 3
    
    # Check that replay buffer still has all items
    assert len(buffer.replay_buffer) == 10


def test_buffer_replay_buffer_partial_fill(dataset, make_rollouts):
    """Test that we fill from replay_buffer when rollout_buffer doesn't have enough."""
    buffer = Buffer(dataset, BufferConfig(use_replay_buffer=True))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # Sample 7 rollouts - all go to replay_buffer
    buffer.sample_rollouts(7)
    
    # Add 3 more rollouts to rollout_buffer
    more_rollouts = make_rollouts(dataset, rewards=[0.5] * 2, advantages=[1.0] * 2)
    buffer.update(more_rollouts)
    
    # Sample 10 - should get 7 from rollout_buffer (all available), 3 from replay_buffer
    # After first sample: rollout_buffer had 10, sampled 7, so 3 remaining
    # Then added 4 more, so rollout_buffer has 7 total
    # When sampling 10: take all 7 from rollout_buffer, then 3 from replay_buffer
    sampled = buffer.sample_rollouts(10)
    assert len(sampled) == 10
    assert len(buffer.rollout_buffer) == 0
    # replay_buffer: 3 sampled items moved to front, 4 remaining (7-3=4), then 7 added from rollout_buffer = 14
    assert len(buffer.replay_buffer) == 14


def test_buffer_replay_buffer_save_load(dataset, make_rollouts, tmp_path):
    """Test that replay buffer is saved and loaded correctly."""
    buffer1 = Buffer(dataset, BufferConfig(use_replay_buffer=True))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer1.update(rollouts)
    
    # Sample to populate replay buffer
    buffer1.sample_rollouts(10)
    buffer1.sample_rollouts(3)  # Use some from replay buffer
    
    save_path = tmp_path / "buffer"
    buffer1.save(save_path)
    
    buffer2 = Buffer(dataset, BufferConfig(dataset_path=save_path, use_replay_buffer=True))
    assert len(buffer2.replay_buffer) == 10


def test_buffer_take_all_rollouts_enabled_sufficient(dataset, make_rollouts):
    """Test take_all_rollouts when rollout_buffer has at least n rollouts."""
    buffer = Buffer(dataset, BufferConfig(take_all_rollouts=True))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # rollout_buffer has 10 rollouts, request 5
    assert len(buffer.rollout_buffer) == 10
    sampled = buffer.sample_rollouts(5)
    
    # Should take all 10 rollouts, not just 5
    assert len(sampled) == 10
    assert len(buffer.rollout_buffer) == 0


def test_buffer_take_all_rollouts_enabled_insufficient(dataset, make_rollouts):
    """Test take_all_rollouts when rollout_buffer has fewer than n rollouts."""
    buffer = Buffer(dataset, BufferConfig(take_all_rollouts=True, use_replay_buffer=True))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # Sample some to populate replay_buffer
    buffer.sample_rollouts(10)
    assert len(buffer.rollout_buffer) == 0
    assert len(buffer.replay_buffer) == 10
    
    # Add 3 more rollouts to rollout_buffer
    more_rollouts = make_rollouts(dataset, rewards=[0.5] * 2, advantages=[1.0] * 2)
    buffer.update(more_rollouts)
    assert len(buffer.rollout_buffer) == 4
    
    # Request 10, but rollout_buffer only has 4
    sampled = buffer.sample_rollouts(10)
    
    # Should take all 4 from rollout_buffer, then 6 from replay_buffer
    assert len(sampled) == 10
    assert len(buffer.rollout_buffer) == 0
    # After sampling: 6 sampled items moved to front, 4 remaining, then 4 added from rollout_buffer = 14
    assert len(buffer.replay_buffer) == 14


def test_buffer_take_all_rollouts_disabled(dataset, make_rollouts):
    """Test that take_all_rollouts=False uses normal behavior."""
    buffer = Buffer(dataset, BufferConfig(take_all_rollouts=False))
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # rollout_buffer has 10 rollouts, request 5
    assert len(buffer.rollout_buffer) == 10
    sampled = buffer.sample_rollouts(5)
    
    # Should take only 5 rollouts (latest)
    assert len(sampled) == 5
    assert len(buffer.rollout_buffer) == 5
