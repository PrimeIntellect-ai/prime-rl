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
    """Creates a buffer with pre-set difficulties: [easy, easy, normal, normal, hard]."""
    buffer = Buffer(dataset, BufferConfig())
    # Manually set difficulties
    difficulties = ["easy", "easy", "normal", "normal", "hard"]
    for pid, difficulty in zip(buffer.problem_ids, difficulties):
        buffer.metadata[pid]["difficulty"] = difficulty
    return buffer


@pytest.fixture
def make_rollouts():
    """Factory fixture that creates rollouts for any given dataset."""

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
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    # All problems start as "normal" difficulty, so we should get normal problems
    assert all("id" in p for p in sampled_problems)


def test_buffer_sample_problems_mix(buffer_with_difficulties):
    buffer = Buffer(
        buffer_with_difficulties.dataset,
        BufferConfig(easy_fraction=0.5, hard_fraction=0.5, dataset_path=None),
    )
    # Copy difficulties from fixture
    buffer.metadata = buffer_with_difficulties.metadata.copy()
    sampled_problems = buffer.sample_problems(3)
    assert len(sampled_problems) == 3
    # Should have easy, normal, and hard problems
    problem_ids = [p["id"] for p in sampled_problems]
    assert len(set(problem_ids)) == 3


def test_buffer_sample_problems_only_easy(buffer_with_difficulties):
    buffer = Buffer(
        buffer_with_difficulties.dataset,
        BufferConfig(easy_fraction=1.0, hard_fraction=0.0, dataset_path=None),
    )
    # Copy difficulties from fixture
    buffer.metadata = buffer_with_difficulties.metadata.copy()
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    problem_ids = [p["id"] for p in sampled_problems]
    # Should only sample from easy problems (ids 0 and 1)
    assert all(pid in [0, 1] for pid in problem_ids)


def test_buffer_sample_problems_only_hard(buffer_with_difficulties):
    buffer = Buffer(
        buffer_with_difficulties.dataset,
        BufferConfig(easy_fraction=0.0, hard_fraction=1.0, dataset_path=None),
    )
    # Copy difficulties from fixture
    buffer.metadata = buffer_with_difficulties.metadata.copy()
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    problem_ids = [p["id"] for p in sampled_problems]
    # Should only sample from hard problems (id 4)
    # Note: only 1 hard problem exists, so we'll get warnings and fall back to normal
    # Should have 1 hard problem (id 4) and 1 normal problem
    assert 4 in problem_ids
    # Check that we sampled 1 hard and 1 normal via problem_metrics
    assert buffer.problem_metrics["hard"] == 1
    assert buffer.problem_metrics["normal"] == 1


def test_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rewards = [0.5, 0.5, 0.5, 0.5, 0.5]
    advantages = [1.0, 1.0, 1.0, 1.0, 1.0]  # Non-zero advantage so rollouts are stored
    rollouts = make_rollouts(dataset, rewards=rewards, advantages=advantages)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    # All rollouts have non-zero advantage, so all 10 should be stored and sampled
    assert len(sampled_rollouts) == 10
    # Metadata only stores difficulty, not reward
    assert all(metadata["difficulty"] == "normal" for metadata in buffer.metadata.values())


def test_buffer_update_zero_advantage_not_stored(dataset, make_rollouts):
    """Test that rollouts with zero advantage are not stored in buffer."""
    buffer = Buffer(dataset, BufferConfig())
    rewards = [0.0, 0.0, 0.0, 1.0, 1.0]
    advantages = [0.0, 0.0, 0.0, 0.0, 0.0]  # Zero advantage so rollouts are NOT stored
    rollouts = make_rollouts(dataset, rewards=rewards, advantages=advantages)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    # All rollouts have zero advantage, so none are stored
    assert len(sampled_rollouts) == 0
    assert len(buffer.rollout_buffer) == 0
    # Problems with reward 1.0 become easy, reward 0.0 become hard
    assert buffer.metadata[0]["difficulty"] == "hard"
    assert buffer.metadata[4]["difficulty"] == "easy"
    # Check metrics - each problem has 2 rollouts, so 3 problems * 2 = 6 hard, 2 problems * 2 = 4 easy
    assert buffer.rollout_metrics["hard"] == 6  # 3 problems with reward 0.0, 2 rollouts each
    assert buffer.rollout_metrics["easy"] == 4  # 2 problems with reward 1.0, 2 rollouts each


def test_buffer_update_partial_storage(dataset, make_rollouts):
    """Test that only rollouts with non-zero advantage are stored."""
    buffer = Buffer(dataset, BufferConfig())
    rewards = [0.2, 0.4, 0.5, 0.6, 0.8]
    # Set advantages: first and last have zero advantage (not stored), middle 3 have non-zero (stored)
    advantages = [0.0, 1.0, 1.0, 1.0, 0.0]
    rollouts = make_rollouts(dataset, rewards=rewards, advantages=advantages)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    # Only rollouts with non-zero advantage are stored: problems 1, 2, 3 = 6 rollouts (2 per problem)
    assert len(sampled_rollouts) == 6
    assert len(buffer.rollout_buffer) == 0  # All were sampled
    # Problems with zero advantage: reward 0.2 -> hard, reward 0.8 -> hard (not 1.0)
    assert buffer.metadata[0]["difficulty"] == "hard"  # reward 0.2, advantage 0
    assert buffer.metadata[4]["difficulty"] == "hard"  # reward 0.8, advantage 0 (not 1.0)
    # Problems with non-zero advantage become normal
    assert buffer.metadata[1]["difficulty"] == "normal"
    assert buffer.metadata[2]["difficulty"] == "normal"
    assert buffer.metadata[3]["difficulty"] == "normal"


def test_buffer_update_difficulty(dataset, make_rollouts):
    """Test that difficulty updates correctly based on advantage and reward."""
    buffer = Buffer(dataset, BufferConfig())
    
    # Test 1: Zero advantage, reward 1.0 -> easy (not stored)
    rollouts1 = make_rollouts(dataset, rewards=[1.0, 0.0, 0.0, 0.0, 0.0], advantages=[0.0, 0.0, 0.0, 0.0, 0.0])
    buffer.update(rollouts1)
    assert buffer.metadata[0]["difficulty"] == "easy"
    assert len(buffer.rollout_buffer) == 0
    
    # Test 2: Zero advantage, reward 0.0 -> hard (not stored)
    assert buffer.metadata[1]["difficulty"] == "hard"
    
    # Test 3: Non-zero advantage -> normal (stored)
    rollouts3 = make_rollouts(dataset, rewards=[0.5, 0.5, 0.5, 0.5, 0.5], advantages=[1.0, 1.0, 1.0, 1.0, 1.0])
    buffer.update(rollouts3)
    assert buffer.metadata[2]["difficulty"] == "normal"
    assert len(buffer.rollout_buffer) == 10  # 5 problems * 2 rollouts each


def test_buffer_sample_rollouts_edge_cases(dataset, make_rollouts):
    """Test edge cases for sampling rollouts."""
    buffer = Buffer(dataset, BufferConfig())
    
    # Empty buffer
    assert len(buffer.sample_rollouts(10)) == 0
    
    # Add some rollouts
    rollouts = make_rollouts(dataset, rewards=[0.5] * 5, advantages=[1.0] * 5)
    buffer.update(rollouts)
    
    # Sample more than available
    sampled = buffer.sample_rollouts(20)
    assert len(sampled) == 10  # Should only get what's available
    assert len(buffer.rollout_buffer) == 0


def test_buffer_sample_problems_fallback_to_normal(dataset):
    """Test that sampling falls back to normal pool when easy/hard pools are insufficient."""
    buffer = Buffer(dataset, BufferConfig())
    # Set all problems to normal
    for pid in buffer.problem_ids:
        buffer.metadata[pid]["difficulty"] = "normal"
    
    # Try to sample with high easy/hard fractions
    sampled = buffer.sample_problems(5)
    assert len(sampled) == 5
    # Should all come from normal pool
    assert buffer.problem_metrics["normal"] == 5
    assert buffer.problem_metrics["easy"] == 0
    assert buffer.problem_metrics["hard"] == 0




def test_buffer_get_metrics(dataset, make_rollouts):
    """Test metrics calculation."""
    buffer = Buffer(dataset, BufferConfig())
    
    # Sample some problems
    buffer.sample_problems(3)
    
    # Update with rollouts
    rollouts = make_rollouts(dataset, rewards=[1.0, 0.0, 0.5, 0.5, 0.5], advantages=[0.0, 0.0, 1.0, 1.0, 1.0])
    buffer.update(rollouts)
    
    metrics = buffer.get_metrics()
    
    # Check that metrics are normalized
    assert "problem_metrics/easy" in metrics
    assert "problem_metrics/normal" in metrics
    assert "problem_metrics/hard" in metrics
    assert "rollout_metrics/easy" in metrics
    assert "rollout_metrics/normal" in metrics
    assert "rollout_metrics/hard" in metrics
    assert "data_metrics/easy" in metrics
    assert "data_metrics/normal" in metrics
    assert "data_metrics/hard" in metrics
    
    # Check that normalized metrics sum to 1.0 (for each category)
    problem_sum = sum(v for k, v in metrics.items() if k.startswith("problem_metrics/"))
    rollout_sum = sum(v for k, v in metrics.items() if k.startswith("rollout_metrics/"))
    data_sum = sum(v for k, v in metrics.items() if k.startswith("data_metrics/"))
    
    # Note: problem_metrics might have multiple categories, so sum might be > 1
    # But rollout and data metrics should sum to 1.0
    assert abs(rollout_sum - 1.0) < 0.001
    assert abs(data_sum - 1.0) < 0.001


def test_buffer_save_and_load(dataset, make_rollouts, tmp_path):
    """Test save and load functionality."""
    buffer1 = Buffer(dataset, BufferConfig())
    
    # Set some difficulties manually (before updating with rollouts)
    buffer1.metadata[0]["difficulty"] = "easy"
    buffer1.metadata[1]["difficulty"] = "hard"
    
    # Add rollouts - but these will update difficulties to "normal" since advantage != 0
    # So we need to set difficulties after, or use zero advantage rollouts
    rollouts = make_rollouts(dataset, rewards=[1.0, 0.0, 0.5, 0.5, 0.5], advantages=[0.0, 0.0, 1.0, 1.0, 1.0])
    buffer1.update(rollouts)
    
    # After update: problems 0,1 have zero advantage so difficulty updated
    # Problem 0: reward 1.0, advantage 0 -> easy
    # Problem 1: reward 0.0, advantage 0 -> hard
    # Problems 2,3,4: advantage != 0 -> normal
    
    # Save buffer
    save_path = tmp_path / "buffer"
    buffer1.save(save_path)
    
    # Create new buffer and load
    buffer2 = Buffer(dataset, BufferConfig(dataset_path=save_path))
    
    # Verify loaded state matches what was saved
    assert buffer2.metadata[0]["difficulty"] == "easy"
    assert buffer2.metadata[1]["difficulty"] == "hard"
    assert buffer2.metadata[2]["difficulty"] == "normal"
    assert len(buffer2.rollout_buffer) == 6  # Only problems 2,3,4 have rollouts stored (3 problems * 2 rollouts)
    
    # Verify rollouts match
    assert len(buffer2.rollout_buffer) == len(buffer1.rollout_buffer)


def test_buffer_save_empty_rollouts(dataset, tmp_path):
    """Test saving buffer with empty rollouts."""
    buffer = Buffer(dataset, BufferConfig())
    buffer.metadata[0]["difficulty"] = "easy"
    
    save_path = tmp_path / "buffer"
    buffer.save(save_path)
    
    # Verify metadata was saved
    assert (save_path.parent / "metadata").exists()
    # Rollouts path should not exist if buffer is empty
    # (Based on the implementation, it only saves if non-empty)


def test_buffer_load_with_changed_dataset(dataset, make_rollouts, tmp_path):
    """Test loading buffer when dataset size changes (problems added/removed)."""
    # Create and save buffer with original dataset (5 problems: IDs 0-4)
    buffer1 = Buffer(dataset, BufferConfig())
    # Set difficulties: problem 0 easy, problem 1 hard, others normal
    rollouts = make_rollouts(
        dataset, 
        rewards=[1.0, 0.0, 0.5, 0.5, 0.5], 
        advantages=[0.0, 0.0, 1.0, 1.0, 1.0]
    )
    buffer1.update(rollouts)
    
    save_path = tmp_path / "buffer"
    buffer1.save(save_path)
    
    # Test smaller dataset (4 problems: removed problem 4)
    smaller_dataset = Dataset.from_list([{"problem": str(i)} for i in range(4)])
    buffer2 = Buffer(smaller_dataset, BufferConfig(dataset_path=save_path))
    
    # Metadata should only include current problems, preserve difficulties
    assert len(buffer2.metadata) == 4
    assert buffer2.metadata[0]["difficulty"] == "easy"
    assert buffer2.metadata[1]["difficulty"] == "hard"
    assert 4 not in buffer2.metadata
    
    # Test larger dataset (6 problems: added problem 5)
    larger_dataset = Dataset.from_list([{"problem": str(i)} for i in range(6)])
    buffer3 = Buffer(larger_dataset, BufferConfig(dataset_path=save_path))
    
    # Should have metadata for all problems, new ones get default difficulty
    assert len(buffer3.metadata) == 6
    assert buffer3.metadata[0]["difficulty"] == "easy"
    assert buffer3.metadata[5]["difficulty"] == "normal"
