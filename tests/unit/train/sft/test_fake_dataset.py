import os

from prime_rl.trainer.sft.data import FakeDataset
from prime_rl.trainer.world import reset_world


def get_samples(rank: int, world_size: int, non_dp_size: int, num_samples: int = 2) -> list[list[int]]:
    """Helper to collect the first samples a rank of the given topology is handed."""
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world_size)
    )
    reset_world()
    dataiter = iter(FakeDataset(vocab_size=10000, seq_len=128, non_dp_size=non_dp_size))
    return [next(dataiter)["input_ids"] for _ in range(num_samples)]


def test_init_fake_dataset():
    fake_dataset = FakeDataset(vocab_size=10000, seq_len=128)
    assert fake_dataset is not None


def test_fake_dataset_state():
    dataset = FakeDataset(vocab_size=10000, seq_len=128)
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Iterate
    next(dataiter)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 2, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 3, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 4, "epoch": 0}


def test_fake_dataset_is_deterministic():
    """Samples are a pure function of their index, so every parallelism layout trains on the same data."""
    first, second = iter(FakeDataset(vocab_size=10000, seq_len=128)), iter(FakeDataset(vocab_size=10000, seq_len=128))

    assert [next(first) for _ in range(3)] == [next(second) for _ in range(3)]


def test_fake_dataset_shares_samples_within_non_dp_group():
    """Context parallel ranks split one sample along the sequence dim, so they get the same samples."""
    assert get_samples(rank=0, world_size=2, non_dp_size=2) == get_samples(rank=1, world_size=2, non_dp_size=2)


def test_fake_dataset_splits_samples_across_data_ranks():
    """Data parallel ranks get disjoint samples."""
    rank_0_samples = get_samples(rank=0, world_size=2, non_dp_size=1)
    rank_1_samples = get_samples(rank=1, world_size=2, non_dp_size=1)

    assert all(sample not in rank_1_samples for sample in rank_0_samples)
