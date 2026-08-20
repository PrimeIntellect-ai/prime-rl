import pytest

from prime_rl.trainer.sft.data import FakeDataset


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


@pytest.mark.parametrize("length", ["fixed", "variable"])
@pytest.mark.parametrize("data_world_size", [1, 2, 3])
def test_fake_dataset_random_resume(length: str, data_world_size: int):
    # Resuming mid run must replay the same per perank PR foir every data rank

    def make(data_rank: int) -> FakeDataset:
        dataset = FakeDataset(vocab_size=10000, seq_len=128, length=length, input_ids="random")
        dataset.data_rank, dataset.data_world_size = data_rank, data_world_size
        return dataset

    for data_rank in range(data_world_size):
        dataset = make(data_rank)
        dataiter = iter(dataset)
        for _ in range(3):
            next(dataiter)

        state_dict = dataset.state_dict()
        expected = [next(dataiter)["input_ids"] for _ in range(2)]

        resumed = make(data_rank)
        resumed.load_state_dict(state_dict)
        resumed_dataiter = iter(resumed)
        assert [next(resumed_dataiter)["input_ids"] for _ in range(2)] == expected
        assert resumed.state_dict() == dataset.state_dict()
