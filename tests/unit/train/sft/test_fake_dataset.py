from prime_rl.trainer.sft.data import FakeDataset


def test_init_fake_dataset():
    fake_dataset = FakeDataset(vocab_size=10000, seq_len=128)
    assert fake_dataset is not None


def test_fake_dataset_state():
    dataset = FakeDataset(vocab_size=10000, seq_len=128)
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": -1, "epoch": 0}

    # Iterate
    next(dataiter)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 2, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 3, "epoch": 0}
