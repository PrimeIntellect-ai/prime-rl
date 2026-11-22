import pytest

from prime_rl.trainer.sft.data import FakeSource, SFTDataset


def position(state: dict) -> dict:
    return {key: state[key] for key in ("step", "epoch", "draws")}


def fake_dataset(**kwargs) -> SFTDataset:
    return SFTDataset([FakeSource(vocab_size=10000, seq_len=128, **kwargs)])


def test_fake_dataset_state():
    dataset = fake_dataset()
    dataiter = iter(dataset)

    assert position(dataset.state_dict()) == {"step": 0, "epoch": 0, "draws": {"fake": 0}}
    for step in range(1, 5):
        next(dataiter)
        assert position(dataset.state_dict()) == {"step": step, "epoch": 0, "draws": {"fake": step}}


@pytest.mark.parametrize("length", ["fixed", "variable"])
@pytest.mark.parametrize("input_ids", ["increasing", "random"])
def test_fake_dataset_sample_lengths(length: str, input_ids: str):
    dataiter = iter(fake_dataset(length=length, input_ids=input_ids))
    for _ in range(20):
        sample = next(dataiter)
        seq_len = len(sample["input_ids"])
        assert len(sample["target_ids"]) == seq_len
        assert len(sample["position_ids"]) == seq_len
        assert len(sample["loss_mask"]) == seq_len
        assert sum(sample["seq_lens"]) == seq_len


@pytest.mark.parametrize("length", ["fixed", "variable"])
@pytest.mark.parametrize("data_world_size", [1, 2, 3])
def test_fake_dataset_random_resume(length: str, data_world_size: int):
    """A resumed run draws the same samples on every data rank."""

    def make(data_rank: int) -> SFTDataset:
        return SFTDataset(
            [FakeSource(vocab_size=10000, seq_len=128, length=length, input_ids="random")],
            dp_rank=data_rank,
            dp_size=data_world_size,
        )

    for data_rank in range(data_world_size):
        dataset = make(data_rank)
        dataiter = iter(dataset)
        for _ in range(3):
            next(dataiter)

        state_dict = dataset.state_dict()
        expected = [next(dataiter)["input_ids"].tolist() for _ in range(2)]

        resumed = make(data_rank)
        resumed.load_state_dict(state_dict)
        resumed_dataiter = iter(resumed)
        assert [next(resumed_dataiter)["input_ids"].tolist() for _ in range(2)] == expected
        assert resumed.state_dict() == dataset.state_dict()


def test_data_fake_replaces_the_sources_and_needs_no_renderer():
    from types import SimpleNamespace

    from prime_rl.configs.sft import SFTDataConfig
    from prime_rl.trainer.sft.data import setup_sft_dataset

    config = SFTDataConfig(fake={"input_ids": "increasing"}, seq_len=16)
    dataset = setup_sft_dataset(SimpleNamespace(vocab_size=32), config, dp_rank=0, dp_size=1)
    sample = next(iter(dataset))
    assert sample["input_ids"].tolist() == [0] * 16
    assert dict(dataset.num_samples) == {"fake": 1}
