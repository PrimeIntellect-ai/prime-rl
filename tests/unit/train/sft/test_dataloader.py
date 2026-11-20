import gc

import pytest
import torch
from datasets import Dataset

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.trainer.data import get_dataset_progress, get_dataset_state, setup_dataloader
from prime_rl.trainer.sft.data import FakeSource, HFSource, SFTDataset


def fake_config(length: str = "fixed", input_ids: str = "increasing", **kwargs) -> SFTDataConfig:
    return SFTDataConfig(fake={"length": length, "input_ids": input_ids}, **kwargs)


def setup_fake_dataloader(config: SFTDataConfig, dp_rank: int = 0, dp_size: int = 1):
    fake = FakeSource(vocab_size=32, seq_len=config.seq_len, length=config.fake.length, input_ids=config.fake.input_ids)
    dataset = SFTDataset([fake], dp_rank=dp_rank, dp_size=dp_size)
    return dataset, setup_dataloader(dataset, config, cp=1)


def test_fake_dataset_single_rank_state():
    # Setup stateful dataloader
    config = fake_config("fixed", "increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    # Initial state
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 0, "epoch": 0, "draws": {"fake": 0}}}

    # Iterate over samples
    micro_batch = next(dataiter)
    print(micro_batch)
    assert micro_batch["input_ids"].unique().item() == 0
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 1, "epoch": 0, "draws": {"fake": 1}}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 2, "epoch": 0, "draws": {"fake": 2}}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 3, "epoch": 0, "draws": {"fake": 3}}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 4, "epoch": 0, "draws": {"fake": 4}}}


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_fake_dataset_multi_rank_state(dp_rank: int):
    # Setup stateful dataloader
    config = fake_config("fixed", "increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config, dp_rank, dp_size=2)
    dataiter = iter(dataloader)

    # Initial state
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 0, "epoch": 0, "draws": {"fake": 0}}}

    for index in range(6):
        expected_sample = index * 2 + dp_rank
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].unique().item() == expected_sample
        assert get_dataset_state(dataloader) == {
            "worker_0": {"step": expected_sample + 1, "epoch": 0, "draws": {"fake": expected_sample + 1}}
        }


def test_fake_dataset_single_rank_resume():
    config = fake_config("fixed", "increasing", batch_size=1)
    dataset, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    # First 2 samples
    for step in range(2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert get_dataset_state(dataloader) == {
            "worker_0": {"step": step + 1, "epoch": 0, "draws": {"fake": step + 1}}
        }

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, config, cp=1)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second two samples
    for step in range(2, 4):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert get_dataset_state(dataloader) == {
            "worker_0": {"step": step + 1, "epoch": 0, "draws": {"fake": step + 1}}
        }


def test_fake_dataset_single_rank_state_with_packing():
    config = fake_config("variable", "increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    step = 0
    for _ in range(8):
        micro_batch = next(dataiter)
        num_packed_examples = len(micro_batch["input_ids"][micro_batch["loss_mask"]].unique())
        step += num_packed_examples
        assert micro_batch["input_ids"].shape[0] == 1
        assert micro_batch["input_ids"].shape[1] <= 128
        assert micro_batch["seq_lens"].sum() == micro_batch["input_ids"].shape[1]
        worker_state = dataloader.state_dict()["_snapshot"]["_worker_snapshots"]["worker_0"]["dataset_state"]
        pending_sample = worker_state.get("pending_sample")
        expected_dataset_step = step + (pending_sample is not None)
        assert get_dataset_state(dataloader) == {
            "worker_0": {"step": expected_dataset_step, "epoch": 0, "draws": {"fake": expected_dataset_step}}
        }
        if pending_sample is not None:
            assert pending_sample["input_ids"][0] == step

    state_dict = dataloader.state_dict()
    rng_state = torch.random.get_rng_state()
    expected_batch = next(dataiter)

    _, resumed_dataloader = setup_fake_dataloader(config)
    resumed_dataloader.load_state_dict(state_dict)
    resumed_dataiter = iter(resumed_dataloader)
    torch.random.set_rng_state(rng_state)
    resumed_batch = next(resumed_dataiter)

    for key in ("input_ids", "position_ids", "target_ids", "loss_mask", "seq_lens"):
        torch.testing.assert_close(resumed_batch[key], expected_batch[key])


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_dataloader_shards_across_ranks_and_workers(num_workers: int, dp_size: int, dummy_renderer):
    rounds_before_resume = 3
    rounds_after_resume = 1
    num_examples = 2 * dp_size * num_workers
    samples_before_resume = rounds_before_resume * num_workers
    samples_per_rank = (rounds_before_resume + rounds_after_resume) * num_workers
    samples_by_rank = []

    for dp_rank in range(dp_size):
        config = SFTDataConfig(
            source=[{"name": "fake", "dataset": "unused"}],
            batch_size=1,
            micro_batch_size=1,
            seq_len=7,
            num_workers=num_workers,
            shuffle=False,
        )

        def setup_epoch_dataloader():
            raw_dataset = Dataset.from_list(
                [
                    {
                        "messages": [{"role": "assistant", "content": str(index) * 6}],
                    }
                    for index in range(num_examples)
                ]
            )
            source = HFSource(raw_dataset, lambda _: dummy_renderer, name="fake", shuffle=False, seq_len=config.seq_len)
            dataset = SFTDataset([source], dp_rank=dp_rank, dp_size=dp_size)
            return setup_dataloader(dataset, config, cp=1)

        dataloader = setup_epoch_dataloader()
        dataiter = iter(dataloader)
        epochs = set()

        def next_sample(dataiter, dataloader, index: int) -> int:
            micro_batch = next(dataiter)
            # After the causal shift, the first target is DummyRenderer's first character, offset by its two special tokens.
            sample = micro_batch["target_ids"][0, 0].item() - ord("0") - 2
            worker_id = index % num_workers
            worker_round = index // num_workers
            position = worker_round * dp_size * num_workers + dp_rank * num_workers + worker_id
            progress = get_dataset_progress(dataloader)
            assert sample == position % num_examples
            assert progress["step"] == position + 1
            assert progress["epoch"] == position // num_examples
            assert progress["num_samples"] == {"fake": index + 1}
            assert progress["num_tokens"] == {"fake": (index + 1) * config.seq_len}
            epochs.add(progress["epoch"])
            return sample

        samples = [next_sample(dataiter, dataloader, index) for index in range(samples_before_resume)]

        state_dict = dataloader.state_dict()
        del dataiter, dataloader
        gc.collect()

        dataloader = setup_epoch_dataloader()
        dataloader.load_state_dict(state_dict)
        dataiter = iter(dataloader)
        samples.extend(
            next_sample(dataiter, dataloader, index) for index in range(samples_before_resume, samples_per_rank)
        )

        samples_by_rank.append(samples)
        assert epochs == {0, 1}
        del dataiter, dataloader
        gc.collect()

    expected = sorted(list(range(num_examples)) * 2)
    assert sorted(sample for samples in samples_by_rank for sample in samples) == expected


def test_dataloader_progress_is_monotonic_with_uneven_workers():
    config = fake_config("variable", "increasing", batch_size=1, micro_batch_size=1, seq_len=32, num_workers=2)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    positions = []
    for _ in range(12):
        next(dataiter)
        positions.append(get_dataset_progress(dataloader)["step"])

    del dataiter, dataloader
    gc.collect()
    assert positions == sorted(positions)
