import gc
import os

import pytest
import torch
from datasets import Dataset

from prime_rl.configs.sft import FakeDataConfig, SFTDataConfig
from prime_rl.trainer.sft.data import FakeDataset, SFTDataset, get_dataset_progress, get_dataset_state, setup_dataloader
from prime_rl.trainer.world import reset_world


def setup_fake_dataloader(config: FakeDataConfig, non_dp_size: int = 1):
    dataset = FakeDataset(
        vocab_size=32,
        seq_len=config.seq_len,
        length=config.length,
        input_ids=config.input_ids,
        seed=config.seed,
        non_dp_size=non_dp_size,
    )
    return dataset, setup_dataloader(dataset, config)


def test_fake_dataset_single_rank_state():
    # Setup stateful dataloader
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    # Initial state
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 0, "epoch": 0}}

    # Iterate over samples
    micro_batch = next(dataiter)
    print(micro_batch)
    assert micro_batch["input_ids"].unique().item() == 0
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 1, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 2, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 3, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 4, "epoch": 0}}


@pytest.mark.parametrize("rank", [0, 1], ids=["rank0", "rank1"])
@pytest.mark.parametrize("non_dp_size", [1, 2], ids=["dp", "cp"])
def test_fake_dataset_multi_rank_state(rank: int, non_dp_size: int):
    # Setup world
    reset_world()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(2)

    # Setup stateful dataloader
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config, non_dp_size)
    dataiter = iter(dataloader)

    # Initial state
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 0, "epoch": 0}}

    data_rank = rank // non_dp_size
    data_world_size = 2 // non_dp_size
    for index in range(6):
        expected_sample = index * data_world_size + data_rank
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].unique().item() == expected_sample
        assert get_dataset_state(dataloader) == {"worker_0": {"step": expected_sample + 1, "epoch": 0}}


def test_fake_dataset_single_rank_resume():
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    # First 2 samples
    for step in range(2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert get_dataset_state(dataloader) == {"worker_0": {"step": step + 1, "epoch": 0}}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second two samples
    for step in range(2, 4):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert get_dataset_state(dataloader) == {"worker_0": {"step": step + 1, "epoch": 0}}


def test_fake_dataset_single_rank_state_with_packing():
    config = FakeDataConfig(length="variable", input_ids="increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    step = 0
    for _ in range(8):
        micro_batch = next(dataiter)
        num_packed_examples = len(micro_batch["input_ids"][micro_batch["loss_mask"]].unique())
        step += num_packed_examples
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["seq_lens"].sum() == micro_batch["input_ids"].shape[1]
        worker_state = dataloader.state_dict()["_snapshot"]["_worker_snapshots"]["worker_0"]["dataset_state"]
        pending_sample = worker_state.get("pending_sample")
        expected_dataset_step = step + (pending_sample is not None)
        assert get_dataset_state(dataloader) == {"worker_0": {"step": expected_dataset_step, "epoch": 0}}
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
@pytest.mark.parametrize("world_size", [1, 2])
def test_dataloader_shards_across_ranks_and_workers(num_workers: int, world_size: int, dummy_renderer):
    rounds_before_resume = 3
    rounds_after_resume = 1
    num_examples = 2 * world_size * num_workers
    samples_before_resume = rounds_before_resume * num_workers
    samples_per_rank = (rounds_before_resume + rounds_after_resume) * num_workers
    samples_by_rank = []

    for rank in range(world_size):
        reset_world()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

        config = SFTDataConfig(
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
                        "__split": "fake",
                    }
                    for index in range(num_examples)
                ]
            )
            dataset = SFTDataset(
                raw_dataset,
                dummy_renderer,
                shuffle=False,
                seq_len=config.seq_len,
            )
            return setup_dataloader(dataset, config)

        dataloader = setup_epoch_dataloader()
        dataiter = iter(dataloader)
        epochs = set()

        def next_sample(dataiter, dataloader, index: int) -> int:
            micro_batch = next(dataiter)
            sample = micro_batch["target_ids"][0, 0].item() - ord("0") - 2
            worker_id = index % num_workers
            worker_round = index // num_workers
            position = worker_round * world_size * num_workers + rank * num_workers + worker_id
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
    config = FakeDataConfig(
        batch_size=1,
        micro_batch_size=1,
        seq_len=32,
        num_workers=2,
        length="variable",
        input_ids="increasing",
    )
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    positions = []
    for _ in range(12):
        next(dataiter)
        positions.append(get_dataset_progress(dataloader)["step"])

    del dataiter, dataloader
    gc.collect()
    assert positions == sorted(positions)
