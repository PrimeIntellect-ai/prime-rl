import gc
import os

import pytest
import torch

from prime_rl.configs.sft import FakeDataConfig
from prime_rl.trainer.sft.data import FakeDataset, get_dataset_progress, get_dataset_state, setup_dataloader
from prime_rl.trainer.world import reset_world


def setup_fake_dataloader(config: FakeDataConfig):
    dataset = FakeDataset(
        vocab_size=32,
        seq_len=config.seq_len,
        length=config.length,
        input_ids=config.input_ids,
        seed=config.seed,
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
def test_fake_dataset_multi_rank_state(rank: int):
    # Setup world
    reset_world()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(2)

    # Setup stateful dataloader
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    # Initial state
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 0, "epoch": 0}}

    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 0 + rank
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 1 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2 + rank
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 3 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 4 + rank
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 5 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 6 + rank
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 7 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 8 + rank
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 9 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 10 + rank
    assert get_dataset_state(dataloader) == {"worker_0": {"step": 11 + rank, "epoch": 0}}


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
def test_dataloader_shards_across_ranks_and_workers(num_workers: int, world_size: int):
    rounds_before_resume = 3
    rounds_after_resume = 1
    samples_before_resume = rounds_before_resume * num_workers
    samples_per_rank = (rounds_before_resume + rounds_after_resume) * num_workers
    samples_by_rank = []

    for rank in range(world_size):
        reset_world()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

        config = FakeDataConfig(
            batch_size=1,
            micro_batch_size=1,
            seq_len=8,
            num_workers=num_workers,
            length="fixed",
            input_ids="increasing",
        )
        _, dataloader = setup_fake_dataloader(config)
        dataiter = iter(dataloader)

        def next_sample(dataiter, dataloader, index: int) -> int:
            micro_batch = next(dataiter)
            sample = micro_batch["input_ids"].unique().item()
            progress = get_dataset_progress(dataloader)
            assert progress["step"] == sample + 1
            assert progress["num_samples"] == {"fake": index + 1}
            assert progress["num_tokens"] == {"fake": (index + 1) * (config.seq_len + 1)}
            return sample

        samples = [next_sample(dataiter, dataloader, index) for index in range(samples_before_resume)]

        state_dict = dataloader.state_dict()
        del dataiter, dataloader
        gc.collect()

        _, dataloader = setup_fake_dataloader(config)
        dataloader.load_state_dict(state_dict)
        dataiter = iter(dataloader)
        samples.extend(
            next_sample(dataiter, dataloader, index) for index in range(samples_before_resume, samples_per_rank)
        )

        samples_by_rank.append(samples)
        del dataiter, dataloader
        gc.collect()

    expected = list(range((rounds_before_resume + rounds_after_resume) * world_size * num_workers))
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
