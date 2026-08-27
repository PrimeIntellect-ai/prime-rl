import gc
import os

import pytest

from prime_rl.configs.sft import FakeDataConfig
from prime_rl.trainer.sft.data import FakeDataset, get_dataset_progress, setup_dataloader
from prime_rl.trainer.world import reset_world


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
        dataset = FakeDataset(
            vocab_size=32,
            seq_len=config.seq_len,
            length=config.length,
            input_ids=config.input_ids,
            seed=config.seed,
        )
        dataloader = setup_dataloader(dataset, config)
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

        dataset = FakeDataset(
            vocab_size=32,
            seq_len=config.seq_len,
            length=config.length,
            input_ids=config.input_ids,
            seed=config.seed,
        )
        dataloader = setup_dataloader(dataset, config)
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
    dataset = FakeDataset(
        vocab_size=32,
        seq_len=config.seq_len,
        length=config.length,
        input_ids=config.input_ids,
        seed=config.seed,
    )
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    positions = []
    for _ in range(12):
        next(dataiter)
        positions.append(get_dataset_progress(dataloader)["step"])

    del dataiter, dataloader
    gc.collect()
    assert positions == sorted(positions)
