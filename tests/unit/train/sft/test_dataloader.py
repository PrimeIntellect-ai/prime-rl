import gc
import os
from collections import defaultdict

import pytest
import torch
from datasets import Dataset

from prime_rl.configs.sft import FakeDataConfig, SFTDataConfig
from prime_rl.trainer.sft.data import FakeDataset, SFTDataset, setup_dataloader
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

    # Iterate over samples
    for step in range(4):
        micro_batch = next(dataiter)
        assert micro_batch.input_ids.unique().item() == step
        assert micro_batch.step == step + 1
        assert micro_batch.epoch == 0


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

    data_rank = rank // non_dp_size
    data_world_size = 2 // non_dp_size
    for index in range(6):
        expected_sample = index * data_world_size + data_rank
        micro_batch = next(dataiter)
        assert micro_batch.input_ids.unique().item() == expected_sample
        assert micro_batch.step == expected_sample + 1


def test_fake_dataset_single_rank_resume():
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    # First 2 samples
    for step in range(2):
        micro_batch = next(dataiter)
        assert micro_batch.input_ids.shape == (1, 128)
        assert micro_batch.input_ids.unique().item() == step
        assert micro_batch.step == step + 1

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second two samples
    for step in range(2, 4):
        micro_batch = next(dataiter)
        assert micro_batch.input_ids.shape == (1, 128)
        assert micro_batch.input_ids.unique().item() == step
        assert micro_batch.step == step + 1


def test_fake_dataset_single_rank_state_with_packing():
    config = FakeDataConfig(length="variable", input_ids="increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    step = 0
    for _ in range(8):
        micro_batch = next(dataiter)
        step += len(micro_batch.samples)
        assert micro_batch.input_ids.shape == (1, 128)
        assert micro_batch.seq_lens.sum() == micro_batch.input_ids.shape[1]
        assert micro_batch.step == step
        assert micro_batch.num_padding_tokens == (~micro_batch.loss_mask).sum().item()

    state_dict = dataloader.state_dict()
    rng_state = torch.random.get_rng_state()
    expected_batch = next(dataiter)

    _, resumed_dataloader = setup_fake_dataloader(config)
    resumed_dataloader.load_state_dict(state_dict)
    resumed_dataiter = iter(resumed_dataloader)
    torch.random.set_rng_state(rng_state)
    resumed_batch = next(resumed_dataiter)

    for key in ("input_ids", "position_ids", "target_ids", "loss_mask", "seq_lens"):
        torch.testing.assert_close(getattr(resumed_batch, key), getattr(expected_batch, key))


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize(
    ("world_size", "non_dp_size"),
    [
        pytest.param(1, 1, id="1dp"),
        pytest.param(2, 1, id="2dp"),
        pytest.param(2, 2, id="1dp-2cp"),
        pytest.param(4, 2, id="2dp-2cp"),
    ],
)
def test_dataloader_shards_across_ranks_and_workers(
    num_workers: int, world_size: int, non_dp_size: int, dummy_renderer
):
    rounds_before_resume = 3
    rounds_after_resume = 1
    data_world_size = world_size // non_dp_size
    num_examples = 2 * data_world_size * num_workers
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
                non_dp_size=non_dp_size,
            )
            return setup_dataloader(dataset, config)

        dataloader = setup_epoch_dataloader()
        dataiter = iter(dataloader)
        epochs = set()
        num_samples_by_source = defaultdict(int)
        num_tokens_by_source = defaultdict(int)

        def next_sample(dataiter, index: int) -> int:
            micro_batch = next(dataiter)
            # After the causal shift, the first target is DummyRenderer's first character, offset by its two special tokens.
            sample = micro_batch.target_ids[0, 0].item() - ord("0") - 2
            worker_id = index % num_workers
            worker_round = index // num_workers
            data_rank = rank // non_dp_size
            position = worker_round * data_world_size * num_workers + data_rank * num_workers + worker_id
            for source, num_samples in micro_batch.num_samples_by_source.items():
                num_samples_by_source[source] += num_samples
            for source, num_tokens in micro_batch.num_tokens_by_source.items():
                num_tokens_by_source[source] += num_tokens
            assert sample == position % num_examples
            assert micro_batch.step == position + 1
            assert micro_batch.epoch == position // num_examples
            assert dict(num_samples_by_source) == {"fake": index + 1}
            assert dict(num_tokens_by_source) == {"fake": (index + 1) * config.seq_len}
            epochs.add(micro_batch.epoch)
            return sample

        samples = [next_sample(dataiter, index) for index in range(samples_before_resume)]

        state_dict = dataloader.state_dict()
        del dataiter, dataloader
        gc.collect()

        dataloader = setup_epoch_dataloader()
        dataloader.load_state_dict(state_dict)
        dataiter = iter(dataloader)
        samples.extend(next_sample(dataiter, index) for index in range(samples_before_resume, samples_per_rank))

        samples_by_rank.append(samples)
        assert epochs == {0, 1}
        del dataiter, dataloader
        gc.collect()

    for data_rank in range(data_world_size):
        cp_samples = samples_by_rank[data_rank * non_dp_size : (data_rank + 1) * non_dp_size]
        assert all(samples == cp_samples[0] for samples in cp_samples)

    unique_samples = [samples_by_rank[data_rank * non_dp_size] for data_rank in range(data_world_size)]
    expected = sorted(list(range(num_examples)) * 2)
    assert sorted(sample for samples in unique_samples for sample in samples) == expected


def test_batch_metadata_with_uneven_workers():
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

    seen_steps = []
    for _ in range(12):
        micro_batch = next(dataiter)
        assert micro_batch.num_samples_by_source == {"fake": len(micro_batch.samples)}
        assert micro_batch.num_padding_tokens == (~micro_batch.loss_mask).sum().item()
        seen_steps.extend(sample.step for sample in micro_batch.samples)

    del dataiter, dataloader
    gc.collect()
    assert len(seen_steps) == len(set(seen_steps))
