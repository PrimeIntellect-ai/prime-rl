import gc

import pytest
import torch
from datasets import Dataset

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.trainer.data import get_dataset_progress, get_dataset_state, setup_dataloader, with_workers
from prime_rl.trainer.sft.data import FakeSource, HFSource, SFTDataset


def fake_config(length: str = "fixed", input_ids: str = "increasing", **kwargs) -> SFTDataConfig:
    return SFTDataConfig(fake={"length": length, "input_ids": input_ids}, **kwargs)


def setup_fake_dataloader(config: SFTDataConfig, dp_rank: int = 0, dp_size: int = 1):
    fake = FakeSource(vocab_size=32, seq_len=config.seq_len, length=config.fake.length, input_ids=config.fake.input_ids)
    dataset = SFTDataset([fake], dp_rank=dp_rank, dp_size=dp_size)
    return dataset, setup_dataloader(dataset, config, cp=1)


def positions(dataloader) -> dict:
    """Every worker's position in the draws."""
    states = get_dataset_state(dataloader).items()
    return {worker: {key: state[key] for key in ("step", "epoch", "draws")} for worker, state in states}


def test_fake_dataset_single_rank_state():
    # Setup stateful dataloader
    config = fake_config("fixed", "increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config)
    dataiter = iter(dataloader)

    # Initial state
    assert positions(dataloader) == {"worker_0": {"step": 0, "epoch": 0, "draws": {"fake": 0}}}

    # Iterate over samples
    micro_batch = next(dataiter)
    print(micro_batch)
    assert micro_batch["input_ids"].unique().item() == 0
    assert positions(dataloader) == {"worker_0": {"step": 1, "epoch": 0, "draws": {"fake": 1}}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1
    assert positions(dataloader) == {"worker_0": {"step": 2, "epoch": 0, "draws": {"fake": 2}}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2
    assert positions(dataloader) == {"worker_0": {"step": 3, "epoch": 0, "draws": {"fake": 3}}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3
    assert positions(dataloader) == {"worker_0": {"step": 4, "epoch": 0, "draws": {"fake": 4}}}


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_fake_dataset_multi_rank_state(dp_rank: int):
    # Setup stateful dataloader
    config = fake_config("fixed", "increasing", batch_size=1)
    _, dataloader = setup_fake_dataloader(config, dp_rank, dp_size=2)
    dataiter = iter(dataloader)

    # Initial state
    assert positions(dataloader) == {"worker_0": {"step": 0, "epoch": 0, "draws": {"fake": 0}}}

    for index in range(6):
        expected_sample = index * 2 + dp_rank
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].unique().item() == expected_sample
        assert positions(dataloader) == {
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
        assert positions(dataloader) == {"worker_0": {"step": step + 1, "epoch": 0, "draws": {"fake": step + 1}}}

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
        assert positions(dataloader) == {"worker_0": {"step": step + 1, "epoch": 0, "draws": {"fake": step + 1}}}


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
        assert positions(dataloader) == {
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


@pytest.mark.parametrize(
    ("weights_after", "gpus_before", "gpus_after", "cp_after", "workers_after", "old_state"),
    [
        ({"b": 3, "a": 1, "c": 1}, 2, 2, 1, 2, False),  # another mixture: d removed, c added
        ({"a": 1, "b": 1, "d": 1}, 2, 2, 1, 3, False),  # more workers
        ({"a": 1, "b": 1, "d": 1}, 2, 1, 1, 4, False),  # fewer GPUs, more workers
        ({"a": 1, "b": 1, "d": 1}, 4, 2, 1, 2, False),  # fewer GPUs: 2 ranks read the 4 saved files
        ({"a": 1, "b": 1, "d": 1}, 2, 4, 2, 2, False),  # context parallel: 4 GPUs, still 2 data ranks
        ({"a": 1, "b": 1, "d": 1}, 2, 2, 1, 2, True),  # a state from before the layout was recorded
    ],
)
def test_resume_starts_every_worker_at_the_same_draw(
    tmp_path, dummy_renderer, weights_after, gpus_before, gpus_after, cp_after, workers_after, old_state
):
    """A run on GPUs x 2 workers, resumed with other weights, workers, GPUs or context parallelism, or from a state
    without the recorded layout: the ranks decide together that the data changed, and every worker starts at the
    furthest draw a saved worker had reached, with the skipped draws counted. The GPUs of a context-parallel group
    read the same samples, no row is read twice, a removed source no more, and from the change on the draws hold the
    shares."""
    sizes = {"a": 240, "b": 320, "c": 200, "d": 160}
    rows = {
        name: Dataset.from_list(
            [{"messages": [{"role": "assistant", "content": f"{name}{i:03d}" + "-" * (i % 4)}]} for i in range(size)]
        )
        for name, size in sizes.items()
    }

    def loader(weights: dict[str, float], dp_rank: int, dp_size: int, num_workers: int):
        config = SFTDataConfig(
            source=[{"name": name, "dataset": "unused"} for name in weights],
            batch_size=1,
            micro_batch_size=1,
            seq_len=16,
            num_workers=num_workers,
        )
        sources = [HFSource(rows[name], lambda _: dummy_renderer, name=name, seq_len=16) for name in weights]
        dataset = SFTDataset(sources, weights=list(weights.values()), dp_rank=dp_rank, dp_size=dp_size)
        return dataset, setup_dataloader(dataset, config, cp=1)

    def read(dataiter, micro_batches: int) -> list[str]:
        contents = []
        for _ in range(micro_batches):
            batch = next(dataiter)
            for targets in batch["target_ids"][0].split(batch["seq_lens"].tolist()):
                contents.append("".join(chr(token - 2) for token in targets.tolist() if token > 1))
        return [content for content in contents if content]

    weights_before = {"a": 1, "b": 1, "d": 1}
    before = [loader(weights_before, rank, gpus_before, 2) for rank in range(gpus_before)]
    # Every GPU gets further than the one before, as packing samples of other lengths lets a GPU do.
    contents = [content for rank, (_, dl) in enumerate(before) for content in read(iter(dl), 8 + 4 * rank)]
    directory = tmp_path / "dataloader"
    directory.mkdir()
    saved = []
    for rank, (_, dataloader) in enumerate(before):
        state = dataloader.state_dict()
        for worker in state["_snapshot"]["_worker_snapshots"].values():
            if old_state:
                for key in ("layout", "start", "sources"):
                    del worker["dataset_state"]["dataset"][key]
            saved.append(worker["dataset_state"]["dataset"])
        torch.save(state, directory / f"rank_{rank}.pt")
    if old_state:
        with pytest.raises(ValueError, match="resume once with the sources it was saved with"):
            loader({"a": 1, "b": 1, "c": 1}, 0, 2, 2)[0].load_state_dict(saved[0])

    # As the trainer does: GPU g (data rank g // cp) loads its saved file, or rank 0's, for its workers; then the
    # ranks decide together.
    after = []
    for gpu in range(gpus_after):
        dataset, dataloader = loader(weights_after, gpu // cp_after, gpus_after // cp_after, workers_after)
        path = directory / f"rank_{gpu}.pt"
        dataloader.load_state_dict(
            with_workers(
                torch.load(path if path.exists() else directory / "rank_0.pt", weights_only=False), workers_after
            )
        )
        after.append((dataset, dataloader))
    views = [dataset.saved_view(dl, directory, gpu, gpus_after) for gpu, (dataset, dl) in enumerate(after)]
    skipped = {dataset.start_together(dl, views) for dataset, dl in after}
    iterators = [iter(dl) for _, dl in after]
    furthest = max(state["step"] for state in saved)
    if not old_state:
        # Every draw before the common start was trained on before it, or is counted as skipped.
        assert skipped == {furthest - len(contents)}
    assert {state["step"] for _, dl in after for state in get_dataset_state(dl).values()} == {furthest}
    since = [read(dataiter, 8) for dataiter in iterators]

    assert all(since[gpu] == since[gpu - gpu % cp_after] for gpu in range(gpus_after))
    since_resume = [content for gpu in range(0, gpus_after, cp_after) for content in since[gpu]]
    assert len(contents + since_resume) == len(set(contents + since_resume))
    assert not any(content[0] in weights_before.keys() - weights_after.keys() for content in since_resume)
    shares = {name: weight / sum(weights_after.values()) for name, weight in weights_after.items()}
    for _, dataloader in after:
        for state in get_dataset_state(dataloader).values():
            drawn = {name: state["draws"][name] - state["start"][name] for name in shares}
            # Weighted fair queuing stays within 2 draws of every share (1.2 at most for these weights).
            assert all(abs(drawn[name] - share * sum(drawn.values())) < 2 for name, share in shares.items())
