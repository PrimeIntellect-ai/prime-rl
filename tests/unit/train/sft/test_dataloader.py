import os

import pytest
import torch
from transformers import AutoTokenizer

from prime_rl.configs.sft import FakeDataConfig
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.world import reset_world

pytestmark = [pytest.mark.gpu]


def test_fake_dataset_single_rank_state():
    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    # Initial state
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 0, "epoch": 0}}

    # Iterate over samples
    micro_batch = next(dataiter)
    print(micro_batch)
    assert micro_batch["input_ids"].unique().item() == 0
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 1, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 2, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 3, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 4, "epoch": 0}}


@pytest.mark.parametrize("rank", [0, 1], ids=["rank0", "rank1"])
def test_fake_dataset_multi_rank_state(rank: int):
    # Setup world
    reset_world()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(2)

    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    # Initial state
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 0, "epoch": 0}}

    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 0 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 1 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 3 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 4 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 5 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 6 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 7 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 8 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 9 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 10 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 11 + rank, "epoch": 0}}


def test_fake_dataset_single_rank_resume():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    # First 2 samples
    for step in range(2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step + 1, "epoch": 0}}

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
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step + 1, "epoch": 0}}


def test_fake_dataset_single_rank_state_with_packing():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="variable", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    step = 0
    for _ in range(8):
        micro_batch = next(dataiter)
        num_packed_examples = len(micro_batch["input_ids"][micro_batch["loss_mask"]].unique())
        step += num_packed_examples
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["seq_lens"].sum() == micro_batch["input_ids"].shape[1]
        if micro_batch["padding_len"]:
            assert micro_batch["seq_lens"][-1] == micro_batch["padding_len"]
        dataset_state = dataloader.state_dict()["dataset_state"]
        pending_sample = dataset_state.get("pending_sample")
        expected_dataset_step = step + (pending_sample is not None)
        assert dataset_state["dataset"] == {"step": expected_dataset_step, "epoch": 0}
        if pending_sample is not None:
            assert pending_sample["input_ids"][0] == step

    state_dict = dataloader.state_dict()
    rng_state = torch.random.get_rng_state()
    expected_batch = next(dataiter)

    resumed_dataset = setup_dataset(tokenizer, config)
    resumed_dataloader = setup_dataloader(resumed_dataset, config)
    resumed_dataloader.load_state_dict(state_dict)
    resumed_dataiter = iter(resumed_dataloader)
    torch.random.set_rng_state(rng_state)
    resumed_batch = next(resumed_dataiter)

    for key in ("input_ids", "position_ids", "target_ids", "loss_mask", "seq_lens"):
        torch.testing.assert_close(resumed_batch[key], expected_batch[key])
    assert resumed_batch["padding_len"] == expected_batch["padding_len"]
