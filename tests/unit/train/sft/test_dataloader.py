import os

import pytest
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import FakeDataConfig
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.world import reset_world

pytestmark = [pytest.mark.gpu]


def test_fake_dataset_single_pack_function():
    """Tests the single pack function which yields one sample per batch with padding."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="variable", input_ids="increasing", batch_size=1, pack_function="single")
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    for _ in range(4):
        micro_batch = next(dataiter)
        # Each micro batch should have exactly seq_len tokens
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["target_ids"].shape == (1, 128)
        assert micro_batch["position_ids"].shape == (1, 128)
        assert micro_batch["loss_mask"].shape == (1, 128)


def test_fake_dataset_single_pack_function_micro_batch():
    """Tests the single pack function with micro_batch_size > 1."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(
        length="variable", input_ids="increasing", batch_size=4, micro_batch_size=2, pack_function="single"
    )
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    for _ in range(4):
        micro_batch = next(dataiter)
        # Each micro batch should have micro_batch_size samples
        assert micro_batch["input_ids"].shape == (2, 128)
        assert micro_batch["target_ids"].shape == (2, 128)
        assert micro_batch["position_ids"].shape == (2, 128)
        assert micro_batch["loss_mask"].shape == (2, 128)


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
        num_packed_examples = len(micro_batch["input_ids"].unique())
        step += num_packed_examples
        assert micro_batch["input_ids"].shape == (1, 128)
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step, "epoch": 0}}
