import os

import pytest
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import FakeDataConfig, SFTDataConfig
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.world import reset_world

pytestmark = [pytest.mark.gpu]


def test_stateful_dataloader_single_rank():
    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", num_examples=2, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 0
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 0, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 1, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 2, "epoch": 1}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 3, "epoch": 1}}


@pytest.mark.parametrize("rank", [0, 1], ids=["rank0", "rank1"])
def test_stateful_dataloader_multi_rank(rank: int):
    # Setup world
    reset_world()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(2)

    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", num_examples=8, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 0 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 0 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 2 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 4 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 4 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 6 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 6 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 8 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 8 + rank, "epoch": 1}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 10 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 10 + rank, "epoch": 1}}


def test_stateful_dataloader_resume_fake():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    num_examples = 8
    config = FakeDataConfig(
        length="fixed", input_ids="increasing", num_examples=num_examples, batch_size=1, micro_batch_size=1
    )
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    # First 1/2 epoch
    for step in range(num_examples // 2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert micro_batch["epoch"] == 0
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step, "epoch": 0}}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second 1/2 epoch
    for step in range(num_examples // 2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == num_examples // 2 + step
        assert micro_batch["epoch"] == 0
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": num_examples // 2 + step, "epoch": 0}}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second epoch
    for step in range(num_examples):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == num_examples + step
        assert micro_batch["epoch"] == 1
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": num_examples + step, "epoch": 1}}


def test_dataloader_ckpt_with_packing():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    num_examples = 8
    config = FakeDataConfig(length="variable", input_ids="increasing", num_examples=8, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    step = 0
    for _ in range(num_examples):
        micro_batch = next(dataiter)
        num_packed_examples = len(micro_batch["input_ids"].unique())
        step += num_packed_examples
        epoch = (step - 1) // num_examples
        assert micro_batch["input_ids"].shape == (1, 128)
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step - 1, "epoch": epoch}}


SAMPLE_TEMPLATE = """\
<|im_start|>user
Prompt {idx}<|im_end|>
<|im_start|>assistant
<think>

</think>

Completion {idx}<|im_end|>\
"""


def test_stateful_dataloader_resume_sft():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    num_examples = 8
    config = SFTDataConfig(
        name="mikasenghaas/test-sft",
        num_examples=num_examples,
        seq_len=19,
        batch_size=1,
        micro_batch_size=1,
        shuffle=False,
    )
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    # First 1/2 epoch 0
    for step in range(num_examples // 2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 19)
        assert tokenizer.decode(micro_batch["input_ids"][0]) == SAMPLE_TEMPLATE.format(idx=step)
        assert micro_batch["epoch"] == 0
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step, "epoch": 0}}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second 1/2 epoch 0
    for step in range(num_examples // 2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 19)
        assert tokenizer.decode(micro_batch["input_ids"][0]) == SAMPLE_TEMPLATE.format(idx=num_examples // 2 + step)
        assert micro_batch["epoch"] == 0
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": num_examples // 2 + step, "epoch": 0}}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Epoch 1
    for step in range(num_examples):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 19)
        assert tokenizer.decode(micro_batch["input_ids"][0]) == SAMPLE_TEMPLATE.format(idx=step)
        assert micro_batch["epoch"] == 1
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": num_examples + step, "epoch": 1}}
