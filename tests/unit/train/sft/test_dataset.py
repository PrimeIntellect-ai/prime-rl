from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import FakeDataConfig, SFTDataConfig
from prime_rl.trainer.sft.data import FakeDataset, SFTDataset


def test_init_fake_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed")
    fake_dataset = FakeDataset(tokenizer, config)
    assert fake_dataset is not None


def test_fake_dataset_state():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", num_examples=2)
    dataset = FakeDataset(tokenizer, config)
    dataiter = iter(dataset)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 2, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 3, "epoch": 1}


def test_init_sft_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(num_examples=2)
    dataset = SFTDataset(tokenizer, config)
    assert dataset is not None


SAMPLE_TEMPLATE = """\
<|im_start|>user
Prompt {idx}<|im_end|>
<|im_start|>assistant
<think>

</think>

Completion {idx}<|im_end|>
"""


def test_sft_dataset_state():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(name="mikasenghaas/test-sft", num_examples=2)
    dataset = SFTDataset(tokenizer, config)
    dataiter = iter(dataset)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Step 1
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=0)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}

    # Step 2
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=1)
    assert dataset.state_dict() == {"step": 2, "epoch": 0}

    # Step 3 (next epoch)
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=0)
    assert dataset.state_dict() == {"step": 3, "epoch": 1}
