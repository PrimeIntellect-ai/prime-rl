from dataclasses import dataclass

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from prime_rl.trainer.config import BatchConfig
from prime_rl.trainer.sft.config import DataConfig
from prime_rl.utils.logger import get_logger
from prime_rl.trainer.batch import SFTSample


class FakeDataset(Dataset):
    """A dataset of fake tokens"""

    def __init__(self, tokenizer: AutoTokenizer, batch_config: BatchConfig, data_config: DataConfig):
        self.batch_config = batch_config
        self.data_config = data_config
        self.vocab_size = tokenizer.vocab_size

    def __len__(self) -> int:
        return self.config.fake.n

    def __getitem__(self, index: int) -> SFTSample:
        input_ids = torch.randint(0, self.vocab_size, (self.batch_config.seq_len + 1,)).long()
        position_ids = torch.arange(len(input_ids)).long()
        loss_mask = torch.ones(len(input_ids)).bool()
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class SFTDataset(Dataset):
    """A dataset wrapping a HF SFT dataset with prompt + completion format."""

    def __init__(self, tokenizer: AutoTokenizer, data_config: DataConfig, batch_config: BatchConfig):
        assert not data_config.fake, "HFDataset does not support fake data"
        self.data_config = data_config
        self.batch_config = batch_config
        self.tokenizer = tokenizer
        self._logger = get_logger()

        # Load dataset
        self.dataset: HFDataset = load_dataset(data_config.name, split=data_config.split)

        # Assert that the dataset has a 'text' column
        if "prompt" not in self.dataset.column_names or "completion" not in self.dataset.column_names:
            raise ValueError("HF dataset must have a 'prompt' and 'completion' column for SFT")

        # Preprocess dataset (apply chat template and tokenize)
        columns = self.dataset.column_names
        self.samples = self.dataset.map(self._preprocess, remove_columns=columns).to_list()

    def _preprocess(self, example: dict) -> SFTSample:
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        assert "prompt" in example and "completion" in example, "Prompt and completion must be present in the example"
        assert isinstance(example["prompt"], list) and isinstance(example["completion"], list), (
            "Prompt and completion must be lists"
        )

        prompt_ids = self.tokenizer.apply_chat_template(
            example["prompt"],
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )
        prompt_completion_ids = self.tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )

        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
            self._logger.warning(
                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                "token handling. Verify that the tokenizer is processing text consistently."
            )

        # Create sample
        sample = {
            "input_ids": prompt_completion_ids,
            "position_ids": list(range(len(prompt_completion_ids))),
            "loss_mask": [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids)),
        }

        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SFTSample:
        return self.samples[index]


def setup_dataset(tokenizer: AutoTokenizer, data_config: DataConfig, batch_config: BatchConfig) -> Dataset:
    if data_config.fake:
        return FakeDataset(tokenizer, batch_config, data_config)
    return SFTDataset(tokenizer, data_config, batch_config)
