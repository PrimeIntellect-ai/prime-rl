import math
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from renderers.base import RenderedTokens

from prime_rl.configs.reward_model import BradleyTerryDataConfig, RewardModelConfig
from prime_rl.trainer.reward_model import data as reward_model_data
from prime_rl.trainer.reward_model.data import BradleyTerryDataset, bradley_terry_collate
from prime_rl.trainer.reward_model.train import bradley_terry_losses


def test_bradley_terry_loss_and_gradient_prefer_chosen():
    chosen = torch.tensor([0.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = bradley_terry_losses(chosen, rejected).sum()
    loss.backward()

    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert chosen.grad.item() < 0
    assert rejected.grad.item() > 0


def test_bradley_terry_collate_orders_chosen_before_rejected():
    batch = bradley_terry_collate(
        [
            {"chosen_ids": [10, 11], "rejected_ids": [20]},
            {"chosen_ids": [12], "rejected_ids": [21, 22]},
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[10, 11], [12, 0], [20, 0], [21, 22]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0], [1, 0], [1, 1]]
    assert batch["num_pairs"] == 2
    assert batch["pair_weights"].tolist() == [1.0, 1.0]


class _IndexedRenderer:
    def render(self, messages, **kwargs):
        token_id = int(messages[-1]["content"])
        return RenderedTokens(
            token_ids=[token_id],
            message_indices=[len(messages) - 1],
            sampled_mask=[True],
        )

    def get_stop_token_ids(self):
        return [99]


def test_validation_padding_preserves_distributed_tail(monkeypatch):
    raw_dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "prompt"}],
                "chosen": [{"role": "assistant", "content": str(10 + index)}],
                "rejected": [{"role": "assistant", "content": str(20 + index)}],
            }
            for index in range(3)
        ]
    )
    config = BradleyTerryDataConfig(name="unused", shuffle=False)
    rank_samples = []
    for rank in range(2):
        monkeypatch.setattr(
            reward_model_data,
            "get_world",
            lambda rank=rank: SimpleNamespace(rank=rank, world_size=2),
        )
        dataset = BradleyTerryDataset(
            raw_dataset,
            _IndexedRenderer(),
            config,
            max_epochs=1,
            pad_to_data_world_size=True,
        )
        rank_samples.append(list(dataset))

    assert [len(samples) for samples in rank_samples] == [2, 2]
    assert sum(sample["sample_weight"] for samples in rank_samples for sample in samples) == 3
    assert rank_samples[1][-1]["sample_weight"] is False


def test_reward_model_config_keeps_sft_numerical_defaults():
    config = RewardModelConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B", "impl": "hf", "attn": "flash_attention_2"},
            "data": {"name": "preferences.jsonl"},
            "max_steps": 1,
        }
    )

    assert config.model.optimization_dtype == "float32"
    assert config.model.reduce_dtype == "float32"
    assert config.data.type == "bradley_terry"


def test_reward_model_config_rejects_slurm():
    with pytest.raises(ValueError, match="SLURM launch is not implemented"):
        RewardModelConfig.model_validate(
            {
                "model": {"name": "Qwen/Qwen3-0.6B", "impl": "hf", "attn": "flash_attention_2"},
                "data": {"name": "preferences.jsonl"},
                "slurm": {},
                "max_steps": 1,
            }
        )
