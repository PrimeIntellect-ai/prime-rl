import math

import torch

from prime_rl.configs.reward_model import RewardModelConfig
from prime_rl.trainer.reward_model.data import preference_collate
from prime_rl.trainer.reward_model.train import bradley_terry_losses


def test_bradley_terry_loss_and_gradient_prefer_chosen():
    chosen = torch.tensor([0.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = bradley_terry_losses(chosen, rejected).sum()
    loss.backward()

    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert chosen.grad.item() < 0
    assert rejected.grad.item() > 0


def test_preference_collate_orders_chosen_before_rejected():
    batch = preference_collate(
        [
            {"chosen_ids": [10, 11], "rejected_ids": [20]},
            {"chosen_ids": [12], "rejected_ids": [21, 22]},
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[10, 11], [12, 0], [20, 0], [21, 22]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0], [1, 0], [1, 1]]
    assert batch["num_pairs"] == 2


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
    assert config.data.type == "preference"
