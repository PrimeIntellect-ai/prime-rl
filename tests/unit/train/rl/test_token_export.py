import math

import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.token_export import _export_columns


def test_token_export_masks_rl_diagnostics_without_nan_sentinel():
    micro_batch = {
        "input_ids": torch.tensor([101, 102, 103]),
        "position_ids": torch.tensor([0, 1, 2]),
        "loss_mask": torch.tensor([True, True, True]),
        "echo_mask": torch.tensor([False, True, False]),
        "advantages": torch.tensor([1.0, 0.5, -1.0]),
        "rewards": None,
        "inference_logprobs": torch.tensor([-0.2, -0.3, -0.4]),
        "env_names": ["env", "env", "env"],
        "training_mode": "rl",
    }
    model_output = {
        "logprobs": torch.tensor([-0.1, -0.5, -0.45]),
        "entropy": torch.tensor([1.0, 1.1, 1.2]),
    }

    columns = _export_columns(micro_batch, model_output, DefaultLossConfig())

    assert columns["rl_loss_mask"] == [True, False, True]
    for key in ("mismatch_kl", "log_importance_ratio", "importance_ratio", "prob_delta"):
        assert columns[key][1] is None
        assert math.isfinite(columns[key][0])
        assert math.isfinite(columns[key][2])
