"""Checkpoint resume when trainable params never accumulated optimizer state (#2676)."""

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.optim import AdamW

from prime_rl.trainer.ckpt import AppState, load_trainer_checkpoint


class _TwoParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trained = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.never_trained = nn.Parameter(torch.tensor([3.0, 4.0]))


def _save_partial_optimizer_checkpoint(tmp_path):
    """Write a checkpoint holding optimizer state for only one of two parameters."""
    model = _TwoParamModel()
    optimizer = AdamW(model.parameters(), lr=0.1)
    model.trained.grad = torch.ones_like(model.trained)
    optimizer.step()
    trained_step = optimizer.state[model.trained]["step"].item()
    ckpt_path = tmp_path / "trainer"
    dcp_save({"app": AppState(model, [optimizer], None, None)}, checkpoint_id=ckpt_path, no_dist=True)
    return ckpt_path, trained_step


def test_trainer_resume_without_never_trained_optimizer_state(tmp_path):
    ckpt_path, trained_step = _save_partial_optimizer_checkpoint(tmp_path)
    resumed_model = _TwoParamModel()
    resumed_optimizer = AdamW(resumed_model.parameters(), lr=0.1)

    load_trainer_checkpoint(ckpt_path, resumed_model, [resumed_optimizer], None, None)

    assert resumed_optimizer.state[resumed_model.trained]["step"].item() == trained_step
    assert resumed_model.never_trained in resumed_optimizer.state


@pytest.mark.parametrize("skip_optimizer", [False, True])
def test_trainer_resume_still_rejects_a_mismatched_model(tmp_path, skip_optimizer):
    """Tolerating absent optimizer keys must not make the model side permissive."""
    ckpt_path, _ = _save_partial_optimizer_checkpoint(tmp_path)
    mismatched_model = nn.Linear(2, 2)

    with pytest.raises((RuntimeError, CheckpointException)):
        load_trainer_checkpoint(
            ckpt_path,
            mismatched_model,
            [AdamW(mismatched_model.parameters(), lr=0.1)],
            None,
            None,
            skip_optimizer=skip_optimizer,
        )
