"""Checkpoint resume when trainable params never accumulated optimizer state (#2676)."""

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from prime_rl.trainer.ckpt import AppState, load_distributed_checkpoint, load_trainer_checkpoint


class _TwoParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trained = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.never_trained = nn.Parameter(torch.tensor([3.0, 4.0]))


def _save_partial_optimizer_checkpoint(tmp_path):
    model = _TwoParamModel()
    optimizer = AdamW(model.parameters(), lr=0.1)
    model.trained.grad = torch.ones_like(model.trained)
    optimizer.step()
    trained_step = optimizer.state[model.trained]["step"].item()
    ckpt_path = tmp_path / "trainer"
    dcp_save({"app": AppState(model, [optimizer], None, None)}, checkpoint_id=ckpt_path, no_dist=True)
    return ckpt_path, model, optimizer, trained_step


def test_strict_dcp_load_rejects_missing_optimizer_state(tmp_path):
    ckpt_path, _, _, trained_step = _save_partial_optimizer_checkpoint(tmp_path)
    resumed_model = _TwoParamModel()
    resumed_optimizer = AdamW(resumed_model.parameters(), lr=0.1)
    with pytest.raises((RuntimeError, CheckpointException), match="Missing key"):
        dcp_load(
            {"app": AppState(resumed_model, [resumed_optimizer], None, None)},
            checkpoint_id=ckpt_path,
            no_dist=True,
        )
    assert trained_step == 1.0


def test_trainer_resume_without_never_trained_optimizer_state(tmp_path):
    ckpt_path, model, optimizer, trained_step = _save_partial_optimizer_checkpoint(tmp_path)
    resumed_model = _TwoParamModel()
    resumed_optimizer = AdamW(resumed_model.parameters(), lr=0.1)
    load_trainer_checkpoint(ckpt_path, resumed_model, [resumed_optimizer], None, None)
    assert resumed_optimizer.state[resumed_model.trained]["step"].item() == trained_step
    assert resumed_model.never_trained in resumed_optimizer.state


def test_strict_model_pass_rejects_mismatched_architecture(tmp_path):
    ckpt_path, _, _, _ = _save_partial_optimizer_checkpoint(tmp_path)
    broken_model = nn.Linear(2, 2)
    with pytest.raises((RuntimeError, CheckpointException)):
        load_distributed_checkpoint(
            {"app": AppState(broken_model, [], None, None)},
            ckpt_path,
            allow_partial_load=False,
        )
