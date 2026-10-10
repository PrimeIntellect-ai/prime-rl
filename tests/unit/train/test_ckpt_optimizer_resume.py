"""Resume when a trainable parameter never accumulated optimizer state (#2676).

A parameter with ``requires_grad=True`` that never received a gradient has no entry in
``optimizer.state``, so the save side writes no optimizer shard for it. The load side
builds its template from a fresh optimizer, whose empty state makes torch materialize
state for every ``requires_grad`` parameter, so a strict load demands keys the
checkpoint never wrote.

These are CPU tests on purpose: the mechanism has nothing to do with CUDA, and the
existing offload coverage in ``test_state_offload.py`` is gated behind the ``gpu``
marker, so this path had no CPU coverage at all.
"""

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.optim import AdamW

from prime_rl.trainer.ckpt import AppState, load_trainer_checkpoint


class TwoParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trained = nn.Parameter(torch.randn(4, 4))
        self.never_trained = nn.Parameter(torch.randn(4, 4))


def save_partial_optimizer_checkpoint(path) -> float:
    """Write a checkpoint whose optimizer state covers only `trained`."""
    model = TwoParamModel()
    optimizer = AdamW(model.parameters(), lr=0.1)
    model.trained.grad = torch.ones_like(model.trained)
    optimizer.step()
    optimizer.zero_grad()

    assert model.never_trained not in optimizer.state, "precondition: no state for the untrained param"
    step = optimizer.state[model.trained]["step"].item()

    dcp_save({"app": AppState(model, [optimizer], None, None)}, checkpoint_id=path, no_dist=True)
    return step


def test_resume_restores_state_for_trained_params(tmp_path):
    step = save_partial_optimizer_checkpoint(tmp_path / "trainer")

    # The defect itself: a strict load of this checkpoint demands optimizer keys that
    # were never written, because the fresh optimizer materializes state for every
    # `requires_grad` parameter. This assertion pins the root cause, so the partial
    # load below cannot be mistaken for defensive coding.
    model = TwoParamModel()
    optimizer = AdamW(model.parameters(), lr=0.1)
    with pytest.raises((RuntimeError, CheckpointException)):
        dcp_load({"app": AppState(model, [optimizer], None, None)}, checkpoint_id=tmp_path / "trainer")

    model = TwoParamModel()
    optimizer = AdamW(model.parameters(), lr=0.1)

    load_trainer_checkpoint(tmp_path / "trainer", model, [optimizer], None, None)

    assert optimizer.state[model.trained]["step"].item() == step
    assert model.never_trained in optimizer.state, "a never-trained param must be trainable after resume"


def test_resume_still_rejects_a_model_with_an_unexpected_parameter(tmp_path):
    """Tolerating absent optimizer keys must not make the model side permissive."""
    save_partial_optimizer_checkpoint(tmp_path / "trainer")

    model = TwoParamModel()
    model.extra = nn.Parameter(torch.randn(4, 4))
    optimizer = AdamW(model.parameters(), lr=0.1)

    with pytest.raises((RuntimeError, CheckpointException)):
        load_trainer_checkpoint(tmp_path / "trainer", model, [optimizer], None, None)


def test_resume_skipping_the_optimizer_ignores_absent_optimizer_state(tmp_path):
    save_partial_optimizer_checkpoint(tmp_path / "trainer")

    model = TwoParamModel()
    optimizer = AdamW(model.parameters(), lr=0.1)

    load_trainer_checkpoint(tmp_path / "trainer", model, [optimizer], None, None, skip_optimizer=True)

    assert not optimizer.state, "skip_optimizer must not restore optimizer state"
