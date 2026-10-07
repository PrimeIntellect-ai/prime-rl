import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer import ckpt


@pytest.mark.parametrize("skip_optimizer", [False, True])
def test_checkpoint_save_state_dict(monkeypatch, tmp_path, skip_optimizer):
    model = nn.Linear(2, 2)  # FP32 master weights; bf16 is only the compute dtype.
    model.register_buffer("count", torch.tensor(1))
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2)).sum().backward()
    optimizer.step()

    saved = {}

    def capture_save(state_dict, checkpoint_id):
        saved.update(state_dict["app"].state_dict())

    monkeypatch.setattr(ckpt, "dcp_save", capture_save)
    manager = ckpt.CheckpointManager(tmp_path, CheckpointConfig(skip_optimizer=skip_optimizer))
    manager.save_to_path(tmp_path / "trainer", model, [optimizer], None, ckpt.Progress())

    assert set(saved["model"]) == {"weight", "bias", "count"}
    assert saved["model"]["weight"].dtype == (torch.bfloat16 if skip_optimizer else torch.float32)
    assert saved["model"]["bias"].dtype == (torch.bfloat16 if skip_optimizer else torch.float32)
    assert saved["model"]["count"].dtype == torch.int64
    if skip_optimizer:
        assert "optimizers" not in saved
    else:
        assert saved["optimizers"]["state"]
    assert model.weight.dtype == torch.float32


@pytest.mark.parametrize("skip_optimizer", [False, True])
def test_checkpoint_disk_round_trip(tmp_path, skip_optimizer):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2)).sum().backward()
    optimizer.step()
    manager = ckpt.CheckpointManager(tmp_path, CheckpointConfig(skip_optimizer=skip_optimizer))
    path = tmp_path / "trainer"
    manager.save_to_path(path, model, [optimizer], None, ckpt.Progress())

    expected = model.weight.detach().to(torch.bfloat16 if skip_optimizer else torch.float32).clone()
    model.weight.data.zero_()
    manager.load_from_path(path, model, [optimizer], None, ckpt.Progress())
    torch.testing.assert_close(model.weight, expected.float(), rtol=0, atol=0)
