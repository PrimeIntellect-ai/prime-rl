import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from prime_rl.configs.trainer import AdamWConfig
from prime_rl.specdecode.training import draft_loss_mask
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims


@pytest.mark.parametrize("predict_next", [False, True])
def test_draft_mask_excludes_cross_document_blocks(predict_next):
    mask = torch.tensor([[False, False, True, True, True, True, True, False, False, True, True, True]])
    original = mask.clone()
    lengths = torch.tensor([7, 5])
    actual = draft_loss_mask(mask, lengths, block_size=3, predict_next=predict_next)
    expected = torch.zeros_like(mask)
    for start, end in ((0, 7), (7, 12)):
        for position in range(start, end - 3):
            expected[0, position] = mask[0, position + int(predict_next)]
    assert torch.equal(actual, expected)
    assert torch.equal(mask, original)


def test_draft_mask_handles_empty_loss_and_short_documents():
    lengths = torch.tensor([2, 3, 2])
    assert not draft_loss_mask(torch.ones(1, 7, dtype=torch.bool), lengths, 3, True).any()
    assert not draft_loss_mask(torch.zeros(1, 7, dtype=torch.bool), torch.tensor([7]), 3, True).any()


@pytest.mark.parametrize("freeze_backbone", [False, True])
def test_draft_learning_rate_survives_checkpoint_restore(freeze_backbone):
    model = torch.nn.ModuleDict({"backbone": torch.nn.Linear(2, 2), "speculator": torch.nn.Linear(2, 2)})
    model["backbone"].requires_grad_(not freeze_backbone)
    dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)
    optimizer, _ = setup_optimizer(
        AdamWConfig(lr=1e-3), list(model.named_parameters()), dims, lr_overrides={"speculator.": 1e-2}
    )
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    model_state, optimizer_state = get_state_dict(model, optimizer)
    set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=optimizer_state)
    assert all(group["params"] for group in optimizer.param_groups)
    rates = {id(parameter): group["lr"] for group in optimizer.param_groups for parameter in group["params"]}
    assert rates[id(model["speculator"].weight)] == 1e-2
    if not freeze_backbone:
        assert rates[id(model["backbone"].weight)] == 1e-3
    optimizer.step()
