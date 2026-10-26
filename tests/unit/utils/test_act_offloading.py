import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.utils.act_offloading import OffloadActivations

pytestmark = [pytest.mark.gpu]


def make_model() -> nn.Module:
    torch.manual_seed(0)
    wrap = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode="full"))
    layers = [wrap(nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))) for _ in range(8)]
    return nn.Sequential(*layers, nn.Linear(256, 256)).cuda()


@pytest.mark.parametrize("max_inflight", [1, 3])
def test_offloaded_gradients_match(max_inflight):
    x = torch.randn(4096, 256, device="cuda")
    reference = make_model()
    reference(x).square().sum().backward()

    model = make_model()
    offloader = OffloadActivations(model, max_inflight=max_inflight)
    with offloader:
        loss = model(x).square().sum()
    assert len(offloader.on_cpu) > 8
    loss.backward()

    assert not offloader.on_cpu and not offloader.prefetched and not offloader.kept
    for param, ref in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref.grad, rtol=0, atol=0)


def test_parameter_views_stay_on_gpu():
    model = nn.Linear(256, 256).cuda()
    offloader = OffloadActivations(model)
    with offloader:
        model(torch.randn(4096, 256, device="cuda", requires_grad=True)).sum()
    weight_storage = model.weight.untyped_storage().data_ptr()
    assert any(t.untyped_storage().data_ptr() == weight_storage for t in offloader.kept.values())
    assert all(t.untyped_storage().data_ptr() != weight_storage for t in offloader.on_cpu.values())


def test_tensor_saved_twice_is_offloaded_once():
    model = nn.Linear(256, 256).cuda()
    x = torch.randn(4096, 256, device="cuda", requires_grad=True)
    shared = torch.randn(4096, 256, device="cuda")
    offloader = OffloadActivations(model)
    with offloader:
        loss = (x * shared).sum() + (x.sin() * shared.view(-1)[: 4096 * 256].view(4096, 256)).sum()
    assert len(offloader.on_cpu) == 2
    loss.backward()
    assert not offloader.on_cpu and not offloader.aliases and not offloader.refs
    assert torch.equal(x.grad, shared + x.detach().cos() * shared)
