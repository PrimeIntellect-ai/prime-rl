import pytest
import torch
from torch import nn

from prime_rl.utils.act_offloading import OffloadActivations

pytestmark = [pytest.mark.gpu]


def test_parameter_views_stay_on_gpu():
    model = nn.Linear(256, 256).cuda()
    offloader = OffloadActivations(model)
    with offloader:
        loss = model(torch.randn(4096, 256, device="cuda", requires_grad=True)).sum()
    weight_storage = model.weight.untyped_storage().data_ptr()
    saved = [tensor for tensor, offloaded in offloader.tracker.values() if not offloaded]
    assert any(tensor.untyped_storage().data_ptr() == weight_storage for tensor in saved)
    loss.backward()


def test_tensor_saved_twice_is_offloaded_once():
    model = nn.Linear(256, 256).cuda()
    x = torch.randn(4096, 256, device="cuda", requires_grad=True)
    shared = torch.randn(4096, 256, device="cuda")
    offloader = OffloadActivations(model)
    with offloader:
        loss = (x * shared).sum() + (x.sin() * shared.view(-1)[: 4096 * 256].view(4096, 256)).sum()
    copies = {tensor.data_ptr() for tensor, offloaded in offloader.tracker.values() if offloaded}
    assert len(copies) == 2
    loss.backward()
    assert not offloader.shared_copies
    assert torch.equal(x.grad, shared + x.detach().cos() * shared)
