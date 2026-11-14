import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.utils.act_offloading import OffloadActivations


@pytest.fixture
def offloader(monkeypatch):
    was_enabled = gc.isenabled()
    gc.collect(2)
    gc.disable()
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_stream", nullcontext)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda *args: torch.device("cpu"))
    yield OffloadActivations
    gc.collect(2)
    if was_enabled:
        gc.enable()


def forward_backward(offloader, elements, keep_output=False):
    value = torch.ones(elements, requires_grad=True)
    manager = offloader(use_pin_memory=False)
    reference = weakref.ref(manager)
    with manager:
        used = value.sin().sin()
        unused = value.sin().sin()
        loss = used.sum()
    loss.backward()
    expected = torch.cos(value.detach()) * torch.cos(torch.sin(value.detach()))
    torch.testing.assert_close(value.grad, expected)
    return (reference, unused) if keep_output else reference


@pytest.mark.parametrize("elements", [1, 1024])
def test_unused_graph_releases_offloader_without_collection(offloader, elements):
    reference = forward_backward(offloader, elements)
    assert reference() is None


def test_live_graph_survives_collection_then_releases_offloader(offloader):
    reference, output = forward_backward(offloader, 1024, keep_output=True)
    gc.collect(1)
    assert reference() is not None
    del output
    assert reference() is None


def test_backward_after_context_exit_and_collection(offloader):
    value = torch.ones(1024, requires_grad=True)
    manager = offloader(use_pin_memory=False)
    reference = weakref.ref(manager)
    with manager:
        output = value.sin().sin()
    del manager
    gc.collect(1)
    output.sum().backward()
    expected = torch.cos(value.detach()) * torch.cos(torch.sin(value.detach()))
    torch.testing.assert_close(value.grad, expected)
    del output
    assert reference() is None


def test_forward_exception_releases_offloader(offloader):
    manager = offloader(use_pin_memory=False)
    reference = weakref.ref(manager)
    with pytest.raises(ValueError, match="test forward failure"):
        with manager:
            raise ValueError("test forward failure")
    del manager
    assert reference() is None


def test_context_can_be_reused_after_backward(offloader):
    manager = offloader(use_pin_memory=False)
    for _ in range(2):
        value = torch.ones(1024, requires_grad=True)
        with manager:
            loss = (value * value).sum()
        loss.backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, 2.0))
        assert not manager.tracker


def test_empty_context_can_be_reused(offloader):
    manager = offloader(use_pin_memory=False)
    with manager:
        pass
    with manager:
        pass


def test_reentry_preserves_pending_backward_graph(offloader):
    manager = offloader(use_pin_memory=False)
    first = torch.ones(1024, requires_grad=True)
    second = torch.full((1024,), 2.0, requires_grad=True)
    with manager:
        first_loss = (first * first).sum()
    with manager:
        second_loss = (second * second).sum()
    first_loss.backward()
    second_loss.backward()
    torch.testing.assert_close(first.grad, torch.full_like(first, 2.0))
    torch.testing.assert_close(second.grad, torch.full_like(second, 4.0))
    assert not manager.tracker


def test_selective_checkpoint_preserves_cached_metadata_identity(offloader):
    metadata = torch.tensor([0, 2, 4], dtype=torch.int32)
    cache = []

    def cached_lengths(value):
        for key, result in cache:
            if key is value:
                return result
        result = torch.diff(value)
        cache.append((value, result))
        return result

    cached_lengths(metadata)

    class Block(torch.nn.Module):
        def forward(self, value, cu_seqlens):
            scale = cached_lengths(cu_seqlens).sum()
            return value.sin().sin() * scale

    checkpointed = get_activation_checkpoint_wrapper(SimpleNamespace(mode="full"))(Block())
    value = torch.ones(1024, requires_grad=True)
    with offloader(use_pin_memory=False):
        output = checkpointed(value, metadata)
    output.sum().backward()
    expected = 4 * torch.cos(value.detach()) * torch.cos(torch.sin(value.detach()))
    torch.testing.assert_close(value.grad, expected)
    assert len(cache) == 1
