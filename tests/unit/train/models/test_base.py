import pytest
import torch
from torch import nn

from prime_rl.trainer.models.base import _run_init_buffers_post_meta


class _BufferOwnerNoHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("x", torch.zeros(3))


class _BufferOwnerWithHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("x", torch.zeros(3))

    def init_buffers_post_meta(self) -> None:
        self.x.zero_()


class _Wrapper(nn.Module):
    def __init__(self, child: nn.Module):
        super().__init__()
        self.child = child


def test_run_init_buffers_post_meta_raises_for_uncovered_buffer_owner():
    with pytest.raises(TypeError, match="init_buffers_post_meta"):
        _run_init_buffers_post_meta(_Wrapper(_BufferOwnerNoHook()))


def test_run_init_buffers_post_meta_dispatches_to_submodule_hook():
    _run_init_buffers_post_meta(_Wrapper(_BufferOwnerWithHook()))


def test_run_init_buffers_post_meta_respects_exempt_types():
    _run_init_buffers_post_meta(_Wrapper(_BufferOwnerNoHook()), exempt=(_BufferOwnerNoHook,))
