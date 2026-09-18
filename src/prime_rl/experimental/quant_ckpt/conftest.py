import pytest
import torch

from prime_rl.experimental.quant_ckpt.fake_kernels import _calls


@pytest.fixture(autouse=True)
def _reset():
    _calls.clear()
    torch._dynamo.reset()
    yield
