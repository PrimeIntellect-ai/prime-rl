import torch

from prime_rl.configs.trainer import AdamWConfig, SGDConfig
from prime_rl.trainer.optim import _create_optimizer


class _DummyParallelDims:
    pass


def _make_named_params() -> list[tuple[str, torch.nn.Parameter]]:
    trainable = torch.nn.Parameter(torch.randn(4), requires_grad=True)
    frozen = torch.nn.Parameter(torch.randn(4), requires_grad=False)
    return [("trainable", trainable), ("frozen", frozen)]


def test_create_optimizer_adamw_ignores_frozen_params():
    named_params = _make_named_params()
    optimizer = _create_optimizer(AdamWConfig(type="adamw"), named_params, _DummyParallelDims())
    param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(named_params[0][1]) in param_ids
    assert id(named_params[1][1]) not in param_ids


def test_create_optimizer_sgd_ignores_frozen_params():
    named_params = _make_named_params()
    optimizer = _create_optimizer(SGDConfig(type="sgd"), named_params, _DummyParallelDims())
    param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(named_params[0][1]) in param_ids
    assert id(named_params[1][1]) not in param_ids
