import inspect
from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.minimax_rms_norm import rms_norm_tp
from vllm.model_executor.model_loader.reload.layerwise import _get_original_loader

from prime_rl.inference.patches import (
    monkey_patch_minimax_m2_named_norm_loaders,
    monkey_patch_moe_runner_loads_score_correction_bias,
)


class _RoutedExperts(torch.nn.Module):
    def __init__(self, with_bias):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        if with_bias:
            self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self.seen = []

    def load_weights(self, weights):
        for name, weight in weights:
            self.seen.append(name)
            if name == "w13_weight":
                self.w13_weight.data.copy_(weight)
                yield name


def test_moe_runner_loads_score_correction_bias(monkeypatch):
    monkeypatch.setattr(MoERunner, "load_weights", MoERunner.load_weights)
    monkey_patch_moe_runner_loads_score_correction_bias()
    patched = MoERunner.load_weights
    monkey_patch_moe_runner_loads_score_correction_bias()
    assert MoERunner.load_weights is patched

    runner = SimpleNamespace(routed_experts=_RoutedExperts(with_bias=True))
    weights = [("w13_weight", torch.ones(2, 2)), ("e_score_correction_bias", torch.tensor([0.5, -0.5]))]
    loaded = list(MoERunner.load_weights(runner, iter(weights)))

    assert loaded == ["routed_experts.w13_weight", "routed_experts.e_score_correction_bias"]
    assert runner.routed_experts.seen == ["w13_weight"]
    torch.testing.assert_close(runner.routed_experts.e_score_correction_bias, torch.tensor([0.5, -0.5]))


def test_moe_runner_without_bias_is_unchanged(monkeypatch):
    monkeypatch.setattr(MoERunner, "load_weights", MoERunner.load_weights)
    monkey_patch_moe_runner_loads_score_correction_bias()
    runner = SimpleNamespace(routed_experts=_RoutedExperts(with_bias=False))
    weights = [("w13_weight", torch.ones(2, 2)), ("e_score_correction_bias", torch.ones(2))]
    assert list(MoERunner.load_weights(runner, iter(weights))) == ["routed_experts.w13_weight"]
    assert runner.routed_experts.seen == ["w13_weight", "e_score_correction_bias"]


def test_minimax_norm_loader_is_named_for_layerwise_reload(monkeypatch):
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(rms_norm_tp.MiniMaxText01RMSNormTP, "__init__", rms_norm_tp.MiniMaxText01RMSNormTP.__init__)
    monkey_patch_minimax_m2_named_norm_loaders()
    patched = rms_norm_tp.MiniMaxText01RMSNormTP.__init__
    monkey_patch_minimax_m2_named_norm_loaders()
    assert rms_norm_tp.MiniMaxText01RMSNormTP.__init__ is patched

    with set_current_vllm_config(VllmConfig()):
        norm = rms_norm_tp.MiniMaxText01RMSNormTP(4)
    loader = norm.weight.weight_loader
    assert loader.__name__ == "weight_loader"
    assert _get_original_loader(norm.weight) is loader
    assert list(inspect.signature(loader).parameters)[:2] == ["param", "loaded_weight"]
    loader(norm.weight, torch.arange(4.0))
    torch.testing.assert_close(norm.weight.data, torch.arange(4.0))
