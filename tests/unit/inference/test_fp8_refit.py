import sys
from types import ModuleType, SimpleNamespace

import torch

from prime_rl.inference.vllm.worker.fp8_refit import (
    load_checkpoint_weights_layerwise,
    maybe_convert_weights_for_fp8_refit,
)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded_weights: list[tuple[str, torch.Tensor]] | None = None

    def load_weights(self, weights_iter):
        self.loaded_weights = list(weights_iter)
        return "loaded"


def _install_reload_stub(monkeypatch, events: list[object]) -> None:
    reload_module = ModuleType("vllm.model_executor.model_loader.reload")
    model_loader_module = ModuleType("vllm.model_executor.model_loader")
    model_executor_module = ModuleType("vllm.model_executor")
    vllm_module = ModuleType("vllm")

    def initialize_layerwise_reload(model):
        events.append(("init", model))

    def finalize_layerwise_reload(model, model_config):
        events.append(("finalize", model, model_config))

    reload_module.initialize_layerwise_reload = initialize_layerwise_reload
    reload_module.finalize_layerwise_reload = finalize_layerwise_reload
    model_loader_module.reload = reload_module
    model_executor_module.model_loader = model_loader_module
    vllm_module.model_executor = model_executor_module

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader", model_loader_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.reload", reload_module)


def test_maybe_convert_weights_for_fp8_refit_without_vllm_config():
    model_runner = SimpleNamespace()
    model = _FakeModel()
    weights = [("layer.weight", torch.ones(1, 1, dtype=torch.bfloat16))]

    assert maybe_convert_weights_for_fp8_refit(model_runner, model, weights) is weights


def test_load_checkpoint_weights_layerwise_falls_back_without_vllm_config(monkeypatch):
    events: list[object] = []
    _install_reload_stub(monkeypatch, events)

    model = _FakeModel()
    load_config = SimpleNamespace(device="cpu")
    device_config = SimpleNamespace(device="cpu")
    model_config = SimpleNamespace(model="dummy")
    model_runner = SimpleNamespace(
        load_config=load_config,
        device_config=device_config,
        model_config=model_config,
    )
    weights = [("layer.weight", torch.ones(1, 1, dtype=torch.bfloat16))]

    result = load_checkpoint_weights_layerwise(model_runner, model, weights)

    assert result == "loaded"
    assert model.loaded_weights == weights
    assert events == [("init", model), ("finalize", model, model_config)]
