from types import SimpleNamespace

import pytest
import torch

import prime_rl.inference.vllm.worker.filesystem as filesystem_worker
import prime_rl.inference.vllm.worker.fp8_refit as fp8_refit


class _DummyLoadConfig:
    def __init__(self, *, device: str = "cpu", load_format: str = "safetensors"):
        self.device = device
        self.load_format = load_format


class _DummyRunner:
    def __init__(self, *, load_config: _DummyLoadConfig, model_path: str = "base-model"):
        self.vllm_config = SimpleNamespace(load_config=load_config)
        self.model_config = SimpleNamespace(model=model_path)
        self._model = object()

    def get_model(self):
        return self._model


def test_update_weights_uses_vllm_config_load_config_and_restores_model_path(monkeypatch):
    runner = _DummyRunner(load_config=_DummyLoadConfig())
    worker = filesystem_worker.FileSystemWeightUpdateWorker()
    worker.model_runner = runner

    observed: dict[str, object] = {}

    class _DummyLoader:
        def get_all_weights(self, model_config, model):
            observed["loader_model_path"] = model_config.model
            observed["loader_model"] = model
            return iter([("layer.weight", torch.ones(1))])

    monkeypatch.setattr(filesystem_worker, "get_model_loader", lambda load_config: _DummyLoader())
    monkeypatch.setattr(
        filesystem_worker,
        "load_checkpoint_weights_layerwise",
        lambda model_runner, model, weights_iter: observed.setdefault("weights", list(weights_iter)),
    )

    worker.update_weights_from_path("/tmp/checkpoint")

    assert observed["loader_model_path"] == "/tmp/checkpoint"
    assert observed["loader_model"] is runner.get_model()
    loaded_name, loaded_tensor = observed["weights"][0]
    assert loaded_name == "layer.weight"
    assert torch.equal(loaded_tensor, torch.ones(1))
    assert runner.model_config.model == "base-model"


def test_update_weights_restores_model_path_when_reload_fails(monkeypatch):
    runner = _DummyRunner(load_config=_DummyLoadConfig(), model_path="meta-llm")
    worker = filesystem_worker.FileSystemWeightUpdateWorker()
    worker.model_runner = runner

    class _DummyLoader:
        def get_all_weights(self, model_config, model):
            return iter([("layer.weight", torch.ones(1))])

    monkeypatch.setattr(filesystem_worker, "get_model_loader", lambda load_config: _DummyLoader())
    monkeypatch.setattr(
        filesystem_worker,
        "load_checkpoint_weights_layerwise",
        lambda model_runner, model, weights_iter: (_ for _ in ()).throw(RuntimeError("reload failed")),
    )

    with pytest.raises(RuntimeError, match="reload failed"):
        worker.update_weights_from_path("/tmp/failed-checkpoint")

    assert runner.model_config.model == "meta-llm"


def test_resolve_layerwise_load_device_maps_auto_to_runtime_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    load_config = SimpleNamespace(device="auto")
    device_config = SimpleNamespace(device="cuda")
    resolved = fp8_refit._resolve_layerwise_load_device(load_config, device_config)

    assert resolved == torch.device("cpu")


def test_load_checkpoint_layerwise_accepts_auto_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    call_order: list[object] = []
    monkeypatch.setattr(fp8_refit, "_initialize_layerwise_reload", lambda model: call_order.append("init"))
    monkeypatch.setattr(
        fp8_refit,
        "_finalize_layerwise_reload",
        lambda model, model_config: call_order.append(("finalize", model_config.model)),
    )

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded: list[tuple[str, torch.Tensor]] = []

        def load_weights(self, weights_iter):
            self.loaded = list(weights_iter)
            return "ok"

    model = _DummyModel()
    runner = SimpleNamespace(
        model_config=SimpleNamespace(model="base-model"),
        vllm_config=SimpleNamespace(
            load_config=SimpleNamespace(device="auto"),
            device_config=SimpleNamespace(device="auto"),
            quant_config=None,
        ),
    )

    result = fp8_refit.load_checkpoint_weights_layerwise(runner, model, [("layer.weight", torch.ones(1))])

    assert result == "ok"
    loaded_name, loaded_tensor = model.loaded[0]
    assert loaded_name == "layer.weight"
    assert torch.equal(loaded_tensor, torch.ones(1))
    assert call_order == ["init", ("finalize", "base-model")]
