import pytest
import torch

from prime_rl.inference.vllm.worker import filesystem, nccl


class _DummyModelLoader:
    def __init__(self):
        self.calls: list[tuple[object, object]] = []

    def get_all_weights(self, model_config: object, model: object):
        self.calls.append((model_config, model))
        return [("layer.weight", torch.ones((1,), dtype=torch.float32))]


class _Runner:
    def __init__(self):
        self.load_config = type("LoadConfig", (), {"load_format": "auto"})()
        self.model_config = type("ModelConfig", (), {"model": "old-model"})()

    def get_model(self):
        return object()


def _capture_layerwise_load_calls(calls: dict[str, object]):
    def _capture_layerwise_load(
        model_runner: object,
        inner_model: object,
        weights_iter: object,
    ) -> None:
        calls["model_runner"] = model_runner
        calls["model"] = inner_model
        calls["weights"] = list(weights_iter)

    return _capture_layerwise_load


def _assert_single_weight_payload(weights: object) -> None:
    assert isinstance(weights, list)
    assert [name for name, _ in weights] == ["layer.weight"]
    assert torch.equal(weights[0][1], torch.ones((1,), dtype=torch.float32))


def test_filesystem_worker_uses_layerwise_reload_path(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _DummyModelLoader()
    runner = _Runner()
    model = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(filesystem, "get_model_loader", lambda load_config: loader)
    monkeypatch.setattr(filesystem, "unwrap_worker_model", lambda wrapped: model)
    monkeypatch.setattr(filesystem, "load_checkpoint_weights_layerwise", _capture_layerwise_load_calls(calls))

    worker = filesystem.FileSystemWeightUpdateWorker()
    worker.model_runner = runner

    worker.update_weights_from_path("/tmp/new-checkpoint")

    assert runner.model_config.model == "/tmp/new-checkpoint"
    assert loader.calls == [(runner.model_config, model)]
    assert calls["model_runner"] is runner
    assert calls["model"] is model
    _assert_single_weight_payload(calls["weights"])


def test_filesystem_worker_requires_loader_with_get_all_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _Runner()
    model = object()

    monkeypatch.setattr(filesystem, "get_model_loader", lambda load_config: object())
    monkeypatch.setattr(filesystem, "unwrap_worker_model", lambda wrapped: model)

    worker = filesystem.FileSystemWeightUpdateWorker()
    worker.model_runner = runner

    with pytest.raises(NotImplementedError, match="Model reloading"):
        worker.update_weights_from_path("/tmp/new-checkpoint")


class _DummyNCCLReceiver:
    def receive_state_dict(self):
        return iter([("layer.weight", torch.ones((1,), dtype=torch.float32))])


def test_nccl_worker_uses_layerwise_reload_path(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _Runner()
    model = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(nccl, "unwrap_worker_model", lambda wrapped: model)
    monkeypatch.setattr(nccl, "load_checkpoint_weights_layerwise", _capture_layerwise_load_calls(calls))

    worker = nccl.NCCLWeightUpdateWorker()
    worker.model_runner = runner
    worker.nccl_broadcast_receiver = _DummyNCCLReceiver()

    worker.update_weights_from_path("/unused-in-nccl-path")

    assert calls["model_runner"] is runner
    assert calls["model"] is model
    _assert_single_weight_payload(calls["weights"])
