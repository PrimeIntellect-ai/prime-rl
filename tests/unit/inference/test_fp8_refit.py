import pytest
import torch

from prime_rl.inference.vllm.worker import fp8_refit
from prime_rl.inference.vllm.worker.fp8_refit import (
    load_checkpoint_weights_layerwise,
    maybe_convert_weights_for_fp8_refit,
    reset_fp8_process_flags,
)


class Fp8Config:
    def __init__(self, weight_block_size: tuple[int, int] | None):
        self.weight_block_size = weight_block_size


class _DummyVllmConfig:
    def __init__(self, weight_block_size: tuple[int, int] | None):
        self.quant_config = Fp8Config(weight_block_size)


class _DummyModelRunner:
    def __init__(self, weight_block_size: tuple[int, int] | None):
        self.vllm_config = _DummyVllmConfig(weight_block_size)


class _DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.zeros((4, 4), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "weight_scale_inv",
            torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32), requires_grad=False),
        )
        self._already_called_process_weights_after_loading = True


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _DummyLayer()
        self.register_parameter(
            "plain_weight",
            torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float32), requires_grad=False),
        )


class _PackedScaleModel(torch.nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self):
        super().__init__()
        self._qkv_scale_inv = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32), requires_grad=False)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield "layer.qkv_proj.weight_scale_inv", self._qkv_scale_inv


def _fake_quantize_weight_to_block_fp8(
    tensor: torch.Tensor,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    tiles_m = (tensor.shape[0] + block_size[0] - 1) // block_size[0]
    tiles_n = (tensor.shape[1] + block_size[1] - 1) // block_size[1]
    return torch.zeros_like(tensor, dtype=torch.float8_e4m3fn), torch.ones((tiles_m, tiles_n), dtype=torch.float32)


@pytest.mark.parametrize("include_incoming_scale", [False, True], ids=["without_scale", "with_scale"])
def test_fp8_refit_quantizes_weight_and_emits_single_scale(
    monkeypatch: pytest.MonkeyPatch,
    include_incoming_scale: bool,
) -> None:
    model = _DummyModel()
    runner = _DummyModelRunner((2, 2))
    incoming: list[tuple[str, torch.Tensor]] = [("layer.weight", torch.ones((4, 4), dtype=torch.bfloat16))]
    if include_incoming_scale:
        incoming.append(("layer.weight_scale_inv", torch.ones((2, 2), dtype=torch.float32)))
    monkeypatch.setattr(fp8_refit, "quantize_weight_to_block_fp8", _fake_quantize_weight_to_block_fp8)

    converted = list(maybe_convert_weights_for_fp8_refit(runner, model, incoming))
    assert [name for name, _ in converted] == ["layer.weight", "layer.weight_scale_inv"]
    assert converted[0][1].dtype == torch.float8_e4m3fn
    assert converted[1][1].dtype == torch.float32
    assert converted[1][1].shape == (2, 2)


def test_fp8_refit_skips_manual_quant_for_packed_module_weight_names() -> None:
    model = _PackedScaleModel()
    runner = _DummyModelRunner((2, 2))
    incoming = [("layer.q_proj.weight", torch.randn((4, 4), dtype=torch.bfloat16))]

    converted = list(maybe_convert_weights_for_fp8_refit(runner, model, incoming))
    converted_names = [name for name, _ in converted]
    assert converted_names == ["layer.q_proj.weight"]
    assert converted[0][1].dtype == torch.bfloat16


def test_fp8_refit_noop_when_runner_not_fp8() -> None:
    model = _DummyModel()
    runner = _DummyModelRunner(None)
    incoming = [("layer.weight", torch.randn((4, 4), dtype=torch.bfloat16))]

    converted = list(maybe_convert_weights_for_fp8_refit(runner, model, incoming))
    assert converted[0][0] == "layer.weight"
    assert converted[0][1].dtype == torch.bfloat16


def test_quantize_weight_uses_triton_only(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_qweight = torch.zeros((2, 2), dtype=torch.float8_e4m3fn)
    expected_scale = torch.ones((1, 1), dtype=torch.float32)
    calls: list[tuple[torch.dtype, str, tuple[int, int]]] = []

    monkeypatch.setattr(fp8_refit, "_get_triton_quantization_device", lambda weight: torch.device("cpu"))
    monkeypatch.setattr(fp8_refit, "_require_hopper_cuda_device", lambda device: None)

    def _cast(weight: torch.Tensor, block_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((weight.dtype, weight.device.type, block_size))
        return expected_qweight, expected_scale

    monkeypatch.setattr(
        fp8_refit,
        "_get_triton_blockwise_cast_to_fp8",
        lambda: _cast,
    )

    qweight, scale = fp8_refit.quantize_weight_to_block_fp8(torch.randn((2, 2), dtype=torch.bfloat16), (2, 2))

    assert qweight is expected_qweight
    assert scale is expected_scale
    assert calls == [(torch.bfloat16, "cpu", (2, 2))]


def test_require_hopper_cuda_device_rejects_non_cuda() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fp8_refit._require_hopper_cuda_device(torch.device("cpu"))


def test_require_hopper_cuda_device_uses_compute_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    fp8_refit._require_hopper_cuda_device(torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(RuntimeError, match="Hopper GPUs"):
        fp8_refit._require_hopper_cuda_device(torch.device("cuda:0"))


def test_reset_fp8_process_flags_clears_marker() -> None:
    model = _DummyModel()
    assert hasattr(model.layer, "_already_called_process_weights_after_loading")
    reset_fp8_process_flags(model)
    assert not hasattr(model.layer, "_already_called_process_weights_after_loading")


class _CaptureLoadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights):
        self.loaded = list(weights)
        return {name for name, _ in self.loaded}


class _DummyRunnerWithModelConfig:
    def __init__(self):
        self.model_config = object()
        self.vllm_config = type(
            "DummyVllmConfig",
            (),
            {
                "load_config": type("DummyLoadConfig", (), {"device": None})(),
                "device_config": type("DummyDeviceConfig", (), {"device": "cpu"})(),
            },
        )()


def test_layerwise_load_wraps_model_load_with_init_and_finalize(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(fp8_refit, "_initialize_layerwise_reload", lambda model: events.append("init"))
    monkeypatch.setattr(
        fp8_refit, "_finalize_layerwise_reload", lambda model, model_config: events.append("finalize")
    )

    model = _CaptureLoadModel()
    runner = _DummyRunnerWithModelConfig()
    loaded = load_checkpoint_weights_layerwise(
        runner, model, [("layer.weight", torch.ones((1,), dtype=torch.float32))]
    )

    assert loaded == {"layer.weight"}
    assert events == ["init", "finalize"]
    assert [name for name, _ in model.loaded] == ["layer.weight"]


def test_layerwise_load_applies_fp8_conversion_hook(monkeypatch) -> None:
    monkeypatch.setattr(fp8_refit, "_initialize_layerwise_reload", lambda model: None)
    monkeypatch.setattr(fp8_refit, "_finalize_layerwise_reload", lambda model, model_config: None)
    monkeypatch.setattr(
        fp8_refit,
        "maybe_convert_weights_for_fp8_refit",
        lambda model_runner, model, weights: [("converted.weight", torch.zeros((1,), dtype=torch.float32))],
    )

    model = _CaptureLoadModel()
    runner = _DummyRunnerWithModelConfig()
    load_checkpoint_weights_layerwise(
        runner,
        model,
        [("raw.weight", torch.ones((1,), dtype=torch.bfloat16))],
    )

    assert [name for name, _ in model.loaded] == ["converted.weight"]


def test_layerwise_load_requires_model_config() -> None:
    with pytest.raises(AttributeError, match="model_runner.model_config is required"):
        load_checkpoint_weights_layerwise(
            object(),
            _CaptureLoadModel(),
            [("layer.weight", torch.ones((1,), dtype=torch.float32))],
        )


def test_layerwise_load_requires_vllm_config() -> None:
    runner = type("Runner", (), {"model_config": object()})()
    with pytest.raises(AttributeError, match="model_runner.vllm_config is required"):
        load_checkpoint_weights_layerwise(
            runner,
            _CaptureLoadModel(),
            [("layer.weight", torch.ones((1,), dtype=torch.float32))],
        )
