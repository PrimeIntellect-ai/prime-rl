import pytest
import torch

from prime_rl.inference.vllm.worker.weight_transfer import check_fp8_source


def test_fp8_params_only_take_fp8_tensors():
    check_fp8_source("w", torch.float8_e4m3fn, torch.float8_e4m3fn)
    check_fp8_source("w", torch.bfloat16, torch.bfloat16)
    with pytest.raises(ValueError, match="fp8_per_block"):
        check_fp8_source("w", torch.float8_e4m3fn, torch.bfloat16)


def test_rejected_reload_still_finalizes(monkeypatch):
    # A bf16 tensor for an fp8 param raises mid-load; the layers must not stay on meta
    from prime_rl.inference.vllm.worker import weight_transfer

    calls = []
    monkeypatch.setattr(weight_transfer, "initialize_layerwise_reload", lambda model: calls.append("init"))
    monkeypatch.setattr(weight_transfer, "_guard_fp8_params", lambda model: None)
    monkeypatch.setattr(
        weight_transfer, "set_current_vllm_config", lambda config: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(weight_transfer, "finalize_layerwise_reload", lambda model, config: calls.append("finalize"))

    class Model(torch.nn.Linear):
        def load_weights(self, weights):
            raise ValueError("bf16 into fp8")

    with pytest.raises(ValueError, match="bf16 into fp8"):
        weight_transfer.load_weights_checkpoint_layerwise(Model(2, 2), [], None, None)
    assert calls == ["init", "finalize"]
