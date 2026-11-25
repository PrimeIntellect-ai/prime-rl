import contextlib

import pytest
import torch
from torch import nn
from vllm import ModelRegistry

from prime_rl.inference import patches
from prime_rl.inference.vllm.worker import weight_transfer

ARCHITECTURE = "Inkling2LoopForConditionalGeneration"


def test_looped_inkling2_is_registered_only_on_a_vllm_that_has_the_looped_model(monkeypatch):
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    ModelRegistry.models.pop(ARCHITECTURE, None)

    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: True)
    patches.register_inkling2_loop_for_rl()
    assert ARCHITECTURE not in ModelRegistry.models

    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: False)
    patches.register_inkling2_loop_for_rl()
    entry = ModelRegistry.models[ARCHITECTURE]
    assert entry.module_name == "prime_rl.inference.vllm.models.inkling2_loop.model"
    assert entry.class_name == "Inkling2LoopForRL"


def test_a_model_that_loads_weights_in_place_gets_them_through_its_own_loader(monkeypatch):
    class InPlace(nn.Module):
        loads_weights_in_place = True

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))
            self.loaded: list[str] = []

        def load_weights(self, weights) -> None:
            self.loaded = [name for name, _ in weights]

    monkeypatch.setattr(weight_transfer, "set_current_vllm_config", contextlib.nullcontext)
    # vLLM's layerwise reload re-runs the post-load hooks, which rebind tensors that captured CUDA graphs hold.
    for hook in ("initialize_layerwise_reload", "finalize_layerwise_reload"):
        monkeypatch.setattr(weight_transfer, hook, lambda *args: pytest.fail("ran vLLM's layerwise reload"))
    model = InPlace()

    weight_transfer.load_weights_checkpoint_layerwise(model, iter([("weight", torch.ones(2))]), None, None)

    assert model.loaded == ["weight"]
