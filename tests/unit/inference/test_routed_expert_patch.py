import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("model_config", [SimpleNamespace(enable_return_routed_experts=True), SimpleNamespace()])
@pytest.mark.parametrize("kv_transfer_config", [None, SimpleNamespace(kv_connector="OtherConnector")])
def test_non_nixl_config_preserves_original_post_init(monkeypatch, model_config, kv_transfer_config):
    calls = []

    class VllmConfig:
        def __post_init__(self):
            calls.append(self)

    modules = {
        "torch": ModuleType("torch"),
        "vllm": ModuleType("vllm"),
        "vllm.config": ModuleType("vllm.config"),
        "vllm.config.vllm": ModuleType("vllm.config.vllm"),
        "vllm.logger": ModuleType("vllm.logger"),
    }
    modules["vllm"].envs = SimpleNamespace()
    modules["vllm.config.vllm"].VllmConfig = VllmConfig
    modules["vllm.logger"].init_logger = lambda _: SimpleNamespace(warning=lambda _: None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[3] / "src/prime_rl/inference/patches.py"
    spec = importlib.util.spec_from_file_location("isolated_inference_patches", path)
    patches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patches)
    patches.monkey_patch_return_routed_experts_with_nixl_connector()
    config = VllmConfig()
    config.model_config = model_config
    config.kv_transfer_config = kv_transfer_config
    config.__post_init__()
    assert calls == [config]
