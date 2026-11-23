import sys

import pytest
from vllm.model_executor.models.registry import ModelRegistry, _LazyRegisteredModel

from prime_rl.inference.vllm.models.inkling2 import plugin


def test_plugin_leaves_a_vllm_with_its_own_inkling2_alone(monkeypatch):
    shipped = _LazyRegisteredModel(module_name="vllm.models.inkling2", class_name=plugin.ARCHITECTURE)
    monkeypatch.setitem(ModelRegistry.models, plugin.ARCHITECTURE, shipped)
    # On such a vLLM the port's model module does not import, and nothing of the port may be registered.
    monkeypatch.setitem(sys.modules, "prime_rl.inference.vllm.models.inkling2.model", None)
    monkeypatch.setattr(plugin, "_register_parsers", lambda: pytest.fail("registered the port's parsers"))

    plugin.register()

    assert ModelRegistry.models[plugin.ARCHITECTURE] is shipped
