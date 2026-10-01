import json
import textwrap

import pytest

from prime_rl.inference.patches import load_vllm_plugins

PLUGIN_SOURCE = textwrap.dedent(
    """
    calls = []


    def register():
        calls.append("registered")
    """
)


@pytest.fixture
def plugin_file(tmp_path):
    path = tmp_path / "vllm_plugin.py"
    path.write_text(PLUGIN_SOURCE)
    return path


def test_load_vllm_plugins_runs_each_target(monkeypatch, plugin_file):
    from renderers.plugins import load_plugin_module

    monkeypatch.setenv("PRIME_VLLM_PLUGINS", json.dumps([f"{plugin_file}:register"]))

    load_vllm_plugins()

    assert load_plugin_module(str(plugin_file)).calls == ["registered"]


def test_load_vllm_plugins_is_a_noop_without_targets(monkeypatch):
    monkeypatch.delenv("PRIME_VLLM_PLUGINS", raising=False)
    load_vllm_plugins()

    monkeypatch.setenv("PRIME_VLLM_PLUGINS", "[]")
    load_vllm_plugins()


def test_load_vllm_plugins_reports_missing_attribute(monkeypatch, plugin_file):
    monkeypatch.setenv("PRIME_VLLM_PLUGINS", json.dumps([f"{plugin_file}:missing"]))

    with pytest.raises(ValueError, match="missing"):
        load_vllm_plugins()
