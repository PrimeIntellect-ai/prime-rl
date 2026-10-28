"""The D5 harness plugin (``inrun_eval/plugins/tb4_mini_swe_agent_d5.py``) against the real verifiers 24107a4ea.

Runs only where that verifiers is installed (the eval's harness venv, e.g. one built from the bundle's
requirements); prime-rl's own environment has another verifiers and skips it. Checks that verifiers resolves the
harness id to the plugin and its config, that ``UV_EXCLUDE_NEWER`` reaches only the uv-script preparation, never the
program's command environment, and that the program (so the venv path) is the unmodified one.
"""

import asyncio
import hashlib
import importlib.metadata
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGINS = Path(__file__).resolve().parents[4] / "src" / "prime_rl" / "trainer" / "sft" / "inrun_eval" / "plugins"
PROGRAM_SHA256 = "1467b440577f1a2fee1f628abdb3bd60b8375a2116b2e193ab97c752c14d0297"
EXCLUDE = "2026-10-24T07:23:00Z"

try:
    VERIFIERS = importlib.metadata.version("verifiers")
except importlib.metadata.PackageNotFoundError:
    VERIFIERS = None
pytestmark = pytest.mark.skipif(VERIFIERS != "0.4.1.dev340", reason="needs verifiers 24107a4ea (the harness venv)")


@pytest.fixture
def plugin():
    sys.path.insert(0, str(PLUGINS))
    try:
        from verifiers.v1.utils.loaders import harness_class, harness_config_type

        yield harness_class("tb4-mini-swe-agent-d5"), harness_config_type("tb4-mini-swe-agent-d5")
    finally:
        sys.path.remove(str(PLUGINS))


class FakeRuntime:
    """Records what the harness asks of the box."""

    def __init__(self, user=None):
        self.user = user
        self.prepared: list[dict] = []
        self.runs: list[tuple[list[str], dict]] = []
        self.writes: list[str] = []
        self.backend = SimpleNamespace(cache=SimpleNamespace(curl_ready=False))

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        self.prepared.append({"sha256": hashlib.sha256(script.encode()).hexdigest(), "env": dict(env or {})})
        return ["sh", "-c", "exec", "uv-script", "/venv", "/venv/bin/python", "/tmp/vf-scripts/x.py"]

    async def write(self, path, data):
        self.writes.append(path)

    async def run(self, cmd, env):
        self.runs.append((list(cmd), dict(env)))
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    def with_user(self, user):
        return self


def test_resolves_to_the_plugin_with_its_config(plugin):
    cls, config_type = plugin
    from verifiers.v1.harnesses.mini_swe_agent import MiniSWEAgentHarness

    assert issubclass(cls, MiniSWEAgentHarness) and config_type.__name__ == "D5MiniSWEAgentHarnessConfig"
    config = config_type.model_validate(
        {
            "id": "tb4-mini-swe-agent-d5",
            "command_timeout": 1800,
            "prepare_exclude_newer": EXCLUDE,
            "prepare_uv_version": "0.13.5",
        }
    )
    assert config.version == "2.4.6" and config.command_timeout == 1800 and config.dialect == "chat_completions"
    assert config.prepare_env == {"UV_EXCLUDE_NEWER": EXCLUDE}


def test_exclude_newer_reaches_only_the_preparation(plugin):
    cls, config_type = plugin
    from verifiers.v1.harnesses.mini_swe_agent import harness as mini
    from verifiers.v1.task import TaskData

    assert hashlib.sha256((Path(mini.__file__).parent / "program.py").read_bytes()).hexdigest() == PROGRAM_SHA256
    config = config_type.model_validate(
        {
            "id": "tb4-mini-swe-agent-d5",
            "command_timeout": 1800,
            "prepare_exclude_newer": EXCLUDE,
            "prepare_uv_version": "0.13.5",
            "env": {"TASK_VAR": "1"},
        }
    )
    harness = cls(config)
    runtime = FakeRuntime()
    asyncio.run(harness.setup(runtime))
    ctx = SimpleNamespace(model="nemotron-3.5-super-tb4-sft")
    trace = SimpleNamespace(id="trace1")
    result = asyncio.run(
        harness.launch(ctx, trace, runtime, "http://endpoint/v1", "secret", {}, TaskData(prompt="do it"))
    )
    assert result.exit_code == 0

    # setup + launch prepare the unmodified program, both times with the D5 variable on top of the harness env.
    expected_script = mini.PROGRAM_SOURCE.replace("{version}", "2.4.6")
    assert [p["sha256"] for p in runtime.prepared] == [hashlib.sha256(expected_script.encode()).hexdigest()] * 2
    assert all(p["env"] == {"TASK_VAR": "1", "UV_EXCLUDE_NEWER": EXCLUDE} for p in runtime.prepared)
    # The uv pin ran first, with the harness env only; the program runs with exactly the collection's env.
    pin_cmd, pin_env = runtime.runs[0]
    assert "uv==0.13.5" in pin_cmd[2] and "https://astral.sh/uv/0.13.5/install.sh" in pin_cmd[2]
    assert pin_env == {"TASK_VAR": "1"}
    program_cmd, program_env = runtime.runs[-1]
    assert program_cmd[-1] == ".vf-mini-swe-agent-args-trace1.json"
    assert program_env == {"TASK_VAR": "1", "MSWEA_CONFIGURED": "true", "MSWEA_SILENT_STARTUP": "true"}
    assert not any("UV_EXCLUDE_NEWER" in env for _, env in runtime.runs)


def test_without_d5_settings_it_is_the_collection_harness(plugin):
    cls, config_type = plugin
    harness = cls(config_type.model_validate({"id": "tb4-mini-swe-agent-d5", "command_timeout": 1800}))
    runtime = FakeRuntime()
    asyncio.run(harness.setup(runtime))
    assert runtime.runs == [] and runtime.prepared[0]["env"] == {}


def test_user_box_installs_a_downloader_first(plugin):
    cls, config_type = plugin
    harness = cls(config_type.model_validate({"id": "x", "prepare_uv_version": "0.13.5"}))
    runtime = FakeRuntime(user="agent")
    asyncio.run(harness.setup(runtime))
    assert "apt-get install" in runtime.runs[0][0][2] and runtime.backend.cache.curl_ready is True
    assert "uv==0.13.5" in runtime.runs[1][0][2]
