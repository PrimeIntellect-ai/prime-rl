import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_eval_import_stubs() -> None:
    """Stub out heavy deps so we can import eval script in a minimal env."""
    if "verifiers" not in sys.modules:
        vf_mod = types.ModuleType("verifiers")
        vf_mod.setup_logging = lambda *a, **k: None
        sys.modules["verifiers"] = vf_mod

    # prime_rl.utils.monitor imports wandb/pandas/transformers; avoid those.
    if "prime_rl.utils.monitor" not in sys.modules:
        monitor_mod = types.ModuleType("prime_rl.utils.monitor")
        monitor_mod.setup_monitor = lambda *a, **k: None
        sys.modules["prime_rl.utils.monitor"] = monitor_mod

    # prime_rl.utils.client imports httpx/openai/prime_evals; avoid those.
    if "prime_rl.utils.client" not in sys.modules:
        client_mod = types.ModuleType("prime_rl.utils.client")

        async def _noop_async(*args, **kwargs):
            return None

        client_mod.check_has_model = _noop_async
        client_mod.check_health = _noop_async
        client_mod.reload_weights = _noop_async
        client_mod.setup_admin_clients = lambda *a, **k: ["admin"]
        client_mod.setup_clients = lambda *a, **k: ["client"]
        client_mod.setup_evals_client = lambda *a, **k: object()
        client_mod.update_weights = _noop_async
        sys.modules["prime_rl.utils.client"] = client_mod

    if "prime_rl.orchestrator.utils" not in sys.modules:
        orch_utils_mod = types.ModuleType("prime_rl.orchestrator.utils")

        async def _noop_async(*args, **kwargs):
            return None

        orch_utils_mod.set_semaphore = _noop_async
        sys.modules["prime_rl.orchestrator.utils"] = orch_utils_mod

    if "prime_rl.eval.utils" not in sys.modules:
        eval_utils_mod = types.ModuleType("prime_rl.eval.utils")

        async def _noop_async(*args, **kwargs):
            return None

        eval_utils_mod.run_evals = _noop_async
        sys.modules["prime_rl.eval.utils"] = eval_utils_mod

    if "prime_rl.eval.config" not in sys.modules:
        eval_config_mod = types.ModuleType("prime_rl.eval.config")
        eval_config_mod.OfflineEvalConfig = object
        sys.modules["prime_rl.eval.config"] = eval_config_mod

    if "prime_rl.utils.logger" not in sys.modules:
        logger_mod = types.ModuleType("prime_rl.utils.logger")

        class _Logger:
            def info(self, *a, **k):  # noqa: ANN001
                return None

            def success(self, *a, **k):  # noqa: ANN001
                return None

            def exception(self, *a, **k):  # noqa: ANN001
                return None

            def warning(self, *a, **k):  # noqa: ANN001
                return None

        logger_mod.setup_logger = lambda *a, **k: _Logger()
        sys.modules["prime_rl.utils.logger"] = logger_mod

    if "prime_rl.utils.pydantic_config" not in sys.modules:
        pc_mod = types.ModuleType("prime_rl.utils.pydantic_config")
        pc_mod.parse_argv = lambda *a, **k: None
        sys.modules["prime_rl.utils.pydantic_config"] = pc_mod

    if "prime_rl.utils.utils" not in sys.modules:
        utils_mod = types.ModuleType("prime_rl.utils.utils")

        def clean_exit(fn):  # noqa: ANN001
            return fn

        utils_mod.clean_exit = clean_exit
        utils_mod.get_env_ids_to_install = lambda *a, **k: []

        def get_step_path(weights_dir: Path, step: int) -> Path:
            return weights_dir / f"step_{step}"

        utils_mod.get_step_path = get_step_path
        utils_mod.install_env = lambda *a, **k: None
        sys.modules["prime_rl.utils.utils"] = utils_mod


_install_eval_import_stubs()

# Import after stubbing so test can run in minimal env.
eval_mod = importlib.import_module("prime_rl.eval.eval")


@pytest.mark.asyncio
async def test_eval_continues_on_checkpoint_failure(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for step in (1, 2):
        step_dir = weights_dir / f"step_{step}"
        step_dir.mkdir()
        (step_dir / "STABLE").write_text("ok")

    called_steps: list[int] = []

    async def _update_weights(_admin_clients, step_path):
        # step_path ends with step_{x}
        step = int(str(step_path).split("step_")[-1])
        called_steps.append(step)
        if step == 2:
            raise RuntimeError("boom")

    async def _run_evals(*args, **kwargs):
        return None

    # tests/conftest.py initializes the global logger once; avoid re-initializing it here
    from prime_rl.utils.logger import get_logger

    monkeypatch.setattr(eval_mod, "setup_logger", lambda *a, **k: get_logger())
    monkeypatch.setattr(eval_mod, "update_weights", _update_weights)
    monkeypatch.setattr(eval_mod, "run_evals", _run_evals)

    config = types.SimpleNamespace(
        # logging / monitor
        log=types.SimpleNamespace(level="info", vf_level="warn", file=False),
        wandb=None,
        # client/model/sampling
        client=types.SimpleNamespace(base_url=["http://localhost"], api_key_var="OPENAI_API_KEY", headers={}, timeout=1),
        model=types.SimpleNamespace(name="dummy"),
        sampling=types.SimpleNamespace(),
        reasoning_field="reasoning_content",
        # eval settings
        env=[],
        max_concurrent=None,
        output_dir=tmp_path / "out",
        resume_path=None,
        eval_base=False,
        weights_dir=weights_dir,
        steps=None,
        watcher=False,
        continue_on_ckpt_error=True,
    )

    await eval_mod.eval(config)

    # The eval loop iterates checkpoints in reverse order.
    # Step 2 fails, but we should still attempt step 1 afterwards.
    assert called_steps == [2, 1]

