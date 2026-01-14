from pathlib import Path

import pytest

import prime_rl.eval.eval as eval_mod
from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.utils.config import ClientConfig, LogConfig, ModelConfig
from prime_rl.utils.logger import get_logger


@pytest.mark.anyio
async def test_eval_continues_on_checkpoint_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for step in (1, 2):
        step_dir = weights_dir / f"step_{step}"
        step_dir.mkdir()
        (step_dir / "STABLE").write_text("ok")

    called_steps: list[int] = []

    async def _noop_async(*args, **kwargs):  # noqa: ANN001
        return None

    async def _update_weights(_admin_clients, step_path: Path) -> None:
        step = int(step_path.name.split("_")[-1])
        called_steps.append(step)
        if step == 2:
            raise RuntimeError("boom")

    async def _run_evals(*args, **kwargs):  # noqa: ANN001
        return None

    # Avoid initializing global singletons in the eval script during unit tests.
    monkeypatch.setattr(eval_mod, "setup_logger", lambda *a, **k: get_logger())
    monkeypatch.setattr(eval_mod.vf, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(eval_mod, "setup_monitor", lambda *a, **k: None)
    monkeypatch.setattr(eval_mod, "get_env_ids_to_install", lambda *_a, **_k: [])
    monkeypatch.setattr(eval_mod, "install_env", lambda *_a, **_k: None)

    monkeypatch.setattr(eval_mod, "setup_clients", lambda *_a, **_k: ["client"])
    monkeypatch.setattr(eval_mod, "setup_admin_clients", lambda *_a, **_k: ["admin"])
    monkeypatch.setattr(eval_mod, "setup_evals_client", lambda *_a, **_k: object())

    monkeypatch.setattr(eval_mod, "check_health", _noop_async)
    monkeypatch.setattr(eval_mod, "check_has_model", _noop_async)
    monkeypatch.setattr(eval_mod, "reload_weights", _noop_async)

    monkeypatch.setattr(eval_mod, "update_weights", _update_weights)
    monkeypatch.setattr(eval_mod, "run_evals", _run_evals)

    config = OfflineEvalConfig(
        output_dir=tmp_path / "out",
        weights_dir=weights_dir,
        eval_base=False,
        watcher=False,
        continue_on_ckpt_error=True,
        wandb=None,
        log=LogConfig(level="info", vf_level="warn", file=False),
        client=ClientConfig(base_url=["http://localhost"], timeout=1),
        model=ModelConfig(name="dummy"),
    )

    await eval_mod.eval(config)

    # The eval loop iterates checkpoints in reverse order.
    # Step 2 fails, but we should still attempt step 1 afterwards.
    assert called_steps == [2, 1]

