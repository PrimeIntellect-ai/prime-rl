import json
from pathlib import Path

from pydantic import create_model
from verifiers.v1.serve import EpisodeAssembly

from prime_rl.dashboard import server
from prime_rl.monitors.file.traces.live import LiveFolds
from prime_rl.utils.pathing import create_attempt_dirs, get_config_dir, write_standalone_config


def test_standalone_components_populate_dashboard(tmp_path, monkeypatch):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    run_dir = tmp_path / "standalone"
    config_type = create_model(
        "RunConfig",
        model=(dict, ...),
        max_steps=(int, ...),
        train=(dict, ...),
        eval=(dict | None, None),
        ckpt=(dict | None, None),
    )
    config = config_type(
        model={"name": "test-model"},
        max_steps=50,
        train={
            "source": [
                {"name": "training-env", "env": {"taskset": {"id": "taskset"}}},
                {"env": {"id": "paired-env", "taskset": {"id": "taskset"}}},
            ]
        },
        eval={"source": [{"env": {"taskset": {"id": "eval-env"}}}]},
    )
    toml_path = tmp_path / "orchestrator.toml"
    toml_path.write_text('max_steps = 10\n[model]\nname = "original-model"\n')
    monkeypatch.setattr("sys.argv", ["orchestrator", "@", str(toml_path), "--max-steps", "50"])

    write_standalone_config(run_dir, config, "trainer")
    write_standalone_config(run_dir, config, "orchestrator")
    monkeypatch.setitem(server._run_registry, "standalone", run_dir)

    metadata = server.run_meta(run_dir)
    assert metadata["type"] == "rl"
    assert metadata["model"] == "test-model"
    assert metadata["max_steps"] == 50
    assert metadata["train_envs"] == ["paired-env+taskset", "training-env"]
    assert metadata["eval_envs"] == ["eval-env"]
    assert server.list_configs("standalone")["files"] == [
        "orchestrator.toml",
        "trainer.toml",
        "resolved",
        "command.txt",
    ]
    config_dir = get_config_dir(run_dir)
    assert json.loads((config_dir / "trainer.json").read_text())["eval"] == config.eval
    assert json.loads((config_dir / "trainer.json").read_text())["ckpt"] is None
    assert (config_dir.parent / "orchestrator.toml").read_text() == toml_path.read_text()
    assert "--max-steps 50" in (config_dir.parent / "command.txt").read_text()


def test_launcher_configs_are_preserved(tmp_path, monkeypatch):
    config_dir, log_dir = create_attempt_dirs(tmp_path)
    monkeypatch.setenv("PRL_ATTEMPT_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("PRL_ATTEMPT_LOG_DIR", str(log_dir))
    original = '{"model": {"name": "launcher-model"}}'
    (config_dir / "trainer.json").write_text(original)
    config_type = create_model("RunConfig", model=(dict, ...))
    write_standalone_config(tmp_path, config_type(model={"name": "new-model"}), "trainer")
    assert (config_dir / "trainer.json").read_text() == original
    assert not (tmp_path / "configs" / "resolved").exists()
    assert get_config_dir(tmp_path) == config_dir


def test_live_trace_removed_between_stat_and_open(tmp_path, monkeypatch):
    path = tmp_path / "live.jsonl"
    path.write_bytes(b"{}\n")
    folds = LiveFolds()
    folds._folds[path] = (0, {}, EpisodeAssembly())
    original_open = Path.open

    def remove_before_open(self, *args, **kwargs):
        if self == path:
            self.unlink()
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", remove_before_open)
    assert folds.read(path) is None
    assert path not in folds._folds
