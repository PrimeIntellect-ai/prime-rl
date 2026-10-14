import json
from pathlib import Path

from pydantic import create_model
from verifiers.v1.serve import EpisodeAssembly

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.dashboard import server
from prime_rl.monitors.file.traces.live import LiveFolds
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.pathing import (
    create_attempt_dirs,
    get_config_dir,
    launcher_attempt_env,
    prepare_attempt_dirs,
    write_resolved_config,
)


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

    write_resolved_config(run_dir, config, "trainer")
    write_resolved_config(run_dir, config, "orchestrator")
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


def test_components_write_configs_into_launcher_attempt(tmp_path, monkeypatch):
    config_dir, log_dir = create_attempt_dirs(tmp_path)
    for key, value in launcher_attempt_env(config_dir, log_dir).items():
        monkeypatch.setenv(key, value)
    config_type = create_model("RunConfig", model=(dict, ...))

    write_resolved_config(tmp_path, config_type(model={"name": "new-model"}), "trainer")

    assert json.loads((config_dir / "trainer.json").read_text()) == {"model": {"name": "new-model"}}
    assert prepare_attempt_dirs(tmp_path) == (config_dir, log_dir)
    assert not (tmp_path / "configs" / "resolved").exists()
    assert get_config_dir(tmp_path) == config_dir


def test_loaded_trainer_config_round_trips_in_launcher_attempt(tmp_path, monkeypatch):
    config_dir, log_dir = create_attempt_dirs(tmp_path)
    for key, value in launcher_attempt_env(config_dir, log_dir).items():
        monkeypatch.setenv(key, value)
    config_path = config_dir / "trainer.json"
    config = TrainerConfig(output_dir=tmp_path, ckpt=None, optim={"max_norm": None})
    original = json.dumps(dump_resolved_config(config), indent=2)
    config_path.write_text(original)
    command_path = config_dir.parent / "command.txt"
    command_path.write_text("uv run rl @ rl.toml\n")
    toml_path = config_dir.parent / "trainer.toml"
    toml_path.write_text("max_steps = 10\n")
    args = ["@", str(config_path)]
    monkeypatch.setattr("sys.argv", ["trainer", *args])

    write_resolved_config(tmp_path, cli(TrainerConfig, args=args), "trainer")

    assert config_path.read_text() == original
    assert command_path.read_text() == "uv run rl @ rl.toml\n"
    assert toml_path.read_text() == "max_steps = 10\n"
    assert not (tmp_path / "configs" / "resolved").exists()


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
