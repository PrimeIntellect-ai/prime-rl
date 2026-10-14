import re

from prime_rl.configs.pretrain import EvalWorkerConfig, PretrainConfig
from prime_rl.entrypoints.pretrain import pretrain
from prime_rl.utils.config import cli


def test_multi_node_dry_run_writes_trainer_sbatch(tmp_path, monkeypatch):
    monkeypatch.setenv("PRL_RUN_ID", "test")
    monkeypatch.setenv("PRL_RUN_NAME", "test")
    config = PretrainConfig.model_validate(
        {
            "output_dir": tmp_path,
            "model": {"arch": {"model_type": "llama"}},
            "tokenizer": {"name": "PrimeIntellect/Qwen3-0.6B"},
            "data": {"type": "fake", "seq_len": 8, "batch_tokens": 16, "micro_batch_tokens": 8},
            "optim": {"lr": 1e-3},
            "max_steps": 1,
            "deployment": {"type": "multi_node", "num_train_nodes": 2},
            "slurm": {},
            "dry_run": True,
        }
    )
    pretrain(config)

    script = (config.run_dir / "launcher" / "pretrain.sbatch").read_text()
    assert "#SBATCH --nodes=2" in script
    assert "-m prime_rl.trainer.pretrain.train" in script
    # The trainers on every node parse the written config, which has no [deployment] or [slurm].
    config_path = re.search(r"^export TRAIN_CONFIG_PATH=(.+)$", script, re.M).group(1)
    trainer_config = cli(PretrainConfig, args=["@", config_path])
    assert trainer_config.slurm is None
    assert trainer_config.model == config.model


def test_multi_node_dry_run_adds_the_eval_worker_node(tmp_path, monkeypatch):
    monkeypatch.setenv("PRL_RUN_ID", "test")
    monkeypatch.setenv("PRL_RUN_NAME", "test")
    config = PretrainConfig.model_validate(
        {
            "output_dir": tmp_path,
            "model": {"arch": {"model_type": "llama"}},
            "tokenizer": {"name": "PrimeIntellect/Qwen3-0.6B"},
            "data": {"type": "fake", "seq_len": 8, "batch_tokens": 16, "micro_batch_tokens": 8},
            "optim": {"lr": 1e-3},
            "max_steps": 1,
            "eval_snapshots": {"interval": 1},
            "eval_worker": {"data_dir": tmp_path, "sets": ["holdout"], "seq_lens": [2048, 4096]},
            "deployment": {"type": "multi_node", "num_train_nodes": 2},
            "slurm": {},
            "dry_run": True,
        }
    )
    pretrain(config)

    script = (config.run_dir / "launcher" / "pretrain.sbatch").read_text()
    assert "#SBATCH --nodes=3" in script
    worker_config_path = re.search(r"-m prime_rl.trainer.pretrain.eval_worker \\\n\s+@ (\S+)", script).group(1)
    worker_config = cli(EvalWorkerConfig, args=["@", worker_config_path])
    assert worker_config.run_dir == config.run_dir
    assert worker_config.seq_lens == [2048, 4096]
