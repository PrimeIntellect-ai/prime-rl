import re

from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.entrypoints.pretrain import pretrain
from prime_rl.utils.config import cli


def test_multi_node_dry_run_writes_trainer_sbatch(tmp_path, monkeypatch):
    monkeypatch.setenv("PRL_RUN_ID", "test")
    monkeypatch.setenv("PRL_RUN_NAME", "test")
    config = PretrainConfig.model_validate(
        {
            "output_dir": tmp_path,
            "model": {"name": "configs/debug/fake/pretrain-model"},
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
