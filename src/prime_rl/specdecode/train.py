import json
import subprocess
import sys
import warnings
from copy import deepcopy

from prime_rl.configs.specdecode import SpecDecodeConfig
from prime_rl.utils.config import cli


def train(config: SpecDecodeConfig) -> None:
    from speculators.train.config import TrainConfig

    config.output_dir.mkdir(parents=True, exist_ok=True)
    training = deepcopy(config.train)
    training.setdefault("trainer", {}).setdefault("save_path", str(config.output_dir / "checkpoints"))
    config_path = config.output_dir / "speculators.json"
    config_path.write_text(json.dumps({"train": training}, indent=2) + "\n")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*has unrecognised keys")
        upstream = TrainConfig.from_sources(cli={}, config_path=str(config_path), argv=sys.argv[1:])
    (config.output_dir / "specdecode.json").write_text(config.model_dump_json(indent=2) + "\n")
    (config.output_dir / "resolved.yaml").write_text(upstream.dump_yaml())
    if config.dry_run:
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={config.num_gpus}",
            "-m",
            "prime_rl.specdecode.worker",
            "--config",
            str(config_path),
        ],
        check=True,
    )


def main() -> None:
    train(cli(SpecDecodeConfig))


if __name__ == "__main__":
    main()
