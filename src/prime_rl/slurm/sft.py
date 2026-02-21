import subprocess
import sys
from pathlib import Path

import tomli_w
from jinja2 import Environment, FileSystemLoader
from pydantic import Field, model_validator

from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pydantic_config import parse_argv

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "sft_slurm.sh.j2"


class SFTSLURMConfig(SFTTrainerConfig):
    job_name: str
    num_nodes: int = Field(default=1, description="Number of training nodes.")
    gpus_per_node: int = Field(default=8, description="Number of GPUs per node.")
    nodes_per_fsdp_group: int | None = Field(
        default=None,
        description="Number of nodes per FSDP island. Auto-sets model.dp_replicate = num_nodes / nodes_per_fsdp_group.",
    )

    project_dir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Path to the project root. Used to source .env, activate .venv, and run uv sync.",
    )
    hf_hub_offline: bool = Field(
        default=False, description="Set HF_HUB_OFFLINE=1 on training nodes to prevent downloading models at runtime."
    )

    slurm_template: Path | None = Field(
        default=None, description="The path to the SLURM template file. If none, will use the default template."
    )
    dry_run: bool = Field(default=False, description="Only generate the SLURM script and configs without submitting.")

    @model_validator(mode="after")
    def auto_setup_dp_replicate(self):
        if self.nodes_per_fsdp_group is not None:
            if self.num_nodes % self.nodes_per_fsdp_group != 0:
                raise ValueError(
                    f"num_nodes ({self.num_nodes}) must be divisible by nodes_per_fsdp_group ({self.nodes_per_fsdp_group})"
                )
            self.model.dp_replicate = self.num_nodes // self.nodes_per_fsdp_group
        return self


def write_trainer_config(config: SFTSLURMConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    slurm_only_fields = set(SFTSLURMConfig.model_fields) - set(SFTTrainerConfig.model_fields)
    trainer_data = config.model_dump(exclude_none=True, mode="json", exclude=slurm_only_fields)
    with open(output_dir / "trainer.toml", "wb") as f:
        tomli_w.dump(trainer_data, f)


def render_slurm_script(config: SFTSLURMConfig, config_dir: Path) -> str:
    if config.slurm_template is not None:
        template_dir = config.slurm_template.parent
        template_name = config.slurm_template.name
    else:
        template_dir = TEMPLATE_DIR
        template_name = TEMPLATE_NAME
    env = Environment(loader=FileSystemLoader(template_dir), keep_trailing_newline=True)
    template = env.get_template(template_name)
    return template.render(
        job_name=config.job_name,
        output_dir=config.output_dir,
        config_dir=config_dir,
        num_nodes=config.num_nodes,
        project_dir=config.project_dir,
        gpus_per_node=config.gpus_per_node,
        hf_hub_offline=config.hf_hub_offline,
    )


def sft_slurm(config: SFTSLURMConfig):
    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = config.output_dir / "configs"
    write_trainer_config(config, config_dir)

    logger.info(f"Wrote trainer config to {config_dir}")

    script = render_slurm_script(config, config_dir)
    script_path = config.output_dir / "sft.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_message = f"Logs:\n  Trainer:  tail -f {config.output_dir}/slurm/latest_train_node_rank_0.log"

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def main():
    sft_slurm(parse_argv(SFTSLURMConfig))


if __name__ == "__main__":
    main()
