from pathlib import Path

from pydantic import model_validator

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig


class EnvServerConfig(BaseConfig):
    env: EnvConfig = EnvConfig()
    """Environment configuration."""

    log: LogConfig = LogConfig()
    """Logger configuration."""

    env_install_prerelease: bool = False
    """Allow pre-release versions when installing environments. Passes ``--prerelease`` to ``prime env install``."""

    output_dir: Path = Path("outputs")
    """Directory to write outputs to — logs and any generated artifacts are written as subdirectories."""

    @model_validator(mode="after")
    def validate_num_workers(self):
        if self.env.num_workers == "auto":
            self.env.num_workers = 1
        return self
