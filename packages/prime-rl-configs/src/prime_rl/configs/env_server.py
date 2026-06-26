from pathlib import Path

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig


class EnvServerConfig(BaseConfig):
    env: EnvConfig

    log: LogConfig = LogConfig()

    output_dir: Path = Path("outputs")
    """Directory to write outputs to — logs and any generated artifacts are written as subdirectories."""
