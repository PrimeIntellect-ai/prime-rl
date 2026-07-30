from pathlib import Path

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig


class EnvServerConfig(BaseConfig):
    env: EnvConfig

    address: str
    """ZMQ address the server binds and clients connect to (e.g. ``tcp://127.0.0.1:5000``). Required — the ``rl`` launcher writes each source's deterministic address here; pass it explicitly when running standalone."""

    log: LogConfig = LogConfig()

    output_dir: Path = Path("outputs")
    """Directory to write outputs to — logs and any generated artifacts are written as subdirectories."""
