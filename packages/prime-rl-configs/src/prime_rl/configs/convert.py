from pathlib import Path

from prime_rl.utils.config import BaseConfig


class ConvertConfig(BaseConfig):
    model: str
    """HF repo id or local snapshot path whose weights are converted to the
    PrimeRL grouped format under `<snapshot>/prime/`."""

    conversion_dir: Path | None = None
    """Directory that receives the `prime` subdirectory. Defaults to the model snapshot directory."""
