from prime_rl.utils.config import BaseConfig


class ConvertConfig(BaseConfig):
    model: str
    """HF repo id or local snapshot path whose weights are converted to the
    PrimeRL grouped format under `<snapshot>/prime/`."""
