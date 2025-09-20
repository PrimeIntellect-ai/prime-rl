from pathlib import Path

from prime_rl.utils.logger import reset_logger, setup_logger


def test_setup_default():
    reset_logger()
    setup_logger("info", log_file=Path("test") / "logs" / "test.log")
