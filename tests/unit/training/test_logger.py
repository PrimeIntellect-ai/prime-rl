from prime_rl.trainer.logger import setup_logger
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import LogConfig
from prime_rl.utils.logger import reset_logger


def test_setup_default():
    reset_logger()
    setup_logger(LogConfig(), get_world())
