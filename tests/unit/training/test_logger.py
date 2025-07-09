from zeroband.trainer.config import LogConfig
from zeroband.trainer.logger import setup_logger
from zeroband.trainer.world import get_world
from zeroband.utils.logger import reset_logger


def test_setup_default():
    reset_logger()
    setup_logger(LogConfig(), get_world())
