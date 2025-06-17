from zeroband.training.config import LogConfig
from zeroband.training.logger import setup_logger
from zeroband.training.world_info import get_world_info
from zeroband.utils.logger import reset_logger


def test_setup_default():
    reset_logger()
    setup_logger(LogConfig(), get_world_info())
