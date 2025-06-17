from zeroband.inference.config import LogConfig, ParallelConfig
from zeroband.inference.logger import setup_logger


def test_setup_default():
    setup_logger(LogConfig(), ParallelConfig())
