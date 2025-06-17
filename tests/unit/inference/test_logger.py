from zeroband.inference.config import LogConfig, ParallelConfig
from zeroband.inference.logger import setup_logger
from zeroband.utils.logger import reset_logger


def test_setup_default():
    reset_logger()
    setup_logger(LogConfig(), ParallelConfig(), dp_rank=0)
