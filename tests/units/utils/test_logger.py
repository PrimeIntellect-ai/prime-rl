import logging
import os

import pytest

from zeroband.utils.logger import ALLOWED_LEVELS, LoggerName, PrimeFormatter, get_logger


def test_init_with_default_args():
    logger = get_logger("TRAIN")
    assert logger.name == "TRAIN"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.handlers[0].formatter, PrimeFormatter)


@pytest.mark.parametrize("log_level", list(ALLOWED_LEVELS.keys()))
def test_init_with_valid_log_level(log_level: str):
    os.environ["PRIME_LOG_LEVEL"] = log_level
    logger = get_logger("TRAIN")
    assert logger.level == ALLOWED_LEVELS[log_level]


def test_init_with_invalid_log_level():
    os.environ["PRIME_LOG_LEVEL"] = "INVALID"
    logger = get_logger("TRAIN")
    assert logger.level == logging.INFO


def test_init_multiple_loggers():
    train_logger = get_logger("TRAIN")
    inference_logger = get_logger("INFER")

    assert train_logger.name == "TRAIN"
    assert inference_logger.name == "INFER"
    assert train_logger != inference_logger


@pytest.mark.parametrize("name", ["TRAIN", "INFER"])
@pytest.mark.parametrize("local_rank", [0, 1])
def test_init_with_different_ranks(name: LoggerName, local_rank: int):
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(local_rank + 1)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_rank + 1)  # To not raise exception in world info
    logger = get_logger(name)
    if name == "TRAIN":
        if local_rank == 0:
            assert logger.level == logging.INFO
        else:
            assert logger.level == logging.CRITICAL
    else:
        assert logger.level == logging.INFO
