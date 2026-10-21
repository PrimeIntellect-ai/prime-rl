from functools import reduce

import pytest
from pydantic_config import cli

from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from tests.regression.distributed.sft.cases import CASES, NUM_GPUS, sft_args


@pytest.mark.parametrize("case", [case.param() for case in CASES])
def test_config_validates(case):
    config = cli(SFTConfig, args=sft_args(case, "placeholder/tiny-model"))
    for key, value in case.overrides().items():
        assert reduce(getattr, key.split("."), config) == value, key
    layout = case.layout
    ParallelDims(dp_replicate=layout.dp_replicate, dp_shard=-1, cp=layout.cp, pp=1, ep=layout.ep, world_size=NUM_GPUS)
