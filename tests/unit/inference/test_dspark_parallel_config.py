from copy import deepcopy

import pytest
from vllm.config import ParallelConfig
from vllm.v1.worker.gpu.spec_decode.dspark import utils

from prime_rl.inference.patches import monkey_patch_dspark_dp_compile_cache


@pytest.mark.parametrize("replica", [0, 1])
def test_dspark_preserves_independent_replica_cache_index(replica, monkeypatch):
    target = ParallelConfig(
        tensor_parallel_size=2,
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_rank=replica,
        distributed_executor_backend="mp",
    )
    target.reconfigure_for_independent_dp_rank()
    assert target.data_parallel_rank == 0
    assert target.data_parallel_index == replica
    snapshot = deepcopy(target)
    monkeypatch.setattr(utils, "_get_dspark_parallel_config", utils._get_dspark_parallel_config)

    monkey_patch_dspark_dp_compile_cache()
    factory = utils._get_dspark_parallel_config
    monkey_patch_dspark_dp_compile_cache()
    assert utils._get_dspark_parallel_config is factory
    draft = factory(target, tensor_parallel_size=1)

    assert draft.data_parallel_index == replica
    assert draft.data_parallel_rank == 0
    assert draft.data_parallel_size == 1
    assert draft.tensor_parallel_size == 1
    assert draft.pipeline_parallel_size == 1
    assert not draft.enable_eplb and not draft.enable_elastic_ep
    assert target == snapshot
