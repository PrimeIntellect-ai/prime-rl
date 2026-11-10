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
    draft = factory(target, tensor_parallel_size=1, draft_model_config=None)

    assert draft.data_parallel_index == replica
    assert draft.data_parallel_rank == 0
    assert draft.data_parallel_size == 1
    assert draft.tensor_parallel_size == 1
    assert draft.pipeline_parallel_size == 1
    assert not draft.enable_eplb and not draft.enable_elastic_ep
    assert target == snapshot


def test_packed_block_stride_fits_deepseek_v4_mla_alignment(monkeypatch):
    import torch
    from vllm.v1.core import kv_cache_planning
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MLAAttentionSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from prime_rl.inference.patches import monkey_patch_block_outermost_mla_stride

    target = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=4,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        model_version="deepseek_v4",
        state_content_bytes=584,
    )
    # A bf16 DSpark draft KV group is the widest page and sets the shared stride.
    draft = FullAttentionSpec(block_size=256, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    groups = [KVCacheGroupSpec(["target"], target), KVCacheGroupSpec([f"draft.{i}" for i in range(5)], draft)]
    monkeypatch.setattr(
        kv_cache_planning, "_get_kv_cache_bytes_per_block", kv_cache_planning._get_kv_cache_bytes_per_block
    )
    assert kv_cache_planning._get_kv_cache_bytes_per_block(groups, KVCacheLayout.BLHNC) % 576 != 0

    monkey_patch_block_outermost_mla_stride()
    stride = kv_cache_planning._get_kv_cache_bytes_per_block(groups, KVCacheLayout.BLHNC)
    assert stride % 576 == 0 and stride >= 5 * draft.page_size_bytes
