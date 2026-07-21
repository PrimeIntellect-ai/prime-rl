import importlib.util
from pathlib import Path

import pytest

RANKS_PATH = Path(__file__).parents[3] / "src/prime_rl/inference/vllm/worker/ranks.py"
SPEC = importlib.util.spec_from_file_location("prime_rl_nccl_ranks_under_test", RANKS_PATH)
assert SPEC and SPEC.loader
ranks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranks)


def test_dense_aggregate_dp4_tp2_ranks_are_unique_across_nodes():
    actual = {
        ranks.global_inference_rank(
            rank_offset=0,
            data_parallel_index=dp_index,
            data_parallel_size=4,
            worker_rank=tp_rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=8,
        )
        for dp_index in range(4)
        for tp_rank in range(2)
    }

    assert actual == set(range(8))


def test_dense_aggregate_uses_engine_size_when_vllm_rewrites_dp_size():
    actual = {
        ranks.global_inference_rank(
            rank_offset=0,
            data_parallel_index=dp_index,
            data_parallel_size=1,
            worker_rank=tp_rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=8,
            engine_world_size=8,
        )
        for dp_index in range(4)
        for tp_rank in range(2)
    }

    assert actual == set(range(8))


def test_already_global_moe_worker_ranks_are_not_double_counted():
    actual = {
        ranks.global_inference_rank(
            rank_offset=0,
            data_parallel_index=dp_index,
            data_parallel_size=4,
            worker_rank=dp_index * 2 + tp_rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=8,
        )
        for dp_index in range(4)
        for tp_rank in range(2)
    }

    assert actual == set(range(8))


def test_rank_offset_composes_with_pipeline_parallel_rank():
    actual = {
        ranks.global_inference_rank(
            rank_offset=8,
            data_parallel_index=dp_index,
            data_parallel_size=2,
            worker_rank=model_parallel_rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            inference_world_size=16,
        )
        for dp_index in range(2)
        for model_parallel_rank in range(4)
    }

    assert actual == set(range(8, 16))


def test_per_server_rank_offset_with_internal_dp_zero_is_unchanged():
    actual = {
        ranks.global_inference_rank(
            rank_offset=4,
            data_parallel_index=0,
            data_parallel_size=1,
            worker_rank=tp_rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=8,
        )
        for tp_rank in range(2)
    }

    assert actual == {4, 5}


def test_global_inference_rank_rejects_out_of_bounds_rank():
    with pytest.raises(ValueError, match="outside inference world size"):
        ranks.global_inference_rank(
            rank_offset=2,
            data_parallel_index=3,
            data_parallel_size=4,
            worker_rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=8,
        )


def test_prefill_context_parallel_ranks_are_unique():
    actual = {
        ranks.global_inference_rank(
            rank_offset=0,
            data_parallel_index=0,
            data_parallel_size=1,
            worker_rank=worker_rank,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=2,
            inference_world_size=2,
            engine_world_size=2,
        )
        for worker_rank in range(2)
    }

    assert actual == {0, 1}


def test_engine_world_size_must_be_divisible_by_model_parallel_size():
    for worker_rank in range(2):
        with pytest.raises(ValueError, match="engine world size 1 is not divisible by model parallel size 2"):
            ranks.global_inference_rank(
                rank_offset=0,
                data_parallel_index=0,
                data_parallel_size=1,
                worker_rank=worker_rank,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=2,
                inference_world_size=2,
                engine_world_size=1,
            )


def test_preserved_data_parallel_size_must_match_discovered_engine_span():
    with pytest.raises(ValueError, match="data parallel size 4 does not match engine-derived size 5"):
        ranks.global_inference_rank(
            rank_offset=0,
            data_parallel_index=0,
            data_parallel_size=4,
            worker_rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=10,
            engine_world_size=10,
        )


def test_data_parallel_index_must_fit_discovered_engine_span():
    with pytest.raises(ValueError, match="data parallel index 4 is outside logical data parallel size 4"):
        ranks.global_inference_rank(
            rank_offset=0,
            data_parallel_index=4,
            data_parallel_size=1,
            worker_rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=8,
            engine_world_size=8,
        )
