import pytest

from prime_rl.inference.vllm.ranks import global_inference_rank


def test_dense_aggregate_uses_engine_size_when_vllm_rewrites_dp_size():
    actual = {
        global_inference_rank(
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
        global_inference_rank(
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
        global_inference_rank(
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


def test_global_inference_rank_rejects_out_of_bounds_rank():
    with pytest.raises(ValueError, match="outside inference world size"):
        global_inference_rank(
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
        global_inference_rank(
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
