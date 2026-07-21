def global_inference_rank(
    *,
    rank_offset: int,
    data_parallel_index: int,
    worker_rank: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    inference_world_size: int,
    prefill_context_parallel_size: int = 1,
    engine_world_size: int | None = None,
) -> int:
    """Map one vLLM worker to its rank in Prime's inference NCCL group."""
    model_parallel_size = tensor_parallel_size * pipeline_parallel_size * prefill_context_parallel_size
    if model_parallel_size <= 0:
        raise ValueError("model parallel size must be positive")

    local_rank = data_parallel_index * model_parallel_size + worker_rank % model_parallel_size
    rank = rank_offset + local_rank
    if engine_world_size is not None and not (rank_offset <= rank < rank_offset + engine_world_size):
        raise ValueError(
            f"calculated inference rank {rank} is outside engine rank interval "
            f"[{rank_offset}, {rank_offset + engine_world_size})"
        )
    if not 0 <= rank < inference_world_size:
        raise ValueError(f"calculated inference rank {rank} is outside inference world size {inference_world_size}")
    return rank
