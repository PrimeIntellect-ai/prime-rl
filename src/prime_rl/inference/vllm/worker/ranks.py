def global_inference_rank(
    *,
    rank_offset: int,
    data_parallel_index: int,
    worker_rank: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    inference_world_size: int,
) -> int:
    """Map one vLLM worker to its rank in Prime's inference NCCL group."""
    model_parallel_size = tensor_parallel_size * pipeline_parallel_size
    if model_parallel_size <= 0:
        raise ValueError("model parallel size must be positive")

    rank = (
        rank_offset
        + data_parallel_index * model_parallel_size
        + worker_rank % model_parallel_size
    )
    if not 0 <= rank < inference_world_size:
        raise ValueError(
            f"calculated inference rank {rank} is outside inference world size "
            f"{inference_world_size}"
        )
    return rank
