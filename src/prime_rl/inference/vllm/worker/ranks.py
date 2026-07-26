def global_inference_rank(
    *,
    rank_offset: int,
    data_parallel_index: int,
    data_parallel_size: int,
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

    logical_data_parallel_size = data_parallel_size
    if engine_world_size is not None:
        if engine_world_size <= 0 or engine_world_size % model_parallel_size:
            raise ValueError(
                f"engine world size {engine_world_size} is not divisible by model parallel size {model_parallel_size}"
            )
        logical_data_parallel_size = engine_world_size // model_parallel_size
        # Dense vLLM EngineCore processes retain their global DP index but rewrite
        # data_parallel_size to one. MoE EngineCore processes preserve the logical
        # size, so keep validating that value rather than masking bad discovery.
        if data_parallel_size != 1 and data_parallel_size != logical_data_parallel_size:
            raise ValueError(
                f"data parallel size {data_parallel_size} does not match engine-derived size "
                f"{logical_data_parallel_size}"
            )
    if not 0 <= data_parallel_index < logical_data_parallel_size:
        raise ValueError(
            f"data parallel index {data_parallel_index} is outside logical data parallel size "
            f"{logical_data_parallel_size}"
        )

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
