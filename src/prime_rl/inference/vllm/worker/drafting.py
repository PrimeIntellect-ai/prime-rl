import gc

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.inference.vllm.worker_drafting")


def get_draft_model(model_runner) -> torch.nn.Module | None:
    """The served draft, also while drafting is off and the runner holds no speculator."""
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    speculator = getattr(model_runner, "prime_rl_speculator", model_runner.speculator)
    return speculator.model if isinstance(speculator, DraftModelSpeculator) else None


@torch.no_grad()
def set_num_speculative_tokens(model_runner, num: int) -> None:
    """Re-record the target's CUDA graphs for ``num`` draft tokens per step; 0 detaches the draft.

    Full decode graphs are recorded for one query length, 1 + the draft tokens, so a different count needs
    new graphs; the draft's own graphs are kept. Called on every worker while the engines are paused, so
    data-parallel ranks record together."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.workspace import unlock_workspace

    if not hasattr(model_runner, "prime_rl_speculator"):
        model_runner.prime_rl_speculator = model_runner.speculator
    if not 0 <= num <= model_runner.num_speculative_steps:
        raise ValueError(f"num_speculative_tokens must be in [0, {model_runner.num_speculative_steps}]")
    decode_query_len = num + model_runner.model_state.num_new_sampled_tokens_per_step
    drafting = model_runner.speculator is not None
    if decode_query_len == model_runner.decode_query_len and drafting == (num > 0):
        return

    old = model_runner.cudagraph_manager
    settings = dict(
        cudagraph_mode=old.cudagraph_mode,
        lora_capture_cases=old.lora_capture_cases,
        varlen_decode=old.varlen_decode,
        ubatch_runner=old.ubatch_runner,
    )
    # Dropping the manager releases its graphs into the shared pool.
    model_runner.cudagraph_manager = None
    del old
    gc.collect()
    torch.accelerator.empty_cache()
    model_runner.decode_query_len = decode_query_len
    model_runner.cudagraph_manager = ModelCudaGraphManager(
        model_runner.vllm_config, model_runner.device, decode_query_len=decode_query_len, **settings
    )
    # The draft's graphs do not depend on the target's query length: keep them out of the capture.
    model_runner.speculator = None
    unlock_workspace()
    try:
        memory = model_runner.capture_model()
    finally:
        model_runner.speculator = model_runner.prime_rl_speculator if num > 0 else None
    logger.info("Recorded CUDA graphs for %d draft tokens per step (%.2f GiB)", num, memory / 2**30)
