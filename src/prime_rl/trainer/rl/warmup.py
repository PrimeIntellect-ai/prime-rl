import time

import torch
from dion import Muon
from torch import Tensor, nn
from torch.optim import AdamW

from prime_rl.configs.trainer import TrainerConfig, rl_micro_batch_multiple
from prime_rl.trainer.model import forward, is_tt_moe_model, take_moe_routing_stats
from prime_rl.trainer.optim import OffloadOptimizer, OptimizerLike
from prime_rl.trainer.optim.sign_sgd import SignSGD
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.loss import selective_log_softmax, shift_tensor_left
from prime_rl.trainer.utils import clip_grad_norm_, get_peak_memory_across_ranks
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_cp_params, shard_for_cp
from prime_rl.utils.logger import format_time, get_logger

# Their state starts at zero and stays zero under zero gradients, and lr scales both the update and the
# weight decay, so a step with lr = 0 changes nothing but the step counters.
DRY_STEP_OPTIMIZERS = (AdamW, Muon, SignSGD)


def warmup_doc_lens(total: int, long_doc: int, generator: torch.Generator) -> list[int]:
    """Document lengths summing to `total`: one document of `long_doc` tokens, then short ones up to `total` / 8."""
    doc_lens = [long_doc] if long_doc > 0 else []
    while sum(doc_lens) < total:
        doc_len = int(torch.randint(1, max(2, total // 8), (1,), generator=generator))
        doc_lens.append(min(doc_len, total - sum(doc_lens)))
    return doc_lens


def warmup_micro_batches(seq_len: int, multiple: int) -> list[list[int]]:
    """Document lengths of the warmup micro batches.

    Dynamo compiles one graph for the full length and, on the next length, one for any length. It takes the 32-bit
    indexing guards of that dynamic graph from the length it first sees, so the warmup uses the longest length below
    full: every shorter real length then passes them. A long document changes the graphs of some layers, so the
    last micro batch has one.
    """
    generator = torch.Generator().manual_seed(0)
    full = seq_len // multiple * multiple
    near_full = max(multiple, full - multiple)
    return [
        warmup_doc_lens(full, 0, generator),
        warmup_doc_lens(near_full, 0, generator),
        warmup_doc_lens(near_full, near_full // 2, generator),
    ]


def forward_backward(
    model: nn.Module,
    doc_lens: list[int],
    vocab_size: int,
    config: TrainerConfig,
    parallel_dims: ParallelDims,
    final_backward: bool,
) -> None:
    """Run the training forward and a backward of zero loss on random tokens."""
    seq_lens = torch.tensor(doc_lens, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, int(seq_lens.sum())), generator=generator).cuda()
    position_ids = torch.cat([torch.arange(doc_len) for doc_len in doc_lens]).unsqueeze(0).cuda()
    labels = shift_tensor_left(input_ids)
    temperatures = torch.ones(input_ids.shape, device="cuda")
    if parallel_dims.cp_enabled:
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank()
        input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, parallel_dims.cp)
        labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=parallel_dims.cp)
        temperatures = shard_for_cp(temperatures, cp_rank=cp_rank, cp_world_size=parallel_dims.cp)

    with maybe_activation_offloading(config.model.ac_offloading, model):
        out = forward(
            model,
            input_ids,
            position_ids,
            labels=labels,
            temperature=temperatures,
            seq_lens=seq_lens.cuda(),
            seq_lens_are_pre_shard=parallel_dims.cp_enabled,
        )
    logprobs = out.get("logprobs")
    if logprobs is None:
        logprobs = selective_log_softmax(out["logits"] / temperatures.unsqueeze(-1), labels)

    if parallel_dims.dp_replicate_enabled:
        model.set_requires_all_reduce(final_backward)
    (logprobs.sum() * 0.0).backward()


def dry_optimizer_step(optimizer: OptimizerLike, model: nn.Module, max_norm: float | None) -> bool:
    """Take an optimizer step that leaves the parameters and the optimizer's state as they were.

    The first step creates the optimizer state (and pins its offload buffers) and compiles the optimizer's
    kernels. Returns whether it ran: it skips optimizers whose state already exists, such as one loaded from a
    checkpoint, and optimizers not known to stay unchanged under a zero step.
    """
    base_optimizer = optimizer.base_optimizer if isinstance(optimizer, OffloadOptimizer) else optimizer
    if base_optimizer.state or not isinstance(base_optimizer, DRY_STEP_OPTIMIZERS):
        optimizer.zero_grad()
        return False

    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    if max_norm is not None:
        clip_grad_norm_(None, model, max_norm)

    saved = [{key: group[key] for key in ("lr", "step") if key in group} for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    optimizer.step()
    optimizer.zero_grad()
    for group, saved_group in zip(optimizer.param_groups, saved):
        group.update(saved_group)
    for state in base_optimizer.state.values():
        step = state.get("step")
        if isinstance(step, Tensor):
            step.zero_()
        elif step is not None:
            state["step"] = 0
    return True


def warmup(
    model: nn.Module,
    optimizer: OptimizerLike,
    config: TrainerConfig,
    parallel_dims: ParallelDims,
) -> float:
    """Do the trainer's one-time work (compilation, kernel JIT, communicator setup, optimizer state) before the
    first batch arrives. Returns the time it took."""
    logger = get_logger()
    logger.info("Warming up the trainer")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
    micro_batches = warmup_micro_batches(config.model.seq_len, rl_micro_batch_multiple(config.model))
    for index, doc_lens in enumerate(micro_batches):
        forward_backward(model, doc_lens, vocab_size, config, parallel_dims, index == len(micro_batches) - 1)
    if is_tt_moe_model(model):
        take_moe_routing_stats(model)
    torch.cuda.synchronize()
    forward_backward_time = time.perf_counter() - start_time

    optimizer_start_time = time.perf_counter()
    stepped = dry_optimizer_step(optimizer, model, config.optim.max_norm)
    torch.cuda.synchronize()
    optimizer_time = time.perf_counter() - optimizer_start_time

    peak_memory, _ = get_peak_memory_across_ranks()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    warmup_time = time.perf_counter() - start_time
    optimizer_message = f"optimizer {format_time(optimizer_time)}" if stepped else "optimizer skipped"
    logger.info(
        f"Warmed up the trainer in {format_time(warmup_time)} (forward/backward {format_time(forward_backward_time)}, "
        f"{optimizer_message}, peak memory {peak_memory:.1f} GiB)"
    )
    return warmup_time
