import time
from collections.abc import Callable

import torch
from dion import Muon
from torch import Tensor, nn
from torch.optim import AdamW

from prime_rl.configs.trainer import TrainerConfig, rl_micro_batch_multiple
from prime_rl.trainer.model import iter_moe_blocks
from prime_rl.trainer.optim import OffloadOptimizer, OptimizerLike
from prime_rl.trainer.optim.sign_sgd import SignSGD
from prime_rl.trainer.rl.data import TensorMicroBatch
from prime_rl.trainer.utils import clip_grad_norm_, get_peak_memory_across_ranks
from prime_rl.trainer.vlm import get_language_model
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
    # Real micro batches are padded up to a multiple, so the longest one can exceed `seq_len`
    full = -(-seq_len // multiple) * multiple
    near_full = max(multiple, full - multiple)
    return [
        warmup_doc_lens(full, 0, generator),
        warmup_doc_lens(near_full, 0, generator),
        warmup_doc_lens(near_full, near_full // 2, generator),
    ]


def warmup_routed_experts(model: nn.Module, num_tokens: int, generator: torch.Generator) -> Tensor | None:
    """Distinct random experts per token and layer, in the layout inference returns for router replay."""
    routers = [moe.router for moe in iter_moe_blocks(model)]
    if not routers:
        return None
    num_experts = min(router.num_experts for router in routers)
    num_layers = len(get_language_model(model).layers)
    first = torch.randint(0, num_experts, (1, num_tokens, num_layers, 1), generator=generator, dtype=torch.int32)
    return (first + torch.arange(routers[0].top_k, dtype=torch.int32)) % num_experts


def synthetic_micro_batch(
    doc_lens: list[int], model: nn.Module, config: TrainerConfig, generator: torch.Generator
) -> TensorMicroBatch:
    """A micro batch with no token in any loss component, so it trains nothing, like a dropped micro batch. It
    carries the inputs that the trainer config makes real micro batches carry."""
    num_tokens = sum(doc_lens)
    vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
    micro_batch = TensorMicroBatch(
        input_ids=torch.randint(0, vocab_size, (1, num_tokens), generator=generator),
        position_ids=torch.cat([torch.arange(doc_len) for doc_len in doc_lens]).unsqueeze(0),
        advantages=torch.zeros(1, num_tokens),
        inference_logprobs=torch.zeros(1, num_tokens),
        ref_logprobs=None,
        loss_mask=torch.zeros(1, num_tokens, dtype=torch.bool),
        temperatures=torch.ones(1, num_tokens),
        env_names=["warmup"] * len(doc_lens),
        sequence_lengths=doc_lens,
        padding=0,
        trace_ids=None,
        branch_indices=None,
        seq_lens=torch.tensor(doc_lens, dtype=torch.long),
        routed_experts=warmup_routed_experts(model, num_tokens, generator) if config.enable_router_replay else None,
        support_ids=None,
        support_logprobs=None,
        support_offsets=None,
        support_replayed=None,
        # Score centering supports are heads of the sampler's distribution, which take the dense backward
        sparse_head_backward=False,
        mm_refs=None,
        mm_kwargs=None,
        mm_token_type_ids=None,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        staleness=None,
    )
    if config.loss.type == "icepop" and config.loss.score_centering:
        k = config.loss.score_centering_topk
        first = torch.randint(0, vocab_size, (num_tokens, 1), generator=generator, dtype=torch.int32)
        micro_batch["support_ids"] = ((first + torch.arange(k, dtype=torch.int32)) % vocab_size).flatten()
        micro_batch["support_logprobs"] = -torch.rand(num_tokens * k, generator=generator)
        micro_batch["support_offsets"] = torch.arange(0, num_tokens * k + 1, k, dtype=torch.long)
        micro_batch["support_replayed"] = torch.zeros(num_tokens, dtype=torch.bool)
    return micro_batch


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
    forward_backward: Callable[..., object],
) -> float:
    """Do the trainer's one-time work (compilation, kernel JIT, communicator setup, optimizer state) before the
    first batch arrives. `forward_backward` is the training step's forward and backward. Returns the time it took."""
    logger = get_logger()
    logger.info("Warming up the trainer")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    generator = torch.Generator().manual_seed(0)
    micro_batches = [
        synthetic_micro_batch(doc_lens, model, config, generator)
        for doc_lens in warmup_micro_batches(config.model.seq_len, rl_micro_batch_multiple(config.model))
    ]
    # Under full offload, the gradients then stay in the offload buffers instead of updating the weights
    forward_backward(micro_batches, prepare=lambda micro_batch: micro_batch, overlap_optimizer=False, annotate=False)
    del micro_batches
    torch.cuda.synchronize()
    forward_backward_time = time.perf_counter() - start_time

    optimizer_start_time = time.perf_counter()
    # The full offload optimizer runs on the CPU and creates its state up front, so it has nothing to warm up
    if config.model.offload == "full":
        optimizer.zero_grad()
        stepped = False
    else:
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
