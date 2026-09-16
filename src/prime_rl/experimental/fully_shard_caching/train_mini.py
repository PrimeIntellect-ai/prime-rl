"""Standalone torchrun training loop for the A/B matrix in RESULTS.md.

    uv run torchrun --nproc-per-node 8 -m prime_rl.experimental.fully_shard_caching.train_mini \\
        --wrap fp8 --grad-accum 4 --ac full --steps 30 --metrics-out logs/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from prime_rl.configs.trainer import AdamWConfig
from prime_rl.experimental.fully_shard_caching.mini_model import (
    DEFAULT_NUM_EXPERTS,
    WRAP_MODES,
    MiniModelSpec,
    build_mini_model,
    expert_modules,
)
from prime_rl.experimental.fully_shard_caching.prepared_tensor import PREPARE_CALLS
from prime_rl.trainer.model import forward
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import GarbageCollection, clip_grad_norm_, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.world import get_world

BYTES_PER_GIB = 1024**3
GC_INTERVAL = 50


def boolean(value: str) -> bool:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrap", choices=WRAP_MODES, default="none")
    parser.add_argument("--expert-reshard-after-forward", type=boolean, default=True)
    parser.add_argument("--expert-reshard-after-backward", type=boolean, default=True)
    parser.add_argument("--ac", choices=("none", "full"), default="none")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--attn-implementation", default="flash_attention_3")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--install-prepared", type=boolean, default=True)
    parser.add_argument("--fusions", type=boolean, default=True)
    parser.add_argument("--force-balanced-routing", type=boolean, default=True)
    parser.add_argument("--compile", type=boolean, default=False)
    parser.add_argument("--metrics-out", type=Path, default=None)
    return parser.parse_args()


def fake_batch(generator: torch.Generator, seq_len: int, vocab_size: int, device: torch.device):
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=generator, device=device)
    return input_ids, input_ids.roll(-1, dims=1)


def main() -> None:
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")
    setup_torch_distributed()
    world = get_world()
    device = torch.device("cuda", world.local_rank)
    torch.manual_seed(args.seed)

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=world.world_size, cp=1, pp=1, ep=args.ep, world_size=world.world_size
    )
    spec = MiniModelSpec(
        num_hidden_layers=args.layers,
        wrap=args.wrap,
        num_experts=args.num_experts,
        seed=args.seed,
        attn_implementation=args.attn_implementation,
    )
    model = build_mini_model(
        spec,
        parallel_dims,
        expert_reshard_after_forward=args.expert_reshard_after_forward,
        activation_checkpointing=args.ac == "full",
        install_prepared=args.install_prepared,
        fusions=args.fusions,
        force_balanced_routing=args.force_balanced_routing,
        compile_layers=args.compile,
        device=device,
    )
    experts = expert_modules(model)
    optimizer_config = AdamWConfig(lr=args.lr)
    optimizer, _ = setup_optimizer(optimizer_config, list(model.named_parameters()), parallel_dims)

    seq_lens = torch.tensor([args.seq_len], device=device)
    position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0)
    temperature = torch.ones((1, args.seq_len), device=device)
    vocab_size = model.config.vocab_size
    batch_generator = torch.Generator(device=device).manual_seed(args.seed + world.rank)

    dist.barrier()
    torch.cuda.synchronize()
    setup_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    PREPARE_CALLS.reset()

    perf_counter = get_perf_counter(model, args.seq_len)
    garbage_collection = GarbageCollection(GC_INTERVAL)

    measured_steps = max(args.steps - args.warmup_steps, 0)
    step_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measured_steps)]
    step_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measured_steps)]
    step_losses = torch.zeros(args.steps, device=device)
    microbatch_losses = torch.zeros(args.grad_accum, device=device)
    prepare_calls_per_step: list[int] = []

    for step in range(args.steps):
        garbage_collection.run(step)
        prepare_calls_before = PREPARE_CALLS.count
        measured = step - args.warmup_steps
        if measured >= 0:
            step_start_events[measured].record()
        optimizer.zero_grad(set_to_none=True)
        for microbatch in range(args.grad_accum):
            if not args.expert_reshard_after_backward:
                for module in experts:
                    module.set_reshard_after_backward(microbatch == args.grad_accum - 1, recurse=False)
            input_ids, labels = fake_batch(batch_generator, args.seq_len, vocab_size, device)
            loss = -forward(model, input_ids, position_ids, seq_lens=seq_lens, labels=labels, temperature=temperature)[
                "logprobs"
            ].mean()
            if step == 0:
                microbatch_losses[microbatch] = loss.detach()
            (loss / args.grad_accum).backward()
            step_losses[step] += loss.detach() / args.grad_accum
        scale_gradients_(None, model, parallel_dims.fsdp_gradient_divide_factor)
        grad_norm = clip_grad_norm_(None, model, optimizer_config.max_norm, parallel_dims.ep_enabled)
        if step == 0:
            first_step_grad_norm = grad_norm.detach()
        optimizer.step()
        if measured >= 0:
            step_end_events[measured].record()
            prepare_calls_per_step.append(PREPARE_CALLS.count - prepare_calls_before)

    torch.cuda.synchronize()
    step_times = [start.elapsed_time(end) / 1000 for start, end in zip(step_start_events, step_end_events)]
    loss_history = step_losses.tolist()
    median_step_time = statistics.median(step_times)
    tokens_per_step = args.seq_len * args.grad_accum
    metrics = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "world_size": world.world_size,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / BYTES_PER_GIB,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / BYTES_PER_GIB,
        "setup_memory_gib": setup_memory / BYTES_PER_GIB,
        "tokens_per_sec_per_gpu": perf_counter.get_step_tokens_per_second(tokens_per_step, median_step_time),
        "mfu": perf_counter.get_step_mfu(tokens_per_step * world.world_size, median_step_time),
        "median_step_time_s": median_step_time,
        "step_times_s": step_times,
        "prepare_calls_total": PREPARE_CALLS.count,
        "prepare_calls_per_measured_step": prepare_calls_per_step,
        "first_step_microbatch_losses": microbatch_losses.tolist(),
        "first_step_grad_norm": first_step_grad_norm.item(),
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
    }
    if world.is_master:
        for step, step_loss in enumerate(loss_history):
            measured = step - args.warmup_steps
            if measured >= 0:
                print(
                    f"step {step} loss {step_loss:.6f} time {step_times[measured]:.3f}s "
                    f"prepares {prepare_calls_per_step[measured]}",
                    flush=True,
                )
            else:
                print(f"step {step} loss {step_loss:.6f} warmup", flush=True)
        print(json.dumps({key: metrics[key] for key in ("peak_memory_gib", "tokens_per_sec_per_gpu", "final_loss")}))
        if args.metrics_out is not None:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(metrics, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
