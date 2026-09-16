"""Standalone torchrun training loop for the A/B matrix in RESULTS.md.

    uv run torchrun --nproc-per-node 8 -m prime_rl.experimental.fully_shard_caching.train_mini \\
        --wrap fp8 --grad-accum 4 --ac full --steps 30 --metrics-out logs/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from prime_rl.experimental.fully_shard_caching.mini_model import (
    WRAP_MODES,
    MiniModelSpec,
    build_mini_model,
    expert_modules,
)
from prime_rl.experimental.fully_shard_caching.prepared_tensor import PREPARE_CALLS

BYTES_PER_GIB = 1024**3


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
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--attn-implementation", default="flash_attention_3")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--install-prepared", type=boolean, default=True)
    parser.add_argument("--metrics-out", type=Path, default=None)
    return parser.parse_args()


def build_meshes(ep: int):
    world_size = dist.get_world_size()
    if world_size % ep:
        raise ValueError(f"World size {world_size} is not divisible by ep={ep}.")
    mesh = init_device_mesh("cuda", (world_size // ep, ep), mesh_dim_names=("dp_shard_mod_ep", "dp_shard_in_ep"))
    fsdp_mesh = mesh["dp_shard_mod_ep", "dp_shard_in_ep"]._flatten(mesh_dim_name="dp_shard")
    return fsdp_mesh, mesh["dp_shard_mod_ep"], mesh["dp_shard_in_ep"]


def fake_batch(seed: int, step: int, microbatch: int, seq_len: int, vocab_size: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(
        (seed * 1_000_003 + step * 9_973 + microbatch * 97 + dist.get_rank()) % (2**63)
    )
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=generator, device=device)
    return input_ids, input_ids.roll(-1, dims=1)


def grad_norm(model) -> float:
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.to_local().double().pow(2).sum()
    dist.all_reduce(total)
    return total.sqrt().item()


def run_microbatch(model, input_ids, labels, seq_lens, temperature) -> torch.Tensor:
    output = model(input_ids=input_ids, labels=labels, temperature=temperature, seq_lens=seq_lens)
    return -output["logprobs"].mean()


def main() -> None:
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    fsdp_mesh, dp_mod_ep_mesh, ep_mesh = build_meshes(args.ep)
    spec = MiniModelSpec(
        num_hidden_layers=args.layers,
        wrap=args.wrap,
        ep_size=args.ep,
        seed=args.seed,
        attn_implementation=args.attn_implementation,
    )
    model = build_mini_model(
        spec,
        fsdp_mesh,
        dp_mod_ep_mesh,
        ep_mesh,
        expert_reshard_after_forward=args.expert_reshard_after_forward,
        activation_checkpointing=args.ac == "full",
        install_prepared=args.install_prepared,
        device=device,
    )
    experts = expert_modules(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)

    seq_lens = torch.tensor([args.seq_len], device=device)
    temperature = torch.ones((1, args.seq_len), device=device)
    vocab_size = model.config.vocab_size

    dist.barrier()
    torch.cuda.synchronize()
    setup_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    PREPARE_CALLS.reset()

    step_times: list[float] = []
    prepare_calls_per_step: list[int] = []
    loss_history: list[float] = []
    first_step_microbatch_losses: list[float] = []

    for step in range(args.steps):
        prepare_calls_before = PREPARE_CALLS.count
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for microbatch in range(args.grad_accum):
            if not args.expert_reshard_after_backward:
                for module in experts:
                    module.set_reshard_after_backward(microbatch == args.grad_accum - 1, recurse=False)
            input_ids, labels = fake_batch(args.seed, step, microbatch, args.seq_len, vocab_size, device)
            loss = run_microbatch(model, input_ids, labels, seq_lens, temperature)
            if step == 0:
                first_step_microbatch_losses.append(loss.item())
            (loss / args.grad_accum).backward()
            step_loss += loss.item() / args.grad_accum
        if step == 0:
            first_step_grad_norm = grad_norm(model)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        loss_history.append(step_loss)
        if step >= args.warmup_steps:
            step_times.append(elapsed)
            prepare_calls_per_step.append(PREPARE_CALLS.count - prepare_calls_before)
        if dist.get_rank() == 0:
            print(f"step {step} loss {step_loss:.6f} time {elapsed:.3f}s prepares {PREPARE_CALLS.count}", flush=True)

    median_step_time = statistics.median(step_times)
    tokens_per_step = args.seq_len * args.grad_accum
    metrics = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "world_size": dist.get_world_size(),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / BYTES_PER_GIB,
        "setup_memory_gib": setup_memory / BYTES_PER_GIB,
        "tokens_per_sec_per_gpu": tokens_per_step / median_step_time,
        "median_step_time_s": median_step_time,
        "step_times_s": step_times,
        "prepare_calls_total": PREPARE_CALLS.count,
        "prepare_calls_per_measured_step": prepare_calls_per_step,
        "first_step_microbatch_losses": first_step_microbatch_losses,
        "first_step_grad_norm": first_step_grad_norm,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
    }
    if dist.get_rank() == 0:
        print(json.dumps({key: metrics[key] for key in ("peak_memory_gib", "tokens_per_sec_per_gpu", "final_loss")}))
        if args.metrics_out is not None:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(metrics, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
