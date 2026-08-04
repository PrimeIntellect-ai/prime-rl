import argparse
import copy
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from prime_rl.trainer.optim import CPUOffloadOptimizer


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.up = nn.Linear(width, width * 2, bias=False)
        self.down = nn.Linear(width * 2, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.down(torch.nn.functional.silu(self.up(hidden)))


class FakeModel(nn.Module):
    def __init__(self, width: int, layers: int, fp32_router: bool):
        super().__init__()
        self.layers = nn.ModuleList(Block(width) for _ in range(layers))
        self.router = nn.Linear(width, width, bias=False, dtype=torch.float32) if fp32_router else None
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        if self.router is not None:
            hidden = hidden + self.router(hidden.float()).to(hidden.dtype)
        return self.output(hidden).float().square().mean()


def shard_model(model: nn.Module, reduce_dtype: torch.dtype, hsdp: bool) -> nn.Module:
    if hsdp:
        mesh = init_device_mesh("cuda", (dist.get_world_size(), 1), mesh_dim_names=("dp_replicate", "dp_shard"))
    else:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    if model.router is not None:
        fully_shard(
            model.router,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
        )
    fully_shard(model, mesh=mesh, mp_policy=policy)
    return model


def make_optimizer(model: nn.Module, grad_offload: bool, dp_replicate: int) -> CPUOffloadOptimizer:
    named_params = list(model.named_parameters())
    optimizer = torch.optim.AdamW((param for _, param in named_params), lr=1e-3)
    return CPUOffloadOptimizer(
        optimizer,
        named_params,
        model=model,
        grad_cpu_offload=grad_offload,
        dp_replicate=dp_replicate,
    )


@torch.no_grad()
def reference_clip_grad_norm(model: nn.Module, max_norm: float, dp_replicate: int) -> torch.Tensor:
    local_squared_norm = torch.zeros((), dtype=torch.float32, device="cuda")
    for param in model.parameters():
        if param.grad is not None:
            norm = torch.linalg.vector_norm(param.grad.to_local(), dtype=torch.float32)
            local_squared_norm.add_(norm.square())
    dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
    local_squared_norm.div_(dp_replicate)
    total_norm = local_squared_norm.sqrt_()
    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(coefficient)
    return total_norm


@torch.no_grad()
def reference_zero_gradient_ratio(model: nn.Module, dp_replicate: int) -> float:
    counts = torch.zeros(2, dtype=torch.long, device="cuda")
    for param in model.parameters():
        local_param = param.to_local()
        counts[1] += local_param.numel()
        if param.grad is None:
            counts[0] += local_param.numel()
        else:
            counts[0] += local_param.numel() - torch.count_nonzero(param.grad.to_local())
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    counts = torch.div(counts, dp_replicate, rounding_mode="floor")
    return (counts[0].float() / counts[1].clamp_min(1).float()).item()


def train_step(
    model: nn.Module,
    optimizer: CPUOffloadOptimizer,
    inputs: list[torch.Tensor],
    grad_offload: bool,
    dp_replicate: int,
) -> tuple[torch.Tensor, float, int, int, int, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for micro_step, hidden in enumerate(inputs):
        if grad_offload:
            optimizer.begin_backward(collect_stats=micro_step == len(inputs) - 1)
        (model(hidden) / len(inputs)).backward()
        if grad_offload:
            optimizer.finish_backward(wait_for_copies=micro_step == len(inputs) - 1)

    optimizer.scale_gradients_(1.25)
    if grad_offload:
        zero_ratio = optimizer.zero_gradient_ratio()
        grad_norm = optimizer.clip_grad_norm_(0.5)
        gpu_grad_bytes = sum(param.grad.to_local().nbytes for param in model.parameters() if param.grad is not None)
        cpu_grad_bytes = sum(
            buffer.accumulator.nbytes + (buffer.staging.nbytes if buffer.staging is not None else 0)
            for buffer in optimizer.grad_offloader._buffers.values()
        )
    else:
        zero_ratio = reference_zero_gradient_ratio(model, dp_replicate)
        grad_norm = reference_clip_grad_norm(model, max_norm=0.5, dp_replicate=dp_replicate)
        gpu_grad_bytes = sum(param.grad.to_local().nbytes for param in model.parameters() if param.grad is not None)
        cpu_grad_bytes = 0

    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return grad_norm, zero_ratio, gpu_grad_bytes, cpu_grad_bytes, torch.cuda.max_memory_allocated(), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=3)
    parser.add_argument("--reduce-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hsdp", action="store_true")
    parser.add_argument("--fp32-router", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.manual_seed(17)
    reduce_dtype = getattr(torch, args.reduce_dtype)

    reference = FakeModel(args.width, args.layers, args.fp32_router).to(device="cuda")
    for layer in reference.layers:
        layer.to(torch.bfloat16)
    reference.output.to(torch.bfloat16)
    candidate = copy.deepcopy(reference)
    if args.compile:
        for reference_layer, candidate_layer in zip(reference.layers, candidate.layers):
            reference_layer.compile(fullgraph=False)
            candidate_layer.compile(fullgraph=False)
    shard_model(reference, reduce_dtype, args.hsdp)
    shard_model(candidate, reduce_dtype, args.hsdp)
    dp_replicate = dist.get_world_size() if args.hsdp else 1
    reference_optimizer = make_optimizer(reference, False, dp_replicate)
    candidate_optimizer = make_optimizer(candidate, True, dp_replicate)
    parameter_count = sum(param.numel() for param in reference.parameters())

    if dist.get_rank() == 0:
        print(f"parameters={parameter_count / 1e9:.3f}B warmup={args.warmup_steps} measured={args.steps}")

    if args.snapshot_dir is not None:
        torch.cuda.memory._record_memory_history()

    reference_times = []
    candidate_times = []
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        generator = torch.Generator(device="cuda").manual_seed(1000 + step)
        inputs = [
            torch.randn(2, 128, args.width, device="cuda", dtype=torch.bfloat16, generator=generator)
            for _ in range(args.accumulation)
        ]
        reference_norm, reference_zero_ratio, reference_gpu_grads, _, reference_peak, reference_time = train_step(
            reference, reference_optimizer, inputs, False, dp_replicate
        )
        (
            candidate_norm,
            candidate_zero_ratio,
            candidate_gpu_grads,
            candidate_cpu_grads,
            candidate_peak,
            candidate_time,
        ) = train_step(candidate, candidate_optimizer, inputs, True, dp_replicate)
        torch.testing.assert_close(candidate_norm, reference_norm, rtol=2e-3, atol=2e-3, check_dtype=False)
        torch.testing.assert_close(candidate_zero_ratio, reference_zero_ratio, rtol=2e-2, atol=1e-5)
        for reference_param, candidate_param in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(candidate_param.to_local(), reference_param.to_local(), rtol=2e-3, atol=2e-3)
        if step >= args.warmup_steps:
            reference_times.append(reference_time)
            candidate_times.append(candidate_time)
        if dist.get_rank() == 0:
            print(
                f"step={step} reference_time={reference_time:.3f}s offload_time={candidate_time:.3f}s "
                f"norm={candidate_norm.item():.6f} "
                f"reference_gpu_grads={reference_gpu_grads / 1024**2:.1f} MiB "
                f"offload_gpu_grads={candidate_gpu_grads / 1024**2:.1f} MiB "
                f"offload_pinned_grads={candidate_cpu_grads / 1024**2:.1f} MiB "
                f"reference_peak={reference_peak / 1024**2:.1f} MiB "
                f"offload_peak={candidate_peak / 1024**2:.1f} MiB"
            )

    if dist.get_rank() == 0:
        reference_mean = statistics.mean(reference_times)
        candidate_mean = statistics.mean(candidate_times)
        regression = candidate_mean / reference_mean - 1
        print(
            f"steady_state reference={reference_mean:.3f}s offload={candidate_mean:.3f}s regression={regression:+.1%}"
        )

    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(args.snapshot_dir / f"rank-{dist.get_rank()}.pickle"))
    if args.compile:
        graph_breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
        if graph_breaks:
            raise RuntimeError(f"torch.compile recorded {graph_breaks} graph breaks")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
