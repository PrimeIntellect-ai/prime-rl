"""Single-rank torch.compile probe over a wrapped grouped-experts forward.

uv run python -m prime_rl.experimental.fully_shard_caching.compile_smoke
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from prime_rl.experimental.fully_shard_caching.mini_model import PreparedGroupedExperts
from prime_rl.experimental.fully_shard_caching.ops import Fp8GroupedExpertCompute, RowScaledGroupedExpertCompute
from prime_rl.experimental.fully_shard_caching.prepared_tensor import PREPARE_CALLS, install_prepared_weights
from prime_rl.trainer.models.layers.activations import Silu

EXPERTS = 4
DIM = 256
HIDDEN_DIM = 128
TOKENS_PER_EXPERT = 16

FP8_EXPERTS = 16
FP8_DIM = 4096
FP8_HIDDEN_DIM = 1408
FP8_TOKENS_PER_EXPERT = 128


def build_experts(compile_backend: str | None, install: bool, op, dim, hidden_dim, experts_count):
    torch.manual_seed(0)
    experts = PreparedGroupedExperts(
        dim=dim, hidden_dim=hidden_dim, num_experts=experts_count, expert_type="gated", activation="silu"
    )
    experts.compute = op
    experts = experts.cuda().to(torch.float32)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    if install:
        install_prepared_weights(experts, {name: experts.compute.prepare for name, _ in experts.named_parameters()})
    if compile_backend is not None:
        experts.compile(backend=compile_backend)
    fully_shard(experts, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    return experts


def probe(label, compile_backend, install, x, num_tokens_per_expert, op, dim, hidden_dim, experts_count) -> None:
    torch._dynamo.reset()
    try:
        experts = build_experts(compile_backend, install, op, dim, hidden_dim, experts_count)
        output = experts(x, num_tokens_per_expert)
        output.float().sum().backward()
        print(f"{label}: ok, prepare calls {PREPARE_CALLS.count}, output norm {output.float().norm().item():.4f}")
    except Exception as error:
        print(f"{label}: {type(error).__name__}: {str(error).splitlines()[0][:400]}")
    PREPARE_CALLS.reset()


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29741")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", 0))

    toy = RowScaledGroupedExpertCompute(activation=Silu)
    toy_shape = (DIM, HIDDEN_DIM, EXPERTS)
    num_tokens_per_expert = torch.full((EXPERTS,), TOKENS_PER_EXPERT, dtype=torch.int64, device="cuda")
    x = torch.randn(EXPERTS * TOKENS_PER_EXPERT, DIM, device="cuda", dtype=torch.bfloat16)

    for backend in (None, "eager", "inductor"):
        for install in (False, True):
            label = f"toy {backend or 'eager-only'}, {'wrapped' if install else 'unwrapped'}"
            probe(label, backend, install, x, num_tokens_per_expert, toy, *toy_shape)

    fp8 = Fp8GroupedExpertCompute(activation=Silu)
    fp8_shape = (FP8_DIM, FP8_HIDDEN_DIM, FP8_EXPERTS)
    fp8_counts = torch.full((FP8_EXPERTS,), FP8_TOKENS_PER_EXPERT, dtype=torch.int64, device="cuda")
    fp8_x = torch.randn(FP8_EXPERTS * FP8_TOKENS_PER_EXPERT, FP8_DIM, device="cuda", dtype=torch.bfloat16)

    for backend in (None, "eager", "inductor"):
        for install in (False, True):
            label = f"fp8 {backend or 'eager-only'}, {'wrapped' if install else 'unwrapped'}"
            probe(label, backend, install, fp8_x, fp8_counts, fp8, *fp8_shape)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
