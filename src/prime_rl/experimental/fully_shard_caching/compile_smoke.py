"""Single-rank torch.compile probe over a wrapped grouped-experts forward.

uv run python -m prime_rl.experimental.fully_shard_caching.compile_smoke
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard, distribute_tensor

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

EP_MESH_DIM = "ep"
FSDP_MESH_DIM = "dp_shard_mod_ep"


def shard_expert_weights(experts: nn.Module, ep_mesh) -> None:
    """Replace each expert parameter with a DTensor, as ``ExpertWeightParallel`` does in production."""
    for name, parameter in list(experts.named_parameters(recurse=False)):
        experts.register_parameter(name, nn.Parameter(distribute_tensor(parameter, ep_mesh, [Shard(0)])))


def build_experts(compile_backend: str | None, install: bool, op, dim, hidden_dim, experts_count, dp_ep_mesh):
    torch.manual_seed(0)
    experts = PreparedGroupedExperts(
        dim=dim, hidden_dim=hidden_dim, num_experts=experts_count, expert_type="gated", activation="silu"
    )
    experts.compute = op
    experts = experts.cuda().to(torch.float32)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    if install:
        install_prepared_weights(experts, {name: experts.compute for name, _ in experts.named_parameters()})
    if dp_ep_mesh is not None:
        shard_expert_weights(experts, dp_ep_mesh[EP_MESH_DIM])
    if compile_backend is not None:
        experts.compile(backend=compile_backend)
    fsdp_mesh = dp_ep_mesh[FSDP_MESH_DIM] if dp_ep_mesh is not None else None
    fully_shard(experts, mesh=fsdp_mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    return experts


def graph_breaks() -> str:
    reasons = torch._dynamo.utils.graph_break_reasons
    if not reasons:
        return "breaks 0"
    first = " ".join(reasons[0].reason.split())[:200]
    return f"breaks {len(reasons)}, first {first}"


def probe(
    label, compile_backend, install, x, num_tokens_per_expert, op, dim, hidden_dim, experts_count, dp_ep_mesh=None
) -> None:
    torch._dynamo.reset()
    try:
        experts = build_experts(compile_backend, install, op, dim, hidden_dim, experts_count, dp_ep_mesh)
        output = experts(x, num_tokens_per_expert)
        output.float().sum().backward()
        print(
            f"{label}: ok, {graph_breaks()}, prepare calls {PREPARE_CALLS.count}, "
            f"output norm {output.float().norm().item():.4f}"
        )
    except Exception as error:
        message = " ".join(str(error).split())[:400]
        print(f"{label}: {graph_breaks()}, {type(error).__name__}: {message}")
    PREPARE_CALLS.reset()


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29741")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", 0))
    # Matches apply_compile in trainer/model.py, whose MoE histc and offsets produce scalars.
    torch._dynamo.config.capture_scalar_outputs = True

    # A one-rank expert mesh shards nothing, but the parameter FSDP sees is a real
    # DTensor(ShardedPreparedTensor), which is the nesting dynamo has to trace through.
    dp_ep_mesh = init_device_mesh("cuda", (1, 1), mesh_dim_names=(FSDP_MESH_DIM, EP_MESH_DIM))

    toy = RowScaledGroupedExpertCompute(activation=Silu)
    toy_shape = (DIM, HIDDEN_DIM, EXPERTS)
    num_tokens_per_expert = torch.full((EXPERTS,), TOKENS_PER_EXPERT, dtype=torch.int64, device="cuda")
    torch.manual_seed(0)
    x = torch.randn(EXPERTS * TOKENS_PER_EXPERT, DIM, device="cuda", dtype=torch.bfloat16)

    for backend in (None, "eager", "inductor"):
        for install in (False, True):
            for ep in (False, True):
                label = (
                    f"toy {backend or 'eager-only'}, {'wrapped' if install else 'unwrapped'}, "
                    f"{'ep1-mesh' if ep else 'no-ep'}"
                )
                probe(
                    label,
                    backend,
                    install,
                    x,
                    num_tokens_per_expert,
                    toy,
                    *toy_shape,
                    dp_ep_mesh if ep else None,
                )

    fp8 = Fp8GroupedExpertCompute(activation=Silu)
    fp8_shape = (FP8_DIM, FP8_HIDDEN_DIM, FP8_EXPERTS)
    fp8_counts = torch.full((FP8_EXPERTS,), FP8_TOKENS_PER_EXPERT, dtype=torch.int64, device="cuda")
    torch.manual_seed(0)
    fp8_x = torch.randn(FP8_EXPERTS * FP8_TOKENS_PER_EXPERT, FP8_DIM, device="cuda", dtype=torch.bfloat16)

    for backend in (None, "eager", "inductor"):
        for install in (False, True):
            for ep in (False, True):
                label = (
                    f"fp8 {backend or 'eager-only'}, {'wrapped' if install else 'unwrapped'}, "
                    f"{'ep1-mesh' if ep else 'no-ep'}"
                )
                probe(
                    label,
                    backend,
                    install,
                    fp8_x,
                    fp8_counts,
                    fp8,
                    *fp8_shape,
                    dp_ep_mesh if ep else None,
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
