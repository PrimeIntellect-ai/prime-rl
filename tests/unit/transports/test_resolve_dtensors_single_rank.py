"""TEMPORARY: delete before merging this branch.

Exercises the real NCCL-transport call path on CPU, to prove the fp32 declaration survives
`resolve_dtensors` end to end. It needs a live process group, which is why it does not belong in
the permanent suite: the repo's only other transport test is skipped with "fail only in ci".
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM
from prime_rl.transports.weights.nccl import resolve_dtensors


@pytest.fixture
def single_rank_mesh(free_port):
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(free_port),
        RANK="0",
        WORLD_SIZE="1",
    )
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_resolve_dtensors_keeps_declared_keys_in_fp32(single_rank_mesh):
    declared = "model.layers.0.linear_attn.A_log"
    ordinary = "model.layers.0.self_attn.q_proj.weight"
    buffer_key = "model.layers.0.mlp.router.selection_bias"
    state_dict = {
        declared: distribute_tensor(torch.randn(8, dtype=torch.float32), single_rank_mesh, [Shard(0)]),
        ordinary: distribute_tensor(torch.randn(8, 4, dtype=torch.float32), single_rank_mesh, [Shard(0)]),
        buffer_key: torch.zeros(8, dtype=torch.float32),
    }

    resolved = resolve_dtensors(state_dict, Qwen3_5ForCausalLM.keep_in_fp32_for_weight_transfer, torch.bfloat16)

    assert resolved[declared].dtype is torch.float32
    assert resolved[ordinary].dtype is torch.bfloat16
    # Non-DTensor entries, which under FSDP2 means the buffers, never reach the dtype decision at
    # all. qwen3_5 does not declare this key, so were it a DTensor it would come out bf16; it
    # stays fp32 only because the gate skips it.
    assert resolved[buffer_key].dtype is torch.float32
    # `broadcast_state_dict` asserts on any surviving DTensor, so the gather must be complete.
    assert not any(isinstance(value, DTensor) for value in resolved.values())
