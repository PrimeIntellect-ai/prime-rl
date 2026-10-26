import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from prime_rl.trainer.distributed import ExpertWeightParallel


@pytest.fixture(scope="module")
def cpu_mesh():
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29513", rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,))
    dist.destroy_process_group()


def test_expert_parallel_keeps_frozen_params_frozen(cpu_mesh):
    experts = nn.Module()
    experts.gate_proj = nn.Parameter(torch.zeros(2, 4, 4), requires_grad=False)
    experts.down_proj = nn.Parameter(torch.zeros(2, 4, 4))

    ExpertWeightParallel()._apply(experts, cpu_mesh)

    assert not experts.gate_proj.requires_grad
    assert experts.down_proj.requires_grad
