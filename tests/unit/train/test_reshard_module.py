import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from prime_rl.trainer.model import reshard_module

WORLD_SIZE = 2


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(8, 8, bias=False)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.layer(x)))


def _eval_between_backward_and_optimizer_step(rank: int, port: int) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD_SIZE)
    torch.manual_seed(0)
    mesh = init_device_mesh("cpu", (WORLD_SIZE,))
    model = TinyLM()
    # setup_fsdp's layout: the lm_head and norm group is not resharded after a forward
    fully_shard(model.layer, mesh=mesh)
    fully_shard([model.lm_head, model.norm], mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    x = torch.randn(4, 8)

    model(x).sum().backward()
    with torch.no_grad():
        model(x)
    reshard_module(model)
    params_with_grad = sorted(name for name, param in model.named_parameters() if param.grad is not None)
    assert params_with_grad == ["layer.weight", "lm_head.weight", "norm.bias", "norm.weight"]
    optimizer.step()
    optimizer.zero_grad()
    weights_used: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(lambda module, *_: weights_used.append(module.weight.detach().clone()))
    with torch.no_grad():
        model(x)
    reshard_module(model)
    hook.remove()
    updated_lm_head = model.lm_head.weight.full_tensor()
    assert torch.equal(weights_used[0], updated_lm_head)
    dist.destroy_process_group()


def test_reshard_after_no_grad_forward_keeps_gradients_and_updated_weights(free_port: int) -> None:
    mp.spawn(_eval_between_backward_and_optimizer_step, args=(free_port,), nprocs=WORLD_SIZE, join=True)
