import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.moe import MoE, MoEArgs

dist.init_process_group("nccl")
torch.set_default_device("cuda")
torch.cuda.set_device(dist.get_rank())

moe_args = MoEArgs()

moe_args.num_experts = 8
moe_args.num_shared_experts = 0
moe_args.score_func = "softmax"
moe_args.route_norm = False
moe_args.route_scale = 1.0
moe_args.score_before_experts = False
moe_args.top_k = 4
moe_args.use_grouped_mm = True
moe_args.load_balance_coeff = None

MODEL_DIM = 16


class MeowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.moe = MoE(moe_args, dim=MODEL_DIM, hidden_dim=2 * MODEL_DIM)

    def forward(self, x):
        x = self.mlp(x)
        return self.moe(x)


def setup_fsdp(model: nn.Module):
    world_mesh = dist.init_device_mesh("cuda", (2, 2), mesh_dim_names=("fsdp", "ep"))
    ep_mesh = world_mesh["ep"]
    fsdp_mesh = world_mesh["fsdp"]
    parallelize_module(
        module=model.moe.experts,
        device_mesh=ep_mesh,
        parallelize_plan=ExpertParallel(),
    )
    fully_shard(
        model,
        mesh=fsdp_mesh,
    )


def pprint_params(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        print(
            f"DTensor(shape={tensor.shape}, local_tensor={tensor.to_local().shape}, device_mesh={tensor.device_mesh}, placements={tensor.placements})"
        )
    else:
        print(f"Tensor(shape={tensor.shape}, device={tensor.device})")


model = MeowModel()
if dist.get_rank() == 0:
    for name, param in model.named_parameters():
        print(name)
        pprint_params(param)
setup_fsdp(model)
if dist.get_rank() == 0:
    print("== After FSDP ==")
    for name, param in model.named_parameters():
        print(name)
        pprint_params(param)
        print()

input = torch.arange(0, 2 * 3).reshape(2, 3, 1).expand((2, 3, MODEL_DIM)).float()
out = model(input)
print(out.shape)
out.sum().backward()

dist.destroy_process_group()
