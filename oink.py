import torch
import torch.distributed as dist
from torch.distributed.tensor.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM

from prime_rl.trainer.distributed.tensor_parallels import apply_non_moe_tp

dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (2,))

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
print(model)
if dist.get_rank() == 0:
    for name, param in model.named_parameters():
        print(name, type(param.data), param.data.shape)
        if hasattr(param.data, "to_local"):
            print(param.data.to_local().shape)

apply_non_moe_tp(model, mesh)

if dist.get_rank() == 0:
    for name, param in model.named_parameters():
        print(name, type(param.data), param.data.shape)
        if hasattr(param.data, "to_local"):
            print(param.data.to_local().shape)

out = model(torch.randint(0, 100, (1, 10)))
print(out.shape)

dist.destroy_process_group()
