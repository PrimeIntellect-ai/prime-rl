# import torch
# from prime_rl.trainer.models.layers.lora import _run_lora_for_loop
#
# x = torch.randn(10, 5)
# IN_DIM, OUT_DIM, R_DIM = 5, 4, 2
# N_ADAPTERS = 3
# lora_A = torch.randn(N_ADAPTERS, IN_DIM, R_DIM)
# lora_B = torch.randn(N_ADAPTERS, R_DIM, OUT_DIM)
# offsets = torch.tensor([3, 6, 10], dtype=torch.int32)
#
# lora_out = _run_lora_for_loop(x, lora_A, lora_B, offsets)
# print(lora_out.shape)

import torch

torch.set_default_device("cuda")
N = 2**14
A = torch.randn(N, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
B = torch.randn(2, N, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
offs = torch.tensor([N // 2, N], dtype=torch.int32, device="cuda")

C = torch._grouped_mm(A, B, offs=offs)
# C.sum().backward()
grad_output = torch.randn_like(C)
grads = torch.autograd.grad(C, [A, B], grad_outputs=grad_output, create_graph=False, retain_graph=False)
print(grads)
