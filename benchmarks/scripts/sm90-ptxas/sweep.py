import itertools

import torch


def qkv_grad_cat(dq, dk, dv):
    seq = dv.shape[0]
    return torch.cat([dq.transpose(0, 1).reshape(seq, -1), dk.transpose(0, 1).reshape(seq, -1), dv], dim=1)


for q_heads, kv_heads, head_dim, seq in itertools.product([2, 4, 8, 16, 32, 64, 96], [1, 2, 4, 8], [64, 128, 256], [512, 4096]):
    if kv_heads > q_heads:
        continue
    torch._dynamo.reset()
    dq = torch.randn(q_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
    dk = torch.randn(kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
    dv = torch.randn(seq, kv_heads * head_dim, device="cuda", dtype=torch.bfloat16)
    out = torch.compile(qkv_grad_cat, dynamic=False)(dq, dk, dv)
    print(q_heads, kv_heads, head_dim, seq, bool(torch.equal(out, qkv_grad_cat(dq, dk, dv))), flush=True)
