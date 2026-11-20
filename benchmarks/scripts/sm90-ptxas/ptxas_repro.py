"""Minimal repro: ptxas 12.8 to 13.0 miscompile a Triton pointwise kernel on sm90 (13.3+ and -O0 are correct).

    python ptxas_repro.py                      # Triton's bundled ptxas 12.9: mismatches
    TRITON_PTXAS_PATH=<ptxas 13.3> python ptxas_repro.py   # no mismatches
"""

import torch
import triton
import triton.language as tl


@triton.jit
def qkv_grad_cat_kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (64*x1 + 32768*((((x0) // 64) % 4)) + (((x0) % 64))), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp4 >= tmp5
    tmp9 = tl.full([1], 384, tl.int64)
    tmp10 = tmp4 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + (64*x1 + 32768*(((((-256) + x0) // 64) % 2)) + ((((-256) + x0) % 64))), tmp11, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp0 >= tmp9
    tmp14 = tl.full([1], 512, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tmp16 == tmp16
    tmp18 = tl.load(in_ptr2 + (128*x1 + ((((-384) + x0) % 128))), tmp13, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.full([1], 0.0, tl.float32)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp13, tmp20, tmp21)
    tmp23 = tl.where(tmp11, tmp12, tmp22)
    tmp24 = tl.where(tmp6, tmp7, tmp23)
    tl.store(out_ptr0 + (x2), tmp24, None)


tokens = 512
q = torch.randn(4, tokens, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, tokens, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
expected = torch.cat([q.transpose(0, 1).reshape(tokens, 256), k.transpose(0, 1).reshape(tokens, 128), v], dim=1)
for xblock, num_warps in [(1024, 4), (512, 8)]:
    out = torch.empty(tokens, 512, device="cuda", dtype=torch.bfloat16)
    qkv_grad_cat_kernel[(tokens * 512 // xblock,)](q, k, v, out, tokens * 512, XBLOCK=xblock, num_warps=num_warps)
    torch.cuda.synchronize()
    wrong = (out != expected).nonzero()
    print(f"XBLOCK={xblock} num_warps={num_warps}: {len(wrong)} wrong of {out.numel()}, first {wrong[:3].tolist()}")
