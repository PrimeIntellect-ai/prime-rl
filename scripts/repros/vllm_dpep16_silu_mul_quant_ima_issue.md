### Problem

`vllm.model_executor.layers.quantization.utils.fp8_utils._silu_mul_per_token_group_quant_fp8_colmajor`
does row-based pointer arithmetic in int32. With large DeepGEMM MoE warmup/workspace shapes this overflows
and the Triton launch fails with CUDA illegal memory access.

The failure is easy to hit for GLM-style MoE inference with DPEP=16 and 36k max tokens per rank. The
DeepGEMM activation-quant workspace for the SiLU/mul path is:

```text
M = round_up(tokens_per_rank * dpep_size * top_k + local_experts * 127, 128)
  = round_up(36000 * 16 * 8 + 16 * 127, 128)
  = 4,610,048
N = 4096  # 2 * GLM MoE intermediate size
max input element offset = M * N - 1 = 18,882,756,607
int32 max = 2,147,483,647
```

The kernel is called from `DeepGemmExperts._act_mul_quant` for the Hopper/non-E8M0 SiLU path.

### Environment

Validated on 2 nodes x 8 NVIDIA H200 GPUs:

```text
vllm 0.19.0
torch 2.10.0+cu128
triton 3.6.0
cuda 12.8
deep_gemm 2.5.0+891d57b
deep_ep 1.2.1+29d31c0
```

### Minimal reproducer

Save as `repro_vllm_deepgemm_silu_int32.py`:

```python
import os
import socket

import torch
import torch.distributed as dist
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_per_token_group_quant_fp8_colmajor,
)

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
torch.cuda.set_device(local_rank)

if world > 1:
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dist.barrier()

M = 4_610_048
N = 4096
print(f"rank={rank}/{world} host={socket.gethostname()} M={M} N={N} max_offset={M * N - 1}", flush=True)

x = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
y, scales = silu_mul_per_token_group_quant_fp8_colmajor(x, use_ue8m0=False)
torch.cuda.synchronize()
print(f"rank={rank} ok {y.shape} {scales.shape}", flush=True)
```

Launch as 16 ranks over two 8-GPU nodes:

```bash
srun -N2 --ntasks-per-node=8 --gres=gpu:8 --exclusive bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  export MASTER_PORT=29577
  export VLLM_USE_DEEP_GEMM=1
  export VLLM_MOE_USE_DEEP_GEMM=1
  python repro_vllm_deepgemm_silu_int32.py
'
```

Observed failure:

```text
rank=0/16 host=... M=4610048 N=4096 max_offset=18882756607
...
File ".../vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 785,
  in silu_mul_per_token_group_quant_fp8_colmajor
    _silu_mul_per_token_group_quant_fp8_colmajor[grid](...)
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
NCCL WARN Cuda failure 700 'an illegal memory access was encountered'
```

The same kernel also fails on one GPU with the first aligned overflowing shape:

```python
M = 524_416
N = 4096
```

### Proposed fix

Upcast row/column offsets used for pointer arithmetic to `tl.int64` inside the Triton kernel before computing
load/store addresses. In particular:

```python
m_offset = (pid_m * BLOCK_M).to(tl.int64)
n_offset = (pid_n * BLOCK_N).to(tl.int64)
offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

base_y_ptr = y_ptr + m_offset * N + n_offset
base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
```

I validated this exact int64-addressing change with the same 2-node / 16-rank launch and `M=4,610,048, N=4096`;
all ranks completed:

```text
rank 0/16 quant ok output_shape=(4610048, 2048) scales_shape=(4610048, 16)
...
rank 15/16 quant ok output_shape=(4610048, 2048) scales_shape=(4610048, 16)
```
