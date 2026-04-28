from functools import wraps

import torch
import torch.nn.functional as F
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_general_routing_inputs
from torchtitan.distributed.expert_parallel import expert_parallel


def _disable_sonic_autotune() -> None:
    import sonicmoe.functional.backward as sonic_backward
    import sonicmoe.functional.forward as sonic_forward

    def force_no_autotune(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            kwargs["tuned"] = False
            return fn(*args, **kwargs)

        return wrapped

    sonic_forward.gemm = force_no_autotune(sonic_forward.gemm)
    sonic_forward.gemm_gated = force_no_autotune(sonic_forward.gemm_gated)
    sonic_backward.gemm = force_no_autotune(sonic_backward.gemm)
    sonic_backward.gemm_dgated = force_no_autotune(sonic_backward.gemm_dgated)


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def _run_experts_grouped_mm_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    # grouped mm between a 2D tensor and a 3D tensor
    assert x.dim() == 2

    h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


def _run_experts_sonic_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    top_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    gate_up_proj = torch.stack(
        [
            w1.bfloat16().transpose(-2, -1),
            w3.bfloat16().transpose(-2, -1),
        ],
        dim=-1,
    ).flatten(-2)

    out, _ = moe_general_routing_inputs(
        x.bfloat16(),
        top_scores,
        token_indices,
        expert_indices,
        gate_up_proj.permute(2, 1, 0),
        None,
        w2.bfloat16().permute(1, 2, 0),
        None,
        w1.shape[0],
        torch.cuda.current_stream().cuda_stream,
        ActivationType.SWIGLU,
        False,
        False,
    )

    return out.type_as(x)


def _run_nongated_experts_grouped_mm_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    assert x.dim() == 2

    h = relu2(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return out


@expert_parallel
def _run_nongated_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_nongated_experts_grouped_mm_impl(w1, w2, _w3, x, num_tokens_per_expert)
