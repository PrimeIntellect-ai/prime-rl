"""Probe batch invariance and gradients of the shared MoE operators on CUDA."""

import argparse
import json
from pathlib import Path

import torch
from vllm.platforms import current_platform

from prime_rl.trainer.models.layers.moe_alignment import (
    AlignedGroupedGemm,
    AlignedLocalDispatcher,
    _RouterLinear,
    route,
    router_linear,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    current_platform.import_kernels()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {}

    x = torch.randn(129, 2048, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(128, 2048, device="cuda", dtype=torch.float32) * 0.02
    full = router_linear(x, gate)
    for rows in (1, 17, 128):
        logits = router_linear(x[:rows], gate)
        torch.testing.assert_close(logits, full[:rows], rtol=0, atol=0)
        scores, ids, _ = route(logits, 8, True)
        all_scores, all_ids, _ = route(full, 8, True)
        torch.testing.assert_close(scores, all_scores[:rows], rtol=0, atol=0)
        torch.testing.assert_close(ids, all_ids[:rows], rtol=0, atol=0)
    reference = x.double() @ gate.double().T
    torch.testing.assert_close(full.double(), reference, rtol=1e-4, atol=1e-5)
    results["router_batch_shapes_exact"] = [1, 17, 128]
    results["router_fp64_max_abs_error"] = (full.double() - reference).abs().max().item()
    _, tied, _ = route(torch.zeros_like(full[:1]), 8, True)
    assert tied.tolist() == [list(range(8))]

    rx, rw = x[:17].clone().requires_grad_(), gate.clone().requires_grad_()
    gradient = torch.randn(17, 128, device="cuda")
    dx, dw = torch.autograd.grad(_RouterLinear.apply(rx, rw), (rx, rw), gradient)
    torch.testing.assert_close(dx, (gradient @ gate).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(dw, gradient.T @ rx.float(), rtol=0, atol=0)
    results["router_gradients_exact"] = True

    gemm = AlignedGroupedGemm()
    gx = x[:17].clone().requires_grad_()
    weight = (torch.randn(4, 2048, 768, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
    offsets = torch.tensor([1, 1, 5, 17], device="cuda", dtype=torch.int32)
    out = gemm(gx, weight, offs=offsets)
    reference = torch.cat([gx[:1] @ weight[0], gx[1:5] @ weight[2], gx[5:] @ weight[3]])
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, (gx, weight), grad)
    reference_grads = torch.autograd.grad(reference, (gx, weight), grad)
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_grads[1][1].count_nonzero() == 0
    results["grouped_gradients_exact_including_empty_expert"] = True

    dispatcher = AlignedLocalDispatcher(4)
    expert_weight = torch.randn(4, 2048, 2048, device="cuda", dtype=torch.bfloat16) * 0.02

    def experts(inputs, counts):
        return gemm(inputs, expert_weight, offs=counts.cumsum(0))

    selected_scores, selected_ids, _ = route(full[:, :4], 2, True)
    all_out = dispatcher.run(x, selected_scores, selected_ids, experts, score_before_experts=False)
    for rows in (1, 17, 128):
        out = dispatcher.run(x[:rows], selected_scores[:rows], selected_ids[:rows], experts, score_before_experts=False)
        torch.testing.assert_close(out, all_out[:rows], rtol=0, atol=0)
    results["expert_dispatch_and_reduction_batch_shapes_exact"] = [1, 17, 128]

    glm_x = torch.randn(129, 4096, device="cuda", dtype=torch.bfloat16)
    glm_gate = torch.randn(128, 4096, device="cuda") * 0.02
    correction = torch.randn(128, device="cuda") * 0.01
    glm_logits = router_linear(glm_x, glm_gate)
    glm_scores, glm_ids, _ = route(glm_logits, 8, True, score_func="sigmoid", selection_bias=correction)
    for rows in (1, 17, 128):
        logits = router_linear(glm_x[:rows], glm_gate)
        scores, ids, _ = route(logits, 8, True, score_func="sigmoid", selection_bias=correction)
        torch.testing.assert_close(logits, glm_logits[:rows], rtol=0, atol=0)
        torch.testing.assert_close(scores, glm_scores[:rows], rtol=0, atol=0)
        torch.testing.assert_close(ids, glm_ids[:rows], rtol=0, atol=0)
    torch.testing.assert_close(glm_logits.double(), glm_x.double() @ glm_gate.double().T, rtol=1e-4, atol=1e-5)
    results["glm_sigmoid_router_batch_shapes_exact"] = [1, 17, 128]

    from prime_rl.trainer.models.layers.dense_alignment import _Rotary, rope_table

    q = torch.randn(1, 3, 8192, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 1, 8192, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cache = rope_table(64, 8192, 1e6).to(device="cuda", dtype=torch.bfloat16)
    rq, rk = _Rotary.apply(q, k, cache)
    torch.testing.assert_close(rq[..., 64:], q[..., 64:], rtol=0, atol=0)
    torch.testing.assert_close(rk[..., 64:], k[..., 64:], rtol=0, atol=0)
    dq, dk = torch.autograd.grad((rq, rk), (q, k), (torch.ones_like(rq), torch.ones_like(rk)))
    torch.testing.assert_close(dq[..., 64:], torch.ones_like(dq[..., 64:]), rtol=0, atol=0)
    torch.testing.assert_close(dk[..., 64:], torch.ones_like(dk[..., 64:]), rtol=0, atol=0)
    results["glm_partial_rope_preserves_unrotated_channels_and_gradients"] = True

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    cpu_gate = glm_gate.cpu().pin_memory()
    uva_gate = get_accelerator_view_from_cpu_tensor(cpu_gate)
    torch.testing.assert_close(router_linear(glm_x, uva_gate), glm_logits, rtol=0, atol=0)
    cpu_experts = expert_weight.cpu().pin_memory()
    uva_experts = get_accelerator_view_from_cpu_tensor(cpu_experts)
    counts = torch.tensor([1, 1, 5, 17], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(
        gemm(x[:17], uva_experts, offs=counts), gemm(x[:17], expert_weight, offs=counts), rtol=0, atol=0
    )
    results["uva_cpu_weight_router_and_expert_gemm_exact"] = True
    updated_gate = glm_gate + torch.randn_like(glm_gate) * 1e-6
    updated_experts = expert_weight + torch.randn_like(expert_weight) * 0.001
    uva_gate.copy_(updated_gate)
    uva_experts.copy_(updated_experts)
    torch.cuda.synchronize()
    assert torch.equal(cpu_gate.view(torch.int32), updated_gate.cpu().view(torch.int32))
    assert torch.equal(cpu_experts.view(torch.int16), updated_experts.cpu().view(torch.int16))
    assert torch.equal(
        router_linear(glm_x, uva_gate).view(torch.int32), router_linear(glm_x, updated_gate).view(torch.int32)
    )
    assert torch.equal(
        gemm(x[:17], uva_experts, offs=counts).view(torch.int16),
        gemm(x[:17], updated_experts, offs=counts).view(torch.int16),
    )
    results["uva_updated_weight_storage_and_forward_bits_exact"] = True
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
