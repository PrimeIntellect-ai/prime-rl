#!/usr/bin/env python3
"""Benchmark Prime-RL weighted MoE gather implementations."""

from __future__ import annotations

import torch

from common import (
    BaseMoEBenchmarkConfig,
    benchmark_variant,
    cli,
    make_token_weights,
    make_topk_ids,
    print_result_table,
    require_cuda,
    resolve_dtype,
)
from prime_rl.trainer.models.layers import moe_backends as moe_backends_mod
from prime_rl.trainer.models.layers.moe_backends import _torch_histogram, _torch_index_compute


class GatherBenchmarkConfig(BaseMoEBenchmarkConfig):
    """Benchmark the Prime-RL weighted gather stage."""

    invalid_frac: float = 0.1


def _torch_weighted_gather(input: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    token_num, topk = index.shape
    hidden_dim = input.shape[-1]
    out = input.new_zeros((token_num, hidden_dim))
    if token_num == 0 or index.numel() == 0:
        return out

    flat_index = index.reshape(-1)
    valid = flat_index >= 0
    if not valid.any():
        return out

    flat_weight = weight.reshape(-1)
    flat_token_indices = torch.arange(token_num, device=index.device, dtype=torch.long).repeat_interleave(topk)
    gathered = input.index_select(0, flat_index[valid].to(torch.int64))
    gathered = (gathered.to(torch.float32) * flat_weight[valid].unsqueeze(-1)).to(input.dtype)
    out.index_add_(0, flat_token_indices[valid], gathered)
    return out


def main() -> None:
    config = cli(GatherBenchmarkConfig)
    device = require_cuda()
    if moe_backends_mod.triton_moe_weighted_gather is None:
        raise RuntimeError(f"Triton gather kernel is unavailable: {moe_backends_mod._TRITON_IMPORT_ERROR}")

    from prime_rl.trainer.models.layers.moe_triton import triton_moe_weighted_gather

    torch.manual_seed(0)
    token_expert_ids = make_topk_ids(
        config.token_num,
        config.topk,
        config.num_experts,
        device,
        allow_dupe=config.allow_dupe,
        invalid_frac=config.invalid_frac,
    )
    expert_histogram = _torch_histogram(token_expert_ids, config.num_experts)
    scatter_index = _torch_index_compute(token_expert_ids, expert_histogram)
    expert_token_num = int((scatter_index >= 0).sum().item())

    expert_output = torch.randn(
        (expert_token_num, config.hidden),
        device=device,
        dtype=resolve_dtype(config.dtype),
    )
    token_weights = make_token_weights(config.token_num, config.topk, device)

    fw_ok = None
    bw_ok = None
    if config.check:
        expert_output_ref = expert_output.detach().clone().requires_grad_(True)
        token_weights_ref = token_weights.detach().clone().requires_grad_(True)
        out_ref = _torch_weighted_gather(expert_output_ref, scatter_index, token_weights_ref)
        out_ref.sum().backward()
        dx_ref = expert_output_ref.grad.detach().clone()
        dw_ref = token_weights_ref.grad.detach().clone()

        expert_output_test = expert_output.detach().clone().requires_grad_(True)
        token_weights_test = token_weights.detach().clone().requires_grad_(True)
        out_test = triton_moe_weighted_gather(expert_output_test, scatter_index, token_weights_test)
        out_test.sum().backward()

        torch.testing.assert_close(out_test, out_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(expert_output_test.grad, dx_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(token_weights_test.grad, dw_ref, rtol=1e-2, atol=1e-2)
        fw_ok = True
        bw_ok = True

    results = []

    expert_output_base = expert_output.detach().clone().requires_grad_(True)
    token_weights_base = token_weights.detach().clone().requires_grad_(True)

    def run_base() -> torch.Tensor:
        return _torch_weighted_gather(expert_output_base, scatter_index, token_weights_base)

    def reset_base_grads() -> None:
        expert_output_base.grad = None
        token_weights_base.grad = None

    base = benchmark_variant("baseline", run_base, reset_base_grads, warmup=config.warmup, iters=config.iters)
    base.fw_ok = fw_ok
    base.bw_ok = bw_ok
    results.append(base)

    expert_output_triton = expert_output.detach().clone().requires_grad_(True)
    token_weights_triton = token_weights.detach().clone().requires_grad_(True)

    def run_triton() -> torch.Tensor:
        return triton_moe_weighted_gather(expert_output_triton, scatter_index, token_weights_triton)

    def reset_triton_grads() -> None:
        expert_output_triton.grad = None
        token_weights_triton.grad = None

    triton_result = benchmark_variant(
        "triton",
        run_triton,
        reset_triton_grads,
        warmup=config.warmup,
        iters=config.iters,
    )
    triton_result.fw_ok = fw_ok
    triton_result.bw_ok = bw_ok
    results.append(triton_result)

    print_result_table(
        "prime_moe_gather",
        {
            "token_num": config.token_num,
            "hidden": config.hidden,
            "topk": config.topk,
            "experts": config.num_experts,
            "invalid_frac": config.invalid_frac,
        },
        results,
        include_checks=config.check,
    )


if __name__ == "__main__":
    main()
