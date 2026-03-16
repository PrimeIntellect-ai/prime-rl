#!/usr/bin/env python3
"""Benchmark Prime-RL MoE routing + scatter implementations."""

from __future__ import annotations

import torch

from common import (
    BaseMoEBenchmarkConfig,
    benchmark_variant,
    cli,
    make_topk_ids,
    print_result_table,
    require_cuda,
    resolve_dtype,
    scatter_index_to_token_indices,
)
from prime_rl.trainer.models.layers import moe_backends as moe_backends_mod
from prime_rl.trainer.models.layers.moe_backends import _torch_histogram, _torch_index_compute


class ScatterBenchmarkConfig(BaseMoEBenchmarkConfig):
    """Benchmark the Prime-RL routing + scatter stage."""


def _torch_routing_scatter(
    x: torch.Tensor,
    token_expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_histogram = _torch_histogram(token_expert_ids, num_experts)
    scatter_index = _torch_index_compute(token_expert_ids, expert_histogram)
    token_indices = scatter_index_to_token_indices(scatter_index)
    return x.index_select(0, token_indices), scatter_index


def main() -> None:
    config = cli(ScatterBenchmarkConfig)
    device = require_cuda()
    if moe_backends_mod.triton_histogram is None or moe_backends_mod.triton_index_scatter is None:
        raise RuntimeError(f"Triton routing/scatter kernels are unavailable: {moe_backends_mod._TRITON_IMPORT_ERROR}")

    from prime_rl.trainer.models.layers.moe_triton import triton_histogram, triton_index_scatter

    def triton_routing_scatter(
        x: torch.Tensor,
        token_expert_ids: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_histogram = triton_histogram(token_expert_ids, num_experts)
        return triton_index_scatter(x, token_expert_ids, expert_histogram)

    torch.manual_seed(0)
    x = torch.randn((config.token_num, config.hidden), device=device, dtype=resolve_dtype(config.dtype))
    token_expert_ids = make_topk_ids(
        config.token_num,
        config.topk,
        config.num_experts,
        device,
        allow_dupe=config.allow_dupe,
        invalid_frac=config.invalid_frac,
    )

    fw_ok = None
    bw_ok = None
    if config.check:
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref, scatter_index_ref = _torch_routing_scatter(x_ref, token_expert_ids, config.num_experts)
        out_ref.sum().backward()
        dx_ref = x_ref.grad.detach().clone()

        x_test = x.detach().clone().requires_grad_(True)
        out_test, scatter_index_test = triton_routing_scatter(x_test, token_expert_ids, config.num_experts)
        out_test.sum().backward()

        torch.testing.assert_close(out_test, out_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(scatter_index_test, scatter_index_ref, rtol=0, atol=0)
        torch.testing.assert_close(x_test.grad, dx_ref, rtol=1e-2, atol=1e-2)
        fw_ok = True
        bw_ok = True

    results = []

    x_base = x.detach().clone().requires_grad_(True)

    def run_base() -> torch.Tensor:
        out, _ = _torch_routing_scatter(x_base, token_expert_ids, config.num_experts)
        return out

    def reset_base_grads() -> None:
        x_base.grad = None

    base = benchmark_variant("baseline", run_base, reset_base_grads, warmup=config.warmup, iters=config.iters)
    base.fw_ok = fw_ok
    base.bw_ok = bw_ok
    results.append(base)

    x_triton = x.detach().clone().requires_grad_(True)

    def run_triton() -> torch.Tensor:
        out, _ = triton_routing_scatter(x_triton, token_expert_ids, config.num_experts)
        return out

    def reset_triton_grads() -> None:
        x_triton.grad = None

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
        "prime_moe_scatter",
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
