#!/usr/bin/env python3
"""Benchmark Prime-RL routed-expert implementations."""

from __future__ import annotations

from typing import Annotated, Literal

import torch
from pydantic import Field

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
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.moe_backends import MoEBackendSelection, MoEBackends


class RoutedExpertsBenchmarkConfig(BaseMoEBenchmarkConfig):
    """Benchmark the Prime-RL routed-expert path."""

    hidden: int = 4096
    num_experts: int = 36
    topk: int = 2
    warmup: int = 20
    ffn_hidden: Annotated[int, Field(ge=1, description="Per-expert FFN hidden size")] = 1280
    use_grouped_mm: Annotated[bool, Field(description="Use torch._grouped_mm experts path")] = True
    score_before_experts: Annotated[bool, Field(description="Apply expert weights before expert compute")] = False
    grouped_ffn_backend: Annotated[
        Literal["torch", "triton"],
        Field(description="Backend to use for the candidate grouped_ffn path"),
    ] = "torch"
    invalid_frac: float = 0.1


def _build_experts(
    hidden: int,
    ffn_hidden: int,
    num_experts: int,
    use_grouped_mm: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> GroupedExperts:
    experts = GroupedExperts(
        dim=hidden,
        hidden_dim=ffn_hidden,
        num_experts=num_experts,
        use_grouped_mm=use_grouped_mm,
    ).to(device=device)
    experts.init_weights(init_std=0.02)
    return experts.to(dtype=dtype)


def _clone_experts(reference: GroupedExperts, dtype: torch.dtype) -> GroupedExperts:
    experts = GroupedExperts(
        dim=reference.w1.shape[-1],
        hidden_dim=reference.w1.shape[1],
        num_experts=reference.num_experts,
        use_grouped_mm=reference.use_grouped_mm,
    ).to(device=reference.w1.device)
    experts.load_state_dict(reference.state_dict())
    return experts.to(dtype=dtype)


def _run_backend(
    backends: MoEBackends,
    x: torch.Tensor,
    top_scores: torch.Tensor,
    token_expert_ids: torch.Tensor,
    experts: GroupedExperts,
    score_before_experts: bool,
) -> torch.Tensor:
    if backends.use_grouped_ffn(x):
        return backends.run_grouped_ffn(
            x=x,
            top_scores=top_scores,
            selected_experts_indices=token_expert_ids,
            experts=experts,
            score_before_experts=score_before_experts,
        )
    return backends.run_routed_experts(
        x=x,
        top_scores=top_scores,
        selected_experts_indices=token_expert_ids,
        experts=experts,
        score_before_experts=score_before_experts,
    )


def main() -> None:
    config = cli(RoutedExpertsBenchmarkConfig)
    device = require_cuda()
    if (
        moe_backends_mod.triton_histogram is None
        or moe_backends_mod.triton_index_scatter is None
        or moe_backends_mod.triton_moe_weighted_gather is None
    ):
        raise RuntimeError(f"Triton MoE kernels are unavailable: {moe_backends_mod._TRITON_IMPORT_ERROR}")

    torch.manual_seed(0)
    dtype = resolve_dtype(config.dtype)
    x = torch.randn((config.token_num, config.hidden), device=device, dtype=dtype)
    token_expert_ids = make_topk_ids(
        config.token_num,
        config.topk,
        config.num_experts,
        device,
        allow_dupe=config.allow_dupe,
        invalid_frac=config.invalid_frac,
    )
    top_scores = make_token_weights(config.token_num, config.topk, device)

    experts_template = _build_experts(
        hidden=config.hidden,
        ffn_hidden=config.ffn_hidden,
        num_experts=config.num_experts,
        use_grouped_mm=config.use_grouped_mm,
        device=device,
        dtype=dtype,
    )

    torch_backends = MoEBackends(
        MoEBackendSelection(routing="torch", scatter="torch", gather="torch", grouped_ffn="torch"),
        num_experts=config.num_experts,
        top_k=config.topk,
    )
    triton_backends = MoEBackends(
        MoEBackendSelection(
            routing="triton",
            scatter="triton",
            gather="triton",
            grouped_ffn=config.grouped_ffn_backend,
        ),
        num_experts=config.num_experts,
        top_k=config.topk,
    )

    fw_ok = None
    bw_ok = None
    if config.check:
        experts_ref = _clone_experts(experts_template, dtype)
        x_ref = x.detach().clone().requires_grad_(True)
        top_scores_ref = top_scores.detach().clone().requires_grad_(True)
        out_ref = _run_backend(
            torch_backends,
            x_ref,
            top_scores_ref,
            token_expert_ids,
            experts_ref,
            config.score_before_experts,
        )
        out_ref.sum().backward()

        experts_test = _clone_experts(experts_template, dtype)
        x_test = x.detach().clone().requires_grad_(True)
        top_scores_test = top_scores.detach().clone().requires_grad_(True)
        out_test = _run_backend(
            triton_backends,
            x_test,
            top_scores_test,
            token_expert_ids,
            experts_test,
            config.score_before_experts,
        )
        out_test.sum().backward()

        torch.testing.assert_close(out_test, out_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(x_test.grad, x_ref.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(top_scores_test.grad, top_scores_ref.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(experts_test.w1.grad, experts_ref.w1.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(experts_test.w2.grad, experts_ref.w2.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(experts_test.w3.grad, experts_ref.w3.grad, rtol=2e-2, atol=2e-2)
        fw_ok = True
        bw_ok = True

    results = []

    experts_base = _clone_experts(experts_template, dtype)
    x_base = x.detach().clone().requires_grad_(True)
    top_scores_base = top_scores.detach().clone().requires_grad_(True)

    def run_base() -> torch.Tensor:
        return _run_backend(
            torch_backends,
            x_base,
            top_scores_base,
            token_expert_ids,
            experts_base,
            config.score_before_experts,
        )

    def reset_base_grads() -> None:
        x_base.grad = None
        top_scores_base.grad = None
        experts_base.zero_grad(set_to_none=True)

    base = benchmark_variant("baseline", run_base, reset_base_grads, warmup=config.warmup, iters=config.iters)
    base.fw_ok = fw_ok
    base.bw_ok = bw_ok
    results.append(base)

    experts_triton = _clone_experts(experts_template, dtype)
    x_triton = x.detach().clone().requires_grad_(True)
    top_scores_triton = top_scores.detach().clone().requires_grad_(True)
    candidate_name = "triton_grouped_ffn" if config.grouped_ffn_backend == "triton" else "triton_dispatch"

    def run_triton() -> torch.Tensor:
        return _run_backend(
            triton_backends,
            x_triton,
            top_scores_triton,
            token_expert_ids,
            experts_triton,
            config.score_before_experts,
        )

    def reset_triton_grads() -> None:
        x_triton.grad = None
        top_scores_triton.grad = None
        experts_triton.zero_grad(set_to_none=True)

    triton_result = benchmark_variant(
        candidate_name,
        run_triton,
        reset_triton_grads,
        warmup=config.warmup,
        iters=config.iters,
    )
    triton_result.fw_ok = fw_ok
    triton_result.bw_ok = bw_ok
    results.append(triton_result)

    print_result_table(
        "prime_moe_routed_experts",
        {
            "token_num": config.token_num,
            "hidden": config.hidden,
            "ffn_hidden": config.ffn_hidden,
            "topk": config.topk,
            "experts": config.num_experts,
            "grouped_mm": config.use_grouped_mm,
            "grouped_ffn": config.grouped_ffn_backend,
            "invalid_frac": config.invalid_frac,
        },
        results,
        include_checks=config.check,
    )


if __name__ == "__main__":
    main()
