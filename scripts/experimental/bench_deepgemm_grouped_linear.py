import argparse
import os

import torch
import torch.nn as nn

from triton.testing import do_bench

from prime_rl.trainer.experimental.grouped_gemm.fp8_grouped_linear import FP8GroupedLinearDeepGEMM


def _set_cuda_device_from_env() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def _bench_ms(fn) -> float:
    ms = do_bench(fn)
    if ms is None:
        raise RuntimeError("do_bench returned None")
    if isinstance(ms, (tuple, list)):
        return float(ms[0])
    return float(ms)


def _diff_report(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a_d = a.detach()
    b_d = b.detach()
    d = (a_d.float() - b_d.float()).abs()
    denom = b_d.float().abs().max().clamp_min(1e-6)
    rel = (d.max() / denom).item()
    return float(rel), float(d.max().item()), float(d.mean().item())


def _layout_from_offsets(group_offsets: torch.Tensor) -> torch.Tensor:
    counts = torch.empty_like(group_offsets)
    counts[0] = group_offsets[0]
    if group_offsets.numel() > 1:
        counts[1:] = group_offsets[1:] - group_offsets[:-1]
    experts = torch.arange(
        group_offsets.numel(),
        device=group_offsets.device,
        dtype=torch.int32,
    )
    return torch.repeat_interleave(experts, counts.to(torch.long))


def _parse_tokens(tokens_csv: str, num_experts: int, tokens_per_expert: int) -> list[int]:
    if tokens_csv.strip() == "":
        return [tokens_per_expert for _ in range(num_experts)]
    vals = [int(x.strip()) for x in tokens_csv.split(",") if x.strip()]
    if len(vals) == 1:
        return vals * num_experts
    if len(vals) != num_experts:
        raise ValueError("--tokens must have exactly num_experts entries")
    return vals


class TorchGroupedLinearBF16(nn.Module):
    def __init__(self, num_experts: int, k: int, n: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, n, k, dtype=torch.bfloat16, device="cuda"))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, n, dtype=torch.bfloat16, device="cuda"))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        group_offsets: torch.Tensor,
        grouped_layout_long: torch.Tensor,
    ):
        out = torch._grouped_mm(
            x,
            self.weight.transpose(-2, -1),
            offs=group_offsets,
            out_dtype=torch.bfloat16,
        )
        if self.bias is not None:
            out = out + self.bias[grouped_layout_long]
        return out


def run(
    num_experts: int,
    k: int,
    n: int,
    tokens: list[int],
    bias: bool,
    run_correctness: bool,
):
    assert k % 128 == 0
    assert n % 128 == 0
    assert len(tokens) == num_experts
    assert all(t >= 0 for t in tokens)
    assert all(t % 128 == 0 for t in tokens)

    m = sum(tokens)
    group_offsets = torch.tensor(tokens, dtype=torch.int32, device="cuda").cumsum(dim=0).to(torch.int32)
    grouped_layout = _layout_from_offsets(group_offsets)
    grouped_layout_long = grouped_layout.to(torch.long)

    fp8_mod = FP8GroupedLinearDeepGEMM(num_experts, k, n, bias=bias)
    bf16_mod = TorchGroupedLinearBF16(num_experts, k, n, bias=bias)
    bf16_mod.weight.data.copy_(fp8_mod.weight.data)
    if bias:
        bf16_mod.bias.data.copy_(fp8_mod.bias.data)

    print(f"Grouped shape: E={num_experts}, M={m}, K={k}, N={n}, bias={int(bias)}")
    print(f"Tokens/expert: {tokens}")
    print("BF16 baseline: torch._grouped_mm (torch.functional.grouped_mm equivalent)")
    print()

    if run_correctness:
        x_ref = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_fp8 = x_ref.detach().clone().requires_grad_(True)
        grad_out = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") / 1e2

        bf16_mod.zero_grad(set_to_none=True)
        fp8_mod.zero_grad(set_to_none=True)

        y_ref = bf16_mod(x_ref, group_offsets, grouped_layout_long)
        y_fp8 = fp8_mod(x_fp8, group_offsets=group_offsets)
        y_ref.backward(grad_out)
        y_fp8.backward(grad_out)

        assert x_ref.grad is not None and x_fp8.grad is not None
        assert bf16_mod.weight.grad is not None and fp8_mod.weight.grad is not None

        f_diff, f_max, f_mean = _diff_report(y_fp8, y_ref)
        gi_diff, gi_max, gi_mean = _diff_report(x_fp8.grad, x_ref.grad)
        gw_diff, gw_max, gw_mean = _diff_report(fp8_mod.weight.grad, bf16_mod.weight.grad)

        print("Correctness")
        print(f"  Forward:     diff={f_diff:.6f}, max={f_max:.6f}, mean={f_mean:.6f}")
        print(f"  Grad input:  diff={gi_diff:.6f}, max={gi_max:.6f}, mean={gi_mean:.6f}")
        print(f"  Grad weight: diff={gw_diff:.6f}, max={gw_max:.6f}, mean={gw_mean:.6f}")
        if bias:
            assert bf16_mod.bias is not None and bf16_mod.bias.grad is not None
            assert fp8_mod.bias is not None and fp8_mod.bias.grad is not None
            gb_diff, gb_max, gb_mean = _diff_report(fp8_mod.bias.grad, bf16_mod.bias.grad)
            print(f"  Grad bias:   diff={gb_diff:.6f}, max={gb_max:.6f}, mean={gb_mean:.6f}")
        print()

        del x_ref, x_fp8, grad_out, y_ref, y_fp8
        torch.cuda.empty_cache()

    x_fwd = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        fp8_mod.prepare_prequantized_weight()

    def bf16_fwd():
        with torch.no_grad():
            return bf16_mod(x_fwd, group_offsets, grouped_layout_long)

    def fp8_fwd_prequant():
        with torch.no_grad():
            return fp8_mod(x_fwd, group_offsets=group_offsets, quantize_weight_on_the_fly=False)

    def fp8_fwd_onthefly():
        with torch.no_grad():
            return fp8_mod(x_fwd, group_offsets=group_offsets, quantize_weight_on_the_fly=True)

    x_ref_train = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x_fp8_train = x_ref_train.detach().clone().requires_grad_(True)
    grad_out_train = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    def bf16_fwd_bwd():
        bf16_mod.zero_grad(set_to_none=True)
        if x_ref_train.grad is not None:
            x_ref_train.grad = None
        y = bf16_mod(x_ref_train, group_offsets, grouped_layout_long)
        torch.autograd.backward(y, grad_out_train)

    def _invalidate_fp8_weight_cache():
        fp8_mod._w_q_version = -1
        fp8_mod._w_t_q_version = -1

    def fp8_fwd_bwd_requant_each_iter():
        fp8_mod.zero_grad(set_to_none=True)
        if x_fp8_train.grad is not None:
            x_fp8_train.grad = None
        _invalidate_fp8_weight_cache()
        y = fp8_mod(x_fp8_train, group_offsets=group_offsets)
        torch.autograd.backward(y, grad_out_train)

    with torch.no_grad():
        _ = bf16_fwd()
        _ = fp8_fwd_prequant()
        _ = fp8_fwd_onthefly()
    bf16_fwd_bwd()
    fp8_fwd_bwd_requant_each_iter()
    torch.cuda.synchronize()

    bf16_fwd_ms = _bench_ms(bf16_fwd)
    fp8_pre_ms = _bench_ms(fp8_fwd_prequant)
    fp8_otf_ms = _bench_ms(fp8_fwd_onthefly)
    bf16_fwbw_ms = _bench_ms(bf16_fwd_bwd)
    fp8_fwbw_requant_ms = _bench_ms(fp8_fwd_bwd_requant_each_iter)

    print()
    print("Performance")
    print(f"  BF16 grouped forward:          {bf16_fwd_ms:.3f} ms")
    print(f"  FP8 grouped forward (prequant):{fp8_pre_ms:.3f} ms")
    print(f"  FP8 grouped forward (on-fly):  {fp8_otf_ms:.3f} ms")
    print(f"  BF16 grouped forward+backward: {bf16_fwbw_ms:.3f} ms")
    print(f"  FP8 grouped forward+backward (requant/iter): {fp8_fwbw_requant_ms:.3f} ms")
    print(f"  FWD speedup (prequant):        {bf16_fwd_ms / fp8_pre_ms:.3f}x")
    print(f"  FWD speedup (on-fly):          {bf16_fwd_ms / fp8_otf_ms:.3f}x")
    print(f"  FWD+BWD speedup (requant/iter): {bf16_fwbw_ms / fp8_fwbw_requant_ms:.3f}x")


if __name__ == "__main__":
    _set_cuda_device_from_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--tokens-per-expert", type=int, default=2048)
    parser.add_argument("--tokens", type=str, default="")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    tokens = _parse_tokens(args.tokens, args.experts, args.tokens_per_expert)
    run(
        args.experts,
        args.k,
        args.n,
        tokens,
        args.bias,
        run_correctness=not args.skip_correctness,
    )
