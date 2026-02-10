import argparse
import os

import torch
import torch.nn as nn

from triton.testing import do_bench

from deep_gemm.testing import calc_diff
from prime_rl.trainer.experimental.grouped_gemm.fp8_grouped_linear import FP8LinearDeepGEMM


def _set_cuda_device_from_env() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def _diff_report(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a_d = a.detach()
    b_d = b.detach()
    d = (a_d.float() - b_d.float()).abs()
    return float(calc_diff(a_d, b_d)), float(d.max().item()), float(d.mean().item())


def _bench_ms(fn) -> float:
    ms = do_bench(fn)
    if ms is None:
        raise RuntimeError("do_bench returned None")
    if isinstance(ms, (tuple, list)):
        return float(ms[0])
    return float(ms)


def _run_correctness(
    m: int,
    k: int,
    n: int,
    bias: bool,
    fp8_mod: FP8LinearDeepGEMM,
    bf16_mod: nn.Linear,
):
    x_ref = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x_fp8 = x_ref.detach().clone().requires_grad_(True)
    grad_out = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") / 1e2

    bf16_mod.zero_grad(set_to_none=True)
    fp8_mod.zero_grad(set_to_none=True)

    y_ref = bf16_mod(x_ref)
    y_fp8 = fp8_mod(x_fp8)
    y_ref.backward(grad_out)
    y_fp8.backward(grad_out)

    assert x_ref.grad is not None
    assert x_fp8.grad is not None
    assert bf16_mod.weight.grad is not None
    assert fp8_mod.weight.grad is not None

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


def _run_bench(
    m: int,
    k: int,
    n: int,
    bias: bool,
    fp8_mod: FP8LinearDeepGEMM,
    bf16_mod: nn.Linear,
):
    x_fwd = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        fp8_mod.prepare_prequantized_weight()

    def bf16_fwd():
        return bf16_mod(x_fwd)

    def fp8_fwd_prequant():
        return fp8_mod(x_fwd, quantize_weight_on_the_fly=False)

    def fp8_fwd_onthefly():
        return fp8_mod(x_fwd, quantize_weight_on_the_fly=True)

    with torch.no_grad():
        _ = bf16_fwd()
        _ = fp8_fwd_prequant()
        _ = fp8_fwd_onthefly()
    torch.cuda.synchronize()

    x_ref_train = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x_fp8_train = x_ref_train.detach().clone().requires_grad_(True)
    grad_out = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    def bf16_fwd_bwd():
        bf16_mod.zero_grad(set_to_none=True)
        if x_ref_train.grad is not None:
            x_ref_train.grad = None
        y = bf16_mod(x_ref_train)
        torch.autograd.backward(y, grad_out)

    def _invalidate_fp8_weight_cache():
        # Simulate post-optimizer-step state where weight changed and must be re-quantized.
        fp8_mod._w_q_version = -1
        fp8_mod._w_t_q_version = -1

    def fp8_fwd_bwd_requant_each_iter():
        fp8_mod.zero_grad(set_to_none=True)
        if x_fp8_train.grad is not None:
            x_fp8_train.grad = None
        _invalidate_fp8_weight_cache()
        y = fp8_mod(x_fp8_train)
        torch.autograd.backward(y, grad_out)

    bf16_fwd_ms = _bench_ms(bf16_fwd)
    fp8_fwd_pre_ms = _bench_ms(fp8_fwd_prequant)
    fp8_fwd_otf_ms = _bench_ms(fp8_fwd_onthefly)
    bf16_fwbw_ms = _bench_ms(bf16_fwd_bwd)
    fp8_fwbw_requant_ms = _bench_ms(fp8_fwd_bwd_requant_each_iter)

    print()
    print("Performance")
    print(f"  BF16 forward:               {bf16_fwd_ms:.3f} ms")
    print(f"  FP8 forward (prequant):     {fp8_fwd_pre_ms:.3f} ms")
    print(f"  FP8 forward (on-the-fly):   {fp8_fwd_otf_ms:.3f} ms")
    print(f"  BF16 forward+backward:      {bf16_fwbw_ms:.3f} ms")
    print(f"  FP8 forward+backward (requant/iter): {fp8_fwbw_requant_ms:.3f} ms")
    print(f"  FWD speedup (prequant):     {bf16_fwd_ms / fp8_fwd_pre_ms:.3f}x")
    print(f"  FWD speedup (on-the-fly):   {bf16_fwd_ms / fp8_fwd_otf_ms:.3f}x")
    print(f"  FWD+BWD speedup (requant/iter): {bf16_fwbw_ms / fp8_fwbw_requant_ms:.3f}x")


def run(shape: tuple[int, int, int], bias: bool):
    m, k, n = shape
    assert m % 4 == 0
    assert k % 128 == 0
    assert n % 128 == 0

    fp8_mod = FP8LinearDeepGEMM(k, n, bias=bias)
    bf16_mod = nn.Linear(k, n, bias=bias, dtype=torch.bfloat16, device="cuda")
    bf16_mod.weight.data.copy_(fp8_mod.weight.data)
    if bias:
        bf16_mod.bias.data.copy_(fp8_mod.bias.data)

    print(f"Shape: M={m}, K={k}, N={n}, bias={int(bias)}")
    print()

    _run_correctness(m, k, n, bias, fp8_mod, bf16_mod)
    _run_bench(m, k, n, bias, fp8_mod, bf16_mod)


if __name__ == "__main__":
    _set_cuda_device_from_env()

    def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--mkn", type=parse_comma_separated_ints, default=(16384, 4096, 4096))
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()
    run(args.mkn, args.bias)
