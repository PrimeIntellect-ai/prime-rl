import argparse
import os
import sys

import torch

SHAPES = [(1024, 4096), (32768, 1024)]


def wgrad(grad_output_2d, x_2d, out_features, in_features, compiled_dims):
    import deep_gemm

    from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_tp_triton, ue8m0_for_device

    use_ue8m0 = ue8m0_for_device(grad_output_2d.device)
    num_tokens = grad_output_2d.size(0)
    padded_tokens = (num_tokens + 127) // 128 * 128
    if padded_tokens != num_tokens:
        pad_rows = padded_tokens - num_tokens
        grad_output_2d = torch.nn.functional.pad(grad_output_2d, (0, 0, 0, pad_rows))
        x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad_rows))
    grad_output_t_fp8 = per_token_cast_to_fp8_tp_triton(grad_output_2d, use_ue8m0, 128)
    x_t_fp8 = per_token_cast_to_fp8_tp_triton(x_2d, use_ue8m0, 128)
    grad_weight_fp32 = torch.zeros((out_features, in_features), device=x_2d.device, dtype=torch.float32)
    kwargs = {} if compiled_dims is None else {"compiled_dims": compiled_dims}
    deep_gemm.fp8_gemm_nt(grad_output_t_fp8, x_t_fp8, grad_weight_fp32, c=grad_weight_fp32, recipe=(1, 1, 128), **kwargs)
    return grad_weight_fp32.to(torch.bfloat16)


def inputs(out_features, in_features, num_tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    grad_output = torch.randn(num_tokens, out_features, device="cuda", dtype=torch.bfloat16, generator=g)
    x = torch.randn(num_tokens, in_features, device="cuda", dtype=torch.bfloat16, generator=g)
    return grad_output, x


def token_counts():
    return list(range(128, 4097, 128)) + [1, 77, 1000, 3001, 4095]


def compile_mode(args):
    compiled_dims = None if args.arm == "baseline" else "mn"
    for out_features, in_features in SHAPES:
        for num_tokens in token_counts():
            grad_output, x = inputs(out_features, in_features, num_tokens, 0)
            wgrad(grad_output, x, out_features, in_features, compiled_dims)
    torch.cuda.synchronize()
    print("done", args.arm)


def bitwise_mode(args):
    from prime_rl.trainer.models.layers.fp8_linear import _fp8_blockwise_mm_backward

    all_equal = True
    for out_features, in_features in SHAPES:
        for num_tokens in [1, 77, 128, 1000, 1024, 3001, 4096, 14336]:
            grad_output, x = inputs(out_features, in_features, num_tokens, num_tokens)
            weight = torch.empty(out_features, in_features, device="cuda", dtype=torch.bfloat16)
            base = wgrad(grad_output, x, out_features, in_features, None)
            patched = wgrad(grad_output, x, out_features, in_features, "mn")
            _, repo = _fp8_blockwise_mm_backward(grad_output, x, weight, 128, False, True)
            eq_patched = torch.equal(base.view(torch.int16), patched.view(torch.int16))
            eq_repo = torch.equal(base.view(torch.int16), repo.view(torch.int16))
            all_equal &= eq_patched and eq_repo
            print(f"shape=({out_features},{in_features}) T={num_tokens} baseline==mn:{eq_patched} baseline==repo_op:{eq_repo}")
    print("ALL_BITWISE_EQUAL", all_equal)
    sys.exit(0 if all_equal else 1)


def ops_mode(args):
    import deep_gemm

    from prime_rl.trainer.models.layers.fp8_linear import _fp8_blockwise_bmm_backward, _fp8_blockwise_mm_backward

    patched_gemm = deep_gemm.fp8_gemm_nt

    def baseline_gemm(*a, compiled_dims=None, **kw):
        return patched_gemm(*a, **kw)

    def run(op, gemm, *op_args):
        deep_gemm.fp8_gemm_nt = gemm
        try:
            return op(*op_args)[1]
        finally:
            deep_gemm.fp8_gemm_nt = patched_gemm

    all_equal = True
    for num_tokens in [1, 77, 1000, 1024, 3001, 4096, 14336]:
        for out_features, in_features in SHAPES:
            grad_output, x = inputs(out_features, in_features, num_tokens, num_tokens)
            weight = torch.empty(out_features, in_features, device="cuda", dtype=torch.bfloat16)
            op_args = (grad_output, x, weight, 128, False, True)
            eq = torch.equal(*(run(_fp8_blockwise_mm_backward, g, *op_args).view(torch.int16) for g in (baseline_gemm, patched_gemm)))
            all_equal &= eq
            print(f"dense shape=({out_features},{in_features}) T={num_tokens} bitwise:{eq}")
        n_groups, out_per_group, in_per_group = 8, 1024, 4096
        g = torch.Generator(device="cuda").manual_seed(num_tokens)
        x = torch.randn(num_tokens, n_groups, in_per_group, device="cuda", dtype=torch.bfloat16, generator=g)
        grad_output = torch.randn(num_tokens, n_groups, out_per_group, device="cuda", dtype=torch.bfloat16, generator=g)
        weight = torch.empty(n_groups * out_per_group, in_per_group, device="cuda", dtype=torch.bfloat16)
        op_args = (grad_output, x, weight, n_groups, 128, False, True)
        eq = torch.equal(*(run(_fp8_blockwise_bmm_backward, g, *op_args).view(torch.int16) for g in (baseline_gemm, patched_gemm)))
        all_equal &= eq
        print(f"o_a_proj groups=8 weight=(8192,4096) T={num_tokens} bitwise:{eq}")
    print("ALL_OPS_BITWISE_EQUAL", all_equal)
    sys.exit(0 if all_equal else 1)


def layout_mode(args):
    import deep_gemm

    from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_tp_triton, ue8m0_for_device

    all_equal = True
    for out_features, in_features in SHAPES:
        for num_tokens in [128, 1024, 3072, 4096, 14336]:
            grad_output, x = inputs(out_features, in_features, num_tokens, num_tokens)
            use_ue8m0 = ue8m0_for_device(x.device)
            a = per_token_cast_to_fp8_tp_triton(grad_output, use_ue8m0, 128)
            b = per_token_cast_to_fp8_tp_triton(x, use_ue8m0, 128)
            results = []
            for a_sf, b_sf in [(a[1], b[1]), (a[1].contiguous(), b[1].contiguous())]:
                d = torch.zeros(out_features, in_features, device="cuda", dtype=torch.float32)
                deep_gemm.fp8_gemm_nt((a[0], a_sf), (b[0], b_sf), d, c=d, recipe=(1, 1, 128), compiled_dims="mn")
                results.append(d)
            eq = torch.equal(results[0].view(torch.int32), results[1].view(torch.int32))
            in_place = all(deep_gemm.get_mn_major_tma_aligned_tensor(sf).data_ptr() == sf.data_ptr() for sf in (a[1], b[1]))
            all_equal &= eq
            print(f"shape=({out_features},{in_features}) T={num_tokens} sf_stride={tuple(a[1].stride())} new==old_layout:{eq} helper_no_copy:{in_place}")
    print("ALL_LAYOUT_EQUAL", all_equal)
    sys.exit(0 if all_equal else 1)


ARMS = {"main": ("nk", True), "change1": ("mn", True), "change2": ("mn", False)}


def sweep_mode(args):
    """Time each fp8_gemm_nt call over every (shape, T) twice, cold cache then warm, with CUDA events."""
    import random

    import deep_gemm

    from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_tp_triton, ue8m0_for_device

    compiled_dims, old_layout = ARMS[args.arm]
    cases = [(shape, t) for shape in SHAPES for t in range(128, 4097, 128)]
    random.Random(0).shuffle(cases)
    for pass_name in ["cold", "warm"]:
        times = []
        for (out_features, in_features), num_tokens in cases:
            grad_output, x = inputs(out_features, in_features, num_tokens, num_tokens)
            use_ue8m0 = ue8m0_for_device(x.device)
            a = per_token_cast_to_fp8_tp_triton(grad_output, use_ue8m0, 128)
            b = per_token_cast_to_fp8_tp_triton(x, use_ue8m0, 128)
            if old_layout:
                a, b = (a[0], a[1].contiguous()), (b[0], b[1].contiguous())
            d = torch.zeros(out_features, in_features, device="cuda", dtype=torch.float32)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            deep_gemm.fp8_gemm_nt(a, b, d, c=d, recipe=(1, 1, 128), compiled_dims=compiled_dims)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        print(
            f"SWEEP arm={args.arm} pass={pass_name} calls={len(times)} total_ms={sum(times):.1f} "
            f"median_ms={times[len(times) // 2]:.3f} max_ms={times[-1]:.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "bitwise", "ops", "layout", "sweep"])
    parser.add_argument("--arm", choices=["baseline", "patched", "main", "change1", "change2"], default="patched")
    args = parser.parse_args()
    print("DG_JIT_CACHE_DIR", os.environ.get("DG_JIT_CACHE_DIR"))
    {"compile": compile_mode, "bitwise": bitwise_mode, "ops": ops_mode, "layout": layout_mode, "sweep": sweep_mode}[args.mode](args)
