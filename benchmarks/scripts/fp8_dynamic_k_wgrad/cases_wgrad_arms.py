import functools

import deep_gemm
import torch

from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_tp_triton

SHAPES = [(1024, 4096), (32768, 1024)]
TOKENS = [1024, 4096, 14336]
ARMS = {"main": ("nk", True), "change1": ("mn", True), "change2": ("mn", False)}


def cases():
    out = []
    for out_features, in_features in SHAPES:
        for num_tokens in TOKENS:
            grad_output = torch.randn(num_tokens, out_features, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(num_tokens, in_features, device="cuda", dtype=torch.bfloat16)
            a = per_token_cast_to_fp8_tp_triton(grad_output, False, 128)
            b = per_token_cast_to_fp8_tp_triton(x, False, 128)
            fns = {}
            for label, (compiled_dims, old_layout) in ARMS.items():
                a_arm = (a[0], a[1].contiguous()) if old_layout else a
                b_arm = (b[0], b[1].contiguous()) if old_layout else b
                d = torch.zeros(out_features, in_features, device="cuda", dtype=torch.float32)
                fns[label] = functools.partial(
                    deep_gemm.fp8_gemm_nt, a_arm, b_arm, d, c=d, recipe=(1, 1, 128), compiled_dims=compiled_dims
                )
            for fn in fns.values():
                fn()
            out.append((f"wgrad out={out_features} in={in_features} T={num_tokens}", fns))
    return out
