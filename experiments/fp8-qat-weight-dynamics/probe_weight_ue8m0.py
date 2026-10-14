"""Report which FP8 casts in the trainer's dense and grouped GEMMs use power-of-two scales, with and without PRIME_RL_FP8_WEIGHT_UE8M0.

Usage (on a GPU node, from the repo root):
    uv run --no-sync python experiments/fp8-qat-weight-dynamics/probe_weight_ue8m0.py
"""

import os
from collections import defaultdict

import torch

from prime_rl.trainer.models.layers import fp8_grouped_gemm, fp8_linear

CAST_FUNCTIONS = {
    fp8_linear: [
        "per_token_cast_to_fp8_triton",
        "per_token_cast_to_fp8_tp_triton",
        "per_block_cast_to_fp8_triton",
        "per_block_cast_to_fp8_tp_triton",
    ],
    fp8_grouped_gemm: [
        "grouped_per_token_cast_to_fp8_triton",
        "grouped_per_block_cast_to_fp8_triton",
        "grouped_per_channel_cast_to_fp8_sm90_kmajor_triton",
    ],
}


def is_power_of_two(scale: torch.Tensor) -> bool:
    positive = scale[scale > 0].float()
    return bool(torch.all(torch.log2(positive).frac() == 0))


def main() -> None:
    records: dict[str, list[bool]] = defaultdict(list)
    for module, names in CAST_FUNCTIONS.items():
        for name in names:
            original = getattr(module, name)

            def recorded(*args, _original=original, _label=f"{module.__name__.split('.')[-1]}.{name}", **kwargs):
                out, scale = _original(*args, **kwargs)
                records[_label].append(is_power_of_two(scale))
                return out, scale

            setattr(module, name, recorded)

    torch.manual_seed(0)
    x = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    linear_weight = (0.02 * torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    expert_weight = (0.02 * torch.randn(4, 2048, 768, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    offs = torch.tensor([128, 256, 384, 512], device="cuda", dtype=torch.int32)

    for flag in ("0", "1"):
        os.environ["PRIME_RL_FP8_WEIGHT_UE8M0"] = flag
        records.clear()
        fp8_linear._fp8_blockwise_mm(x, linear_weight, 128).float().square().mean().backward()
        fp8_grouped_gemm.grouped_fp8_gemm(x, expert_weight, offs).float().square().mean().backward()
        print(f"PRIME_RL_FP8_WEIGHT_UE8M0={flag}")
        for label, flags in sorted(records.items()):
            print(f"  {label:70s} calls={len(flags)} power_of_two={all(flags)}")


if __name__ == "__main__":
    main()
