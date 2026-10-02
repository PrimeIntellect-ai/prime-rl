"""Single-GPU probe asking whether sm90 DeepGEMM accepts an N-major fp8 B operand in the dx GEMM.

CUDA_VISIBLE_DEVICES=7 DG_JIT_CACHE_DIR=/home/garrett/.cache/jit/research_deep_gemm \
    uv run python -m prime_rl.experimental.fully_shard_caching.nn_contiguous_probe
"""

from __future__ import annotations

import inspect

import torch

from prime_rl.experimental.fully_shard_caching.fp8_cast import grouped_per_block_cast_to_fp8
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_token_cast_to_fp8_triton,
    ue8m0_for_device,
    unpack_rows_triton,
)

EXPERTS = 8
OUT_FEATURES = 2816
IN_FEATURES = 4096
TOKENS_PER_EXPERT = 256
AGREE_FRACTION = 1e-2


def describe(name: str, function) -> None:
    print(f"{name}: repr {function!r}")
    try:
        print(f"{name}: signature {inspect.signature(function)}")
    except (TypeError, ValueError) as error:
        print(f"{name}: signature unavailable, {type(error).__name__}: {error}")
    print(f"{name}: doc {function.__doc__}")


def diffs(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, float]:
    difference = (candidate.float() - reference.float()).abs()
    max_abs = difference.max().item()
    max_rel = (difference / reference.float().abs().clamp_min(1e-6)).max().item()
    return max_abs, max_rel, max_abs / reference.float().abs().max().item()


def classify(label: str, call, reference: torch.Tensor, exact: torch.Tensor) -> Exception | None:
    try:
        result = call()
    except Exception as error:
        print(f"{label}: RAISES {type(error).__name__}")
        for line in str(error).splitlines():
            print(f"{label}:   {line}")
        return error
    max_abs, max_rel, normalized = diffs(result, reference)
    _, _, exact_normalized = diffs(result, exact)
    verdict = "RUNS AND AGREES" if normalized < AGREE_FRACTION else "RUNS BUT DISAGREES"
    print(
        f"{label}: {verdict}, vs reference max abs {max_abs:.6g}, max rel {max_rel:.6g}, "
        f"normalized {normalized:.6g}; vs bf16 exact normalized {exact_normalized:.6g}"
    )
    return None


def main() -> None:
    import deep_gemm

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    use_ue8m0 = ue8m0_for_device(device)
    print(f"device {torch.cuda.get_device_name(device)}, capability {capability}, use_ue8m0 {use_ue8m0}")
    print(f"shape experts {EXPERTS}, out_features {OUT_FEATURES}, in_features {IN_FEATURES}")
    print(f"tokens per expert {TOKENS_PER_EXPERT}, total m {EXPERTS * TOKENS_PER_EXPERT}")

    describe("nn_contiguous", deep_gemm.m_grouped_fp8_gemm_nn_contiguous)
    describe("nt_contiguous", deep_gemm.m_grouped_fp8_gemm_nt_contiguous)
    describe("transform_sf", deep_gemm.transform_sf_into_required_layout)

    torch.manual_seed(0)
    weight = torch.randn(EXPERTS, OUT_FEATURES, IN_FEATURES, device=device, dtype=torch.bfloat16).mul_(0.02)
    grad_output = torch.randn(EXPERTS * TOKENS_PER_EXPERT, OUT_FEATURES, device=device, dtype=torch.bfloat16)
    offs = torch.arange(1, EXPERTS + 1, device=device, dtype=torch.int32) * TOKENS_PER_EXPERT

    qdata, scales = grouped_per_block_cast_to_fp8(weight, use_ue8m0)
    qdata_t, scales_t = grouped_per_block_cast_to_fp8(weight.transpose(1, 2), use_ue8m0)
    print(
        f"qdata {tuple(qdata.shape)} strides {qdata.stride()}, scales {tuple(scales.shape)} strides {scales.stride()}"
    )
    print(
        f"qdata_t {tuple(qdata_t.shape)} strides {qdata_t.stride()}, "
        f"scales_t {tuple(scales_t.shape)} strides {scales_t.stride()}"
    )

    (
        total_m,
        padded_total_m,
        grouped_layout,
        block_to_group,
        _,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = build_grouped_layout(offs, total_m=grad_output.size(0))
    dy_fp8 = grouped_per_token_cast_to_fp8_triton(
        grad_output,
        padded_total_m,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        use_ue8m0,
        GROUP_ALIGNMENT,
    )

    def run(gemm, b_pair, **kwargs) -> torch.Tensor:
        out_padded = torch.empty((padded_total_m, IN_FEATURES), device=device, dtype=grad_output.dtype)
        gemm(dy_fp8, b_pair, out_padded, grouped_layout, **kwargs)
        return unpack_rows_triton(
            out_padded, total_m, block_to_group, starts_tensor, actual_ms_tensor, block_starts_tensor
        )

    reference = run(deep_gemm.m_grouped_fp8_gemm_nt_contiguous, (qdata_t, scales_t), use_psum_layout=False)
    exact = torch.bmm(grad_output.float().view(EXPERTS, TOKENS_PER_EXPERT, OUT_FEATURES), weight.float()).reshape(
        total_m, IN_FEATURES
    )
    _, _, reference_normalized = diffs(reference, exact)
    print(f"reference nt(qdata_t): vs bf16 exact normalized {reference_normalized:.6g}")

    errors = {}
    errors["control nn(qdata_t.mT)"] = classify(
        "control nn(qdata_t.mT)",
        lambda: run(
            deep_gemm.m_grouped_fp8_gemm_nn_contiguous,
            (qdata_t.transpose(1, 2), scales_t.transpose(1, 2)),
            use_psum_layout=False,
        ),
        reference,
        exact,
    )
    for compiled_dims in ("nk", "mn", "mk", "kn", "n", "k", ""):
        label = f"route A nn(qdata) compiled_dims {compiled_dims!r}"
        errors[label] = classify(
            label,
            lambda dims=compiled_dims: run(
                deep_gemm.m_grouped_fp8_gemm_nn_contiguous,
                (qdata, scales),
                compiled_dims=dims,
                use_psum_layout=False,
            ),
            reference,
            exact,
        )
    errors["route A nn(qdata) no kwargs"] = classify(
        "route A nn(qdata) no kwargs",
        lambda: run(deep_gemm.m_grouped_fp8_gemm_nn_contiguous, (qdata, scales)),
        reference,
        exact,
    )
    for compiled_dims in ("nk", "mn", "mk"):
        label = f"route B nt(qdata.mT) compiled_dims {compiled_dims!r}"
        errors[label] = classify(
            label,
            lambda dims=compiled_dims: run(
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous,
                (qdata.transpose(1, 2), scales.transpose(1, 2)),
                compiled_dims=dims,
                use_psum_layout=False,
            ),
            reference,
            exact,
        )

    def transform(label: str, sf: torch.Tensor, **extra) -> torch.Tensor | None:
        try:
            transformed = deep_gemm.transform_sf_into_required_layout(
                sf,
                IN_FEATURES,
                OUT_FEATURES,
                (GROUP_ALIGNMENT, GROUP_ALIGNMENT),
                num_groups=EXPERTS,
                disable_ue8m0_cast=not use_ue8m0,
                **extra,
            )
        except Exception as error:
            print(f"{label}: RAISES {type(error).__name__}")
            for line in str(error).splitlines():
                print(f"{label}:   {line}")
            return None
        print(f"{label}: {tuple(transformed.shape)} strides {transformed.stride()} dtype {transformed.dtype}")
        return transformed

    nn_scales = transform("transform_sf for nn", scales)
    if nn_scales is not None:
        errors["route A nn(qdata) transformed sf"] = classify(
            "route A nn(qdata) transformed sf",
            lambda: run(deep_gemm.m_grouped_fp8_gemm_nn_contiguous, (qdata, nn_scales), use_psum_layout=False),
            reference,
            exact,
        )

    nt_scales = transform("transform_sf for nt.mT", scales.transpose(1, 2))
    if nt_scales is not None:
        errors["route B nt(qdata.mT) transformed sf"] = classify(
            "route B nt(qdata.mT) transformed sf",
            lambda: run(
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous,
                (qdata.transpose(1, 2), nt_scales),
                use_psum_layout=False,
            ),
            reference,
            exact,
        )

    print("failure attribution")
    for label, error in errors.items():
        if error is None:
            print(f"  {label}: no exception")
            continue
        text = str(error).lower()
        mentions_major = "major" in text
        mentions_scale = "sf" in text or "scale" in text
        print(f"  {label}: mentions majorness {mentions_major}, mentions scale factor {mentions_scale}")


if __name__ == "__main__":
    main()
