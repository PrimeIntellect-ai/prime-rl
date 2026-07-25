from prime_rl_kernels.nvfp4.grouped_gemm import grouped_gemm


def prepare_for_compile() -> None:
    """Load the CUDA operators and register FakeTensor shape implementations."""

    from prime_rl_kernels.nvfp4.grouped_gemm._extension import (
        _prepare_extension_for_compile as prepare_grouped_gemm,
    )
    from prime_rl_kernels.nvfp4.quantize._extension import (
        _prepare_extension_for_compile as prepare_quantization,
    )

    prepare_quantization()
    prepare_grouped_gemm()


__all__ = ["grouped_gemm", "prepare_for_compile"]
