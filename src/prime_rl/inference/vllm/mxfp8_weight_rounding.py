"""Weight-only MXFP8 stochastic rounding for vLLM's online loader."""

import os

import torch


def cast_mxfp8_weight(x: torch.Tensor, is_sf_swizzled_layout: bool = False):
    import prime_kernels
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale

    seed = torch.empty((), device=x.device, dtype=torch.int64).random_()
    salt = torch.zeros((), device=x.device, dtype=torch.int64)
    quantized, scales = prime_kernels.load("mxfp8").quantize(x.data, rounding_seed=seed, rounding_salt=salt)
    if is_sf_swizzled_layout:
        scales = swizzle_mxfp8_scale(scales, *x.shape)
    return quantized, scales


def patch_mxfp8_weight_rounding():
    if os.environ.get("PRIME_MXFP8_STOCHASTIC_WEIGHT_ROUNDING") != "1":
        return

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.online import mxfp8

    if getattr(mxfp8.mxfp8_e4m3_quantize, "_prime_rounds_stochastically", False):
        return
    # The online module's binding is used only for dense/expert weight loading.
    cast_mxfp8_weight._prime_rounds_stochastically = True
    mxfp8.mxfp8_e4m3_quantize = cast_mxfp8_weight
    init_logger("vllm.prime_rl.mxfp8").info("Online MXFP8 weights use stochastic rounding; activations use nearest.")
