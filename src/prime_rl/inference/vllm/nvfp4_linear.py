"""Online NVFP4 W4A4 linear layers with per-token activation scales.

vLLM 0.33 quantizes only routed experts to NVFP4 online (``nvfp4_per_token``). This adds the linear
counterpart: bf16 weights are quantized at load (128x4-swizzled E4M3 block scales, one FP32 global scale
per weight), activations per token at runtime, and FlashInfer's CuTe-DSL FP4 GEMM applies the per-row
dequantization scale. It is registered as the ``nvfp4`` online shorthand (NVFP4 linear and MoE), so
``quantization = "online"`` can route layers to it through ``quantization_config.targets``.
"""

import torch

NVFP4_ENCODE_MAX = 448.0 * 6.0  # E4M3 block-scale max x E2M1 max


@torch.library.custom_op("prime_rl::nvfp4_per_token_linear", mutates_args=())
def nvfp4_per_token_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_inv: torch.Tensor
) -> torch.Tensor:
    """``x [M, K] @ dequant(weight [N, K/2]).T`` with ``x`` quantized to NVFP4 per token."""
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE

    if x.shape[0] == 0:
        return x.new_empty(0, weight.shape[0])
    x_q, x_scale, x_token_scale = nvfp4_quantize(
        x, NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE, sfLayout=SfLayout.layout_128x4, per_token_activation=True
    )
    alpha = x_token_scale.reshape(-1) * weight_scale_inv
    return mm_fp4(x_q, weight.T, x_scale, weight_scale.T, alpha, x.dtype, backend="cute-dsl")


@nvfp4_per_token_linear.register_fake
def _(x, weight, weight_scale, weight_scale_inv):
    return x.new_empty(x.shape[0], weight.shape[0])


def quantize_weight_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(FP4 weight [N, K/2], 128x4 block scales [N_pad, K/16_pad], 1 / global scale [1])."""
    from flashinfer import SfLayout, nvfp4_quantize

    global_scale = (NVFP4_ENCODE_MAX / weight.abs().amax().float()).reshape(1)
    weight_q, weight_scale = nvfp4_quantize(weight, global_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    return weight_q, weight_scale.view(torch.float8_e4m3fn).reshape(-1, weight.shape[1] // 16), 1.0 / global_scale


def register_nvfp4_online_quantization():
    """Register the linear method and the ``nvfp4`` shorthand. Call before vLLM validates ``quantization_config``."""
    from vllm.config import quantization as quant_config
    from vllm.model_executor.layers.quantization.online import base as online_base
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4DynamicToken, kNvfp4Static
    from vllm.model_executor.utils import replace_parameter

    if "nvfp4" in quant_config._ONLINE_SHORTHANDS:
        return

    class Nvfp4OnlineLinearMethod(OnlineLinearBase):
        activation_quant_key = kNvfp4DynamicToken

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, *args, **kwargs):
            if input_size_per_partition % 256 != 0 or sum(output_partition_sizes) % 8 != 0:
                raise ValueError(
                    f"NVFP4 linear needs K % 256 == 0 and N % 8 == 0, got K={input_size_per_partition}, "
                    f"N={sum(output_partition_sizes)} ({layer.__class__.__name__})"
                )
            super().create_weights(layer, input_size_per_partition, output_partition_sizes, *args, **kwargs)

        def process_weights_after_loading(self, layer):
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            weight_q, weight_scale, weight_scale_inv = quantize_weight_nvfp4(
                self.get_weight_for_quantization(layer).contiguous()
            )
            layer.input_scale = None
            replace_parameter(layer, "weight", weight_q)
            replace_parameter(layer, "weight_scale", weight_scale)
            replace_parameter(layer, "weight_scale_inv", weight_scale_inv)
            self.release_requantization_source_weights(layer)
            layer._already_called_process_weights_after_loading = True

        def apply(self, layer, x, bias=None):
            out = nvfp4_per_token_linear(
                x.reshape(-1, x.shape[-1]), layer.weight, layer.weight_scale, layer.weight_scale_inv
            ).reshape(*x.shape[:-1], -1)
            return out if bias is None else out + bias

    online_base._ONLINE_LINEAR_METHODS[kNvfp4Static] = Nvfp4OnlineLinearMethod
    quant_config._ONLINE_SHORTHANDS["nvfp4"] = quant_config.QuantizationConfigArgs(
        linear=quant_config.QuantSpec(weight=kNvfp4Static), moe=quant_config.QuantSpec(weight=kNvfp4Static)
    )
