"""Experimental online FP8 padding for ragged linear input dimensions."""

import torch.nn.functional as F


def patch_online_fp8_input_padding():
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod, _Fp8OnlineLinearBase
    from vllm.model_executor.utils import replace_parameter

    method = Fp8PerBlockOnlineLinearMethod
    if getattr(method, "_prime_rl_input_padding", False):
        return
    original_process = method.process_weights_after_loading
    original_apply = method.apply
    original_create = method.create_weights

    def create(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        args = (layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype)
        padding = (-input_size_per_partition) % self.weight_block_size[1]
        if not padding:
            return original_create(self, *args, **extra_weight_attrs)
        _Fp8OnlineLinearBase.create_weights(self, *args, **extra_weight_attrs)
        layer.weight_block_size = self.weight_block_size
        # Kernel selection must see the eventual padded shape, while the weight
        # loader still needs the checkpoint's original unpadded shape.
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=(sum(output_partition_sizes), input_size_per_partition + padding),
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process(self, layer):
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        # Preserve loader metadata and logical dimensions. Pad only after the
        # original checkpoint-shaped tensor has been loaded, on every reload.
        padding = (-layer.weight.shape[-1]) % self.weight_block_size[1]
        layer._prime_rl_fp8_input_padding = padding
        if padding:
            replace_parameter(layer, "weight", F.pad(layer.weight.data, (0, padding)))
        original_process(self, layer)

    def apply(self, layer, x, bias=None):
        padding = layer._prime_rl_fp8_input_padding
        if padding:
            x = F.pad(x, (0, padding))
        return original_apply(self, layer, x, bias)

    method.process_weights_after_loading = process
    method.apply = apply
    method.create_weights = create
    method._prime_rl_input_padding = True
