#pragma once

#include <ATen/core/Tensor.h>

#include <optional>

namespace prime_rl_kernels::nvfp4 {

at::Tensor f4f4bf16_ultra_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    std::optional<at::Tensor> output = std::nullopt);

} // namespace prime_rl_kernels::nvfp4
