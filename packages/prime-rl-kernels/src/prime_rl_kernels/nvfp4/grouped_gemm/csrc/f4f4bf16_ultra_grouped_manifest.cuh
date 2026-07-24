/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in
 * ../MSLK_LICENSE.
 */

#include <ATen/ATen.h>

namespace prime_rl_kernels::nvfp4 {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

// SM100 (B200) NVFP4 variants. Use TileShape K = 256.
at::Tensor f4f4bf16_ultra_grouped_256_128_256_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output);

at::Tensor f4f4bf16_ultra_grouped_256_256_256_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output);

using Kernel_f4f4bf16_ultra_grouped = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor);

const std::unordered_map<std::string, Kernel_f4f4bf16_ultra_grouped>&
get_f4f4bf16_ultra_grouped_kernels() {
  static const std::unordered_map<std::string, Kernel_f4f4bf16_ultra_grouped>
      kernels = {
          {"f4f4bf16_ultra_grouped_256_128_256_2_1_1",
           f4f4bf16_ultra_grouped_256_128_256_2_1_1},
          {"f4f4bf16_ultra_grouped_256_256_256_2_1_1",
           f4f4bf16_ultra_grouped_256_256_256_2_1_1},
      };
  return kernels;
}

#endif // CUDA >= 12.8
} // namespace prime_rl_kernels::nvfp4
