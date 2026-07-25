/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in
 * ../MSLK_LICENSE.
 */

#include "f4f4bf16_ultra_grouped_common.cuh"

namespace prime_rl_kernels::nvfp4 {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor f4f4bf16_ultra_grouped_256_128_256_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output) {
  return f4f4bf16_ultra_grouped_impl<false, 256, 128, 256, 2, 1, 1>(
      XQ,
      WQ,
      x_scale,
      w_scale,
      offsets,
      x_global_scale,
      w_global_scale,
      output);
}

#endif

} // namespace prime_rl_kernels::nvfp4
