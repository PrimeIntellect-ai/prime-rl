/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in
 * ../MSLK_LICENSE.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "f4f4bf16_ultra_grouped_manifest.cuh"
#endif

namespace prime_rl_kernels::nvfp4 {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

namespace {

// Returns the compute capability of the current device as major*10 + minor.
int get_device_sm_version() {
  auto* props = at::cuda::getDeviceProperties(at::cuda::current_device());
  return props->major * 10 + props->minor;
}

} // namespace

Kernel_f4f4bf16_ultra_grouped
get_ultra_kernel_via_heuristics(int M, int N, int K) {
  const int sm = get_device_sm_version();
  TORCH_CHECK(
      sm == 100,
      "prime-rl-kernels NVFP4 grouped GEMM currently requires an sm_100 GPU, "
      "but the current device is sm_",
      sm,
      ".");
  if (M <= 128) {
    return f4f4bf16_ultra_grouped_256_128_256_2_1_1;
  }
  return f4f4bf16_ultra_grouped_256_256_256_2_1_1;
}

at::Tensor f4f4bf16_ultra_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    std::optional<at::Tensor> output_maybe) {
  TORCH_CHECK(XQ.is_cuda(), "XQ must be a CUDA tensor.");
  c10::cuda::CUDAGuard device_guard(XQ.device());
  TORCH_CHECK(
      WQ.device() == XQ.device() && x_scale.device() == XQ.device() &&
          w_scale.device() == XQ.device() &&
          offsets.device() == XQ.device() &&
          x_global_scale.device() == XQ.device() &&
          w_global_scale.device() == XQ.device(),
      "all grouped GEMM inputs must be on the same CUDA device.");
  TORCH_CHECK(offsets.dtype() == at::kInt, "offsets must be int32.");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D tensor.");
  TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous.");
  TORCH_CHECK(XQ.is_contiguous(), "XQ must be row major.");
  TORCH_CHECK(WQ.transpose(-2, -1).is_contiguous(), "WQ must be column major.");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale must be contiguous.");
  TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous.");
  TORCH_CHECK(
      x_global_scale.is_contiguous(),
      "x_global_scale must be contiguous.");
  TORCH_CHECK(
      w_global_scale.is_contiguous(),
      "w_global_scale must be contiguous.");
  TORCH_CHECK(XQ.dtype() == at::kFloat4_e2m1fn_x2, "XQ must be FP4.");
  TORCH_CHECK(WQ.dtype() == at::kFloat4_e2m1fn_x2, "WQ must be FP4.");
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat8_e4m3fn, "x_scale must be FP8 e4m3.");
  TORCH_CHECK(
      w_scale.dtype() == at::kFloat8_e4m3fn, "w_scale must be FP8 e4m3.");
  TORCH_CHECK(
      x_global_scale.dtype() == at::kFloat, "x_global_scale must be float32.");
  TORCH_CHECK(
      w_global_scale.dtype() == at::kFloat, "w_global_scale must be float32.");
  TORCH_CHECK(
      XQ.dim() == 2 && WQ.dim() == 3,
      "Only 2D-3D grouped GEMM (MoE forward) is supported.");

  int64_t G = offsets.size(0);
  int64_t M = XQ.size(0);
  int64_t N = WQ.size(-1);
  int64_t K = WQ.size(-2);

  TORCH_CHECK(G > 0, "offsets must contain at least one expert.");
  TORCH_CHECK(
      XQ.size(-1) == K && WQ.size(0) == G,
      "XQ shape must be (total_M, K) and WQ shape must be (G, K, N).");
  TORCH_CHECK(
      x_global_scale.dim() == 1 && x_global_scale.size(0) == M,
      "x_global_scale must have total_M elements.");
  TORCH_CHECK(
      w_global_scale.dim() == 1 && w_global_scale.size(0) == G,
      "w_global_scale must have G elements.");
  TORCH_CHECK(
      (K * 2) % 32 == 0,
      "the unpacked contraction dimension must be divisible by 32.");
  TORCH_CHECK(N % 8 == 0, "the output dimension must be divisible by 8.");

  const int64_t scale_columns = ((K * 2 / 16 + 3) / 4) * 4;
  const int64_t padded_output_rows = ((N + 127) / 128) * 128;
  TORCH_CHECK(
      x_scale.numel() >= M * scale_columns,
      "x_scale does not contain enough block scales.");
  TORCH_CHECK(
      w_scale.numel() >= G * padded_output_rows * scale_columns,
      "w_scale does not contain enough block scales.");

  at::Tensor out = output_maybe.has_value()
      ? output_maybe.value()
      : at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  TORCH_CHECK(
      out.device() == XQ.device() && out.dtype() == at::kBFloat16 &&
          out.is_contiguous() && out.dim() == 2 && out.size(0) == M &&
          out.size(1) == N,
      "output must be a contiguous BF16 tensor with shape (total_M, N) on "
      "the input device.");

  if (out.numel() == 0) {
    return out;
  }

  int M_per_group = M / G;

  auto kernel = get_ultra_kernel_via_heuristics(M_per_group, N, K * 2);

  return kernel(
      XQ,
      WQ.transpose(-2, -1), // Column-major to row-major for CUTLASS.
      x_scale,
      w_scale,
      offsets,
      x_global_scale,
      w_global_scale,
      out);
}

#else

at::Tensor f4f4bf16_ultra_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    std::optional<at::Tensor> output) {
  throw std::runtime_error(
      "f4f4bf16_ultra_grouped_mm requires CUDA 12.8+ and a Blackwell GPU");
}

#endif

} // namespace prime_rl_kernels::nvfp4
