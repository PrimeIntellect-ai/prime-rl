/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in
 * ../MSLK_LICENSE.
 */

#define CUTLASS_NAMESPACE prime_rl_kernels

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/gemm/dispatch_policy.hpp>                   // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

namespace prime_rl_kernels::nvfp4 {

namespace cutlass = cutlass_prime_rl_kernels;

// Underlying element types (same for both Sm100 NVFP4 and Sm103 ultra NVFP4).
using ElementData = cutlass::float_e2m1_t;
using ElementScale = cutlass::float_ue4m3_t;
// Sm103 ultra CollectiveBuilder expects the (data, scale) pair as a
// cute::tuple, while the Sm100 NVFP4 CollectiveBuilder takes nv_float4_t
// directly.
using ElementPair = cute::tuple<ElementData, ElementScale>;
using ElementNvf4 = cutlass::nv_float4_t<ElementData>;

inline int64_t _byte_align(int64_t offset) {
  int64_t remainder = offset % 16;
  if (remainder != 0) {
    offset += (16 - remainder);
  }
  return offset;
}

template <
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename Sm1xxBlkScaledConfig>
__global__ void set_ultra_grouped_args_kernel(
    int64_t G,
    int64_t N,
    int64_t K,
    ProblemShape* problem_shape_ptr,
    ElementData* xq,
    const ElementData** xq_ptr,
    ElementData* wq,
    const ElementData** wq_ptr,
    ElementScale* x_scale,
    const ElementScale** x_scale_ptr,
    ElementScale* w_scale,
    const ElementScale** w_scale_ptr,
    cutlass::bfloat16_t* output,
    cutlass::bfloat16_t** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    int32_t* offsets,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    float* x_global_scale,
    const float** x_global_scale_ptr,
    float* w_global_scale,
    const float** w_global_scale_ptr) {
  uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_index >= G) {
    return;
  }

  auto round_up = [](int64_t x, int64_t y) { return ((x + y - 1) / y) * y; };

  constexpr int ele_per_quantize_group = 16; // NVFP4

  int32_t M_end = offsets[group_index];
  int32_t M_start = (group_index == 0) ? 0 : offsets[group_index - 1];
  int32_t M_group = M_end - M_start;

  if (M_group <= 0) {
    problem_shape_ptr[group_index] = ProblemShape(0, 0, 0);
    return;
  }

  // Problem shape is (N, M, K) due to transposed AB convention.
  problem_shape_ptr[group_index] = ProblemShape(N, M_group, K);

  // Data pointers.
  xq_ptr[group_index] = xq + (int64_t(M_start) * K / 2);
  wq_ptr[group_index] = wq + (int64_t(group_index) * N * K / 2);
  output_ptr[group_index] = output + (int64_t(M_start) * N);

  // Block scale offsets. Scales are padded to multiples of (128, 4).
  int64_t K_rounded = round_up(K / ele_per_quantize_group, 4);
  int64_t N_rounded = round_up(N, 128);

  int64_t x_scale_offset = 0;
  for (int i = 0; i < group_index; i++) {
    int32_t prev_start = (i == 0) ? 0 : offsets[i - 1];
    int32_t prev_M = offsets[i] - prev_start;
    x_scale_offset += round_up(prev_M, 128) * K_rounded;
  }
  x_scale_ptr[group_index] = x_scale + x_scale_offset;
  w_scale_ptr[group_index] =
      w_scale + int64_t(group_index) * N_rounded * K_rounded;

  // Strides.
  stride_a_ptr[group_index] = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(int(M_group), int(K), 1));
  stride_b_ptr[group_index] = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(int(N), int(K), 1));
  stride_c_ptr[group_index] = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(int(N), int(M_group), 1));

  // Scale factor layouts for block-scaled TMA.
  layout_SFA[group_index] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(int(M_group), int(N), int(K), 1));
  layout_SFB[group_index] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(int(M_group), int(N), int(K), 1));

  // Per-token x global scale and per-expert w global scale for EVT epilogue.
  x_global_scale_ptr[group_index] = x_global_scale + M_start;
  w_global_scale_ptr[group_index] = w_global_scale + group_index;
}

// Sm103 ultra schedules require CUDA 13+. Alias them to Sm100 NVFP4 schedules
// on older toolkits so that template instantiation with UseUltra=false still
// resolves to valid types when those CUDA 12.8 builds reach the conditional.
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
using _UltraArchTag = cutlass::arch::Sm103;
using _UltraSchedule2Sm = cutlass::gemm::
    KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103;
using _UltraSchedule1Sm = cutlass::gemm::
    KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103;
#else
using _UltraArchTag = cutlass::arch::Sm100;
using _UltraSchedule2Sm =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100;
using _UltraSchedule1Sm =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
#endif

template <
    bool UseUltra,
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K>
at::Tensor f4f4bf16_ultra_grouped_impl(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output) {
  c10::cuda::CUDAGuard deviceGuard(XQ.device());

  const int64_t G = offsets.size(0);

  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementData>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementData>::value;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag =
      cute::conditional_t<UseUltra, _UltraArchTag, cutlass::arch::Sm100>;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  // Sm103 ultra path uses cute::tuple<data, scale>; Sm100 NVFP4 path takes
  // nv_float4_t directly.
  using ElementForBuilder =
      cute::conditional_t<UseUltra, ElementPair, ElementNvf4>;

  // Kernel schedule: ultra uses BlockScaledMxNvf4UltraVs16Sm103, Sm100 uses the
  // standard Nvf4 ptr-array schedule (matching f4f4bf16_grouped).
  using KernelSchedule = cute::conditional_t<
      UseUltra,
      cute::conditional_t<
          (TB_M == 256) && (TBS_M % 2 == 0),
          _UltraSchedule2Sm,
          _UltraSchedule1Sm>,
      cute::conditional_t<
          (TB_M == 256) && (TBS_M % 2 == 0),
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>>;
  using EpilogueSchedule = cute::conditional_t<
      (TB_M == 256) && (TBS_M % 2 == 0),
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>;

  // EVT epilogue: Output = Accumulator * XGlobalScaleInv * WGlobalScaleInv
  // Due to the transposed AB convention (CUTLASS N = our M/tokens):
  //   RowBroadcast (along CUTLASS N) → per-token x_global_scale
  //   ScalarBroadcastPtrArray → per-group w_global_scale scalar
  using XGlobalScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue*,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using WGlobalScale = cutlass::epilogue::fusion::Sm90ScalarBroadcastPtrArray<
      ElementComputeEpilogue>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, Accum, XGlobalScale>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementC,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, EVTCompute0, WGlobalScale>;

  using EpilogueEVT = EVTCompute1;

  // Ultra (Sm103) uses OpClassBlockScaledTensorOp for the epilogue; Sm100
  // NVFP4 uses OpClassTensorOp (matches f4f4bf16_grouped_common.cuh).
  using EpilogueOpClass = cute::conditional_t<
      UseUltra,
      cutlass::arch::OpClassBlockScaledTensorOp,
      cutlass::arch::OpClassTensorOp>;

  // On Sm100, Sm90ScalarBroadcastPtrArray (used by WGlobalScale above) needs a
  // non-void C type so its pointer-array stride layout is set up correctly.
  // On Sm103 ultra, void C performs better, so keep the original behavior
  // there. Source C is unused (nullptr passed in arguments, no beta) in both
  // cases — this only affects internal stride bookkeeping.
  using EpilogueElementC = cute::conditional_t<UseUltra, void, ElementC>;
  using EpilogueLayoutC = cute::conditional_t<
      UseUltra,
      void,
      typename cutlass::layout::LayoutTranspose<LayoutC>::type*>;
  static constexpr int EpilogueAlignmentC = UseUltra ? 0 : AlignmentC;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          EpilogueOpClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          EpilogueElementC,
          EpilogueLayoutC,
          EpilogueAlignmentC,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          AlignmentC,
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          cutlass::arch::OpClassBlockScaledTensorOp,
          ElementForBuilder,
          LayoutB_Transpose*,
          AlignmentB,
          ElementForBuilder,
          LayoutA_Transpose*,
          AlignmentA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // Allocate a single contiguous buffer for all per-group kernel arguments.
  const int64_t problem_size_offset = 0;
  int64_t problem_size_buffer =
      _byte_align(G * sizeof(ProblemShape::UnderlyingProblemShape));

  const int64_t xq_offset = problem_size_offset + problem_size_buffer;
  int64_t xq_size_buffer = _byte_align(G * sizeof(ElementData**));

  const int64_t wq_offset = xq_offset + xq_size_buffer;
  int64_t wq_size_buffer = _byte_align(G * sizeof(ElementData**));

  const int64_t x_scale_offset = wq_offset + wq_size_buffer;
  int64_t x_scale_buffer = _byte_align(G * sizeof(ElementScale**));

  const int64_t w_scale_offset = x_scale_offset + x_scale_buffer;
  int64_t w_scale_buffer = _byte_align(G * sizeof(ElementScale**));

  const int64_t output_offset = w_scale_offset + w_scale_buffer;
  int64_t output_buffer = _byte_align(G * sizeof(ElementC**));

  const int64_t stride_a_offset = output_offset + output_buffer;
  int64_t stride_a_buffer = _byte_align(G * sizeof(StrideA));

  const int64_t stride_b_offset = stride_a_offset + stride_a_buffer;
  int64_t stride_b_buffer = _byte_align(G * sizeof(StrideB));

  const int64_t stride_c_offset = stride_b_offset + stride_b_buffer;
  int64_t stride_c_buffer = _byte_align(G * sizeof(StrideC));

  const int64_t layout_SFA_offset = stride_c_offset + stride_c_buffer;
  int64_t layout_SFA_buffer = _byte_align(G * sizeof(LayoutSFA));

  const int64_t layout_SFB_offset = layout_SFA_offset + layout_SFA_buffer;
  int64_t layout_SFB_buffer = _byte_align(G * sizeof(LayoutSFB));

  const int64_t x_gs_offset = layout_SFB_offset + layout_SFB_buffer;
  int64_t x_gs_buffer = _byte_align(G * sizeof(float*));

  const int64_t w_gs_offset = x_gs_offset + x_gs_buffer;
  int64_t w_gs_buffer = _byte_align(G * sizeof(float*));

  int64_t total_buffer_size = w_gs_offset + w_gs_buffer;

  at::TensorOptions options = XQ.options();
  at::Tensor kernel_args =
      at::empty({total_buffer_size}, options.dtype(at::kByte));

  char* kernel_args_ptr = reinterpret_cast<char*>(kernel_args.data_ptr());

  auto* problem_shape_ptr =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          kernel_args_ptr + problem_size_offset);
  const ElementData** xq_ptr =
      reinterpret_cast<const ElementData**>(kernel_args_ptr + xq_offset);
  const ElementData** wq_ptr =
      reinterpret_cast<const ElementData**>(kernel_args_ptr + wq_offset);
  const ElementScale** x_scale_ptr =
      reinterpret_cast<const ElementScale**>(kernel_args_ptr + x_scale_offset);
  const ElementScale** w_scale_ptr =
      reinterpret_cast<const ElementScale**>(kernel_args_ptr + w_scale_offset);
  ElementC** output_ptr =
      reinterpret_cast<ElementC**>(kernel_args_ptr + output_offset);
  StrideA* stride_a_ptr =
      reinterpret_cast<StrideA*>(kernel_args_ptr + stride_a_offset);
  StrideB* stride_b_ptr =
      reinterpret_cast<StrideB*>(kernel_args_ptr + stride_b_offset);
  StrideC* stride_c_ptr =
      reinterpret_cast<StrideC*>(kernel_args_ptr + stride_c_offset);
  LayoutSFA* layout_SFA_ptr =
      reinterpret_cast<LayoutSFA*>(kernel_args_ptr + layout_SFA_offset);
  LayoutSFB* layout_SFB_ptr =
      reinterpret_cast<LayoutSFB*>(kernel_args_ptr + layout_SFB_offset);
  const float** x_gs_ptr =
      reinterpret_cast<const float**>(kernel_args_ptr + x_gs_offset);
  const float** w_gs_ptr =
      reinterpret_cast<const float**>(kernel_args_ptr + w_gs_offset);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // WQ is [G, N, K/2] after transpose in the top-level dispatch.
  const int64_t N = WQ.size(-2);
  const int64_t K = WQ.size(-1) * 2; // 2 FP4 values packed per byte

  constexpr int kSetupThreads = 256;
  const int setup_blocks = (G + kSetupThreads - 1) / kSetupThreads;
  set_ultra_grouped_args_kernel<
      ProblemShape::UnderlyingProblemShape,
      StrideA,
      StrideB,
      StrideC,
      LayoutSFA,
      LayoutSFB,
      Sm1xxBlkScaledConfig><<<setup_blocks, kSetupThreads, 0, stream>>>(
      G,
      N,
      K,
      problem_shape_ptr,
      reinterpret_cast<ElementData*>(XQ.data_ptr()),
      xq_ptr,
      reinterpret_cast<ElementData*>(WQ.data_ptr()),
      wq_ptr,
      reinterpret_cast<ElementScale*>(x_scale.data_ptr()),
      x_scale_ptr,
      reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
      w_scale_ptr,
      reinterpret_cast<ElementC*>(output.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      reinterpret_cast<int32_t*>(offsets.data_ptr()),
      layout_SFA_ptr,
      layout_SFB_ptr,
      reinterpret_cast<float*>(x_global_scale.data_ptr()),
      x_gs_ptr,
      reinterpret_cast<float*>(w_global_scale.data_ptr()),
      w_gs_ptr);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = XQ.get_device();
  hw_info.sm_count =
      min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
              hw_info.device_id),
          2147483647);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {int(G), problem_shape_ptr, nullptr},
      {reinterpret_cast<const ElementData**>(wq_ptr),
       stride_b_ptr,
       reinterpret_cast<const ElementData**>(xq_ptr),
       stride_a_ptr,
       reinterpret_cast<const ElementScale**>(w_scale_ptr),
       layout_SFB_ptr,
       reinterpret_cast<const ElementScale**>(x_scale_ptr),
       layout_SFA_ptr},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr},
      hw_info};

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order =
      cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  arguments.scheduler = scheduler;

  // EVT epilogue arguments: Accum / XGlobalScale / WGlobalScale.
  // Due to AB transpose, x_global_scale (per-token) goes to RowBroadcast and
  // w_global_scale (per-expert) goes to ScalarBroadcastPtrArray.
  // ScalarBroadcastPtrArray args: {scalars, scalar_ptrs, scalar_ptr_arrays,
  // dScalar}
  arguments.epilogue.thread = {
      {
          {}, // Accumulator
          {x_gs_ptr}, // XGlobalScale (RowBroadcast, per-token)
          {} // Compute0 (multiplies)
      },
      {{}, {}, {w_gs_ptr}}, // WGlobalScale (ScalarBroadcastPtrArray)
      {} // Compute1 (multiplies)
  };

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace = at::empty(workspace_size, options.dtype(at::kByte));

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  status = gemm.initialize(
      arguments, reinterpret_cast<uint8_t*>(workspace.data_ptr()));
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run: ") +
        cutlass::cutlassGetStatusString(status));
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

} // namespace prime_rl_kernels::nvfp4

#endif

#undef CUTLASS_NAMESPACE
