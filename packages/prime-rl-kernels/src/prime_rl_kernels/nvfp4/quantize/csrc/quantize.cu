/*
 * Portions of the Blackwell BF16-to-E2M1 conversion sequence are adapted
 * from NVIDIA Transformer Engine, Copyright NVIDIA Corporation & affiliates,
 * under the Apache License 2.0.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace prime_rl_kernels::nvfp4 {
namespace {

constexpr int kSfVectorSize = 16;
constexpr int kScaleRowTile = 128;
constexpr int kScaleColTile = 4;
constexpr int kQuantThreads = 256;
constexpr int kQuantScaleColsPerCta = 64;
constexpr int kQuantScaleTilesPerCta =
    kQuantScaleColsPerCta / kScaleColTile;
constexpr int kQuantScaleTileElements =
    kScaleRowTile * kQuantScaleColsPerCta;
constexpr int kQuantItemsPerThread =
    kQuantScaleTileElements / kQuantThreads;
constexpr int kDequantScaleColsPerCta = 32;
constexpr int kDequantScaleTilesPerCta =
    kDequantScaleColsPerCta / kScaleColTile;
constexpr int kDequantScaleTileElements =
    kScaleRowTile * kDequantScaleColsPerCta;
constexpr int kDequantItemsPerThread =
    kDequantScaleTileElements / kQuantThreads;
constexpr int kReductionThreads = 256;
constexpr float kGlobalScaleDenominator = 6.0f * 448.0f;

__host__ __device__ __forceinline__ int64_t round_up(
    int64_t value,
    int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

void check_sm100() {
  const auto* properties =
      at::cuda::getDeviceProperties(at::cuda::current_device());
  TORCH_CHECK(
      properties->major == 10 && properties->minor == 0,
      "prime-rl-kernels NVFP4 quantization requires SM100, but the current "
      "device is SM",
      properties->major,
      properties->minor,
      ".");
}

void check_bf16_matrix(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_quantized_tensors(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales) {
  TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor.");
  TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must use byte storage.");
  TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous.");
  TORCH_CHECK(
      block_scales.device() == packed.device() &&
          global_scales.device() == packed.device(),
      "quantized tensors must be on the same CUDA device.");
  TORCH_CHECK(
      block_scales.scalar_type() == at::kFloat8_e4m3fn,
      "block_scales must be FP8 E4M3.");
  TORCH_CHECK(
      global_scales.scalar_type() == at::kFloat,
      "global_scales must be FP32.");
  TORCH_CHECK(
      block_scales.is_contiguous() && global_scales.is_contiguous(),
      "quantized tensors must be contiguous.");
}

struct alignas(16) Bf16Block {
  uint4 vectors[2];
};

union FP4Block {
  uint64_t packed;
  __nv_fp4x4_e2m1 values[4];
};

__device__ __forceinline__ Bf16Block load_bf16_block(
    const __nv_bfloat16* input) {
  Bf16Block block;
  const auto* vectors = reinterpret_cast<const uint4*>(input);
  block.vectors[0] = __ldcg(vectors);
  block.vectors[1] = __ldcg(vectors + 1);
  return block;
}

__device__ __forceinline__ float block_amax(const Bf16Block& block) {
  const auto* pairs = reinterpret_cast<const uint32_t*>(block.vectors);
  uint32_t maximum = 0;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    asm volatile(
        "max.xorsign.abs.bf16x2 %0, %1, %2;"
        : "=r"(maximum)
        : "r"(maximum), "r"(pairs[index]));
  }
  const float2 values =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&maximum));
  return fmaxf(fabsf(values.x), fabsf(values.y));
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float block_max(
    float value,
    float* warp_maxima) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) {
    warp_maxima[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < (blockDim.x >> 5) ? warp_maxima[lane] : 0.0f;
    value = warp_max(value);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

// Adapted from Transformer Engine's Blackwell BF16-to-E2M1 conversion. Keeping
// the coefficient in FP32 avoids an extra recipe-changing BF16 rounding.
__device__ __forceinline__ uint32_t pack_e2m1_8(
    uint64_t values_03,
    uint64_t values_47,
    float coefficient) {
  uint32_t output;
  asm volatile(
      "{\n"
      ".reg.b64 coefficient_2x;\n\t"
      "mov.b64 coefficient_2x, {%3, %3};\n\t"
      ".reg.b16 h0, h1, h2, h3, h4, h5, h6, h7;\n\t"
      "mov.b64 {h0, h1, h2, h3}, %1;\n\t"
      "mov.b64 {h4, h5, h6, h7}, %2;\n\t"
      ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7;\n\t"
      "cvt.f32.bf16 v0, h0;\n\t"
      "cvt.f32.bf16 v1, h1;\n\t"
      "cvt.f32.bf16 v2, h2;\n\t"
      "cvt.f32.bf16 v3, h3;\n\t"
      "cvt.f32.bf16 v4, h4;\n\t"
      "cvt.f32.bf16 v5, h5;\n\t"
      "cvt.f32.bf16 v6, h6;\n\t"
      "cvt.f32.bf16 v7, h7;\n\t"
      ".reg.b64 v01, v23, v45, v67;\n\t"
      "mov.b64 v01, {v0, v1};\n\t"
      "mov.b64 v23, {v2, v3};\n\t"
      "mov.b64 v45, {v4, v5};\n\t"
      "mov.b64 v67, {v6, v7};\n\t"
      "mul.f32x2 v01, v01, coefficient_2x;\n\t"
      "mul.f32x2 v23, v23, coefficient_2x;\n\t"
      "mul.f32x2 v45, v45, coefficient_2x;\n\t"
      "mul.f32x2 v67, v67, coefficient_2x;\n\t"
      "mov.b64 {v1, v0}, v01;\n\t"
      "mov.b64 {v3, v2}, v23;\n\t"
      "mov.b64 {v5, v4}, v45;\n\t"
      "mov.b64 {v7, v6}, v67;\n\t"
      ".reg.b8 f0, f1, f2, f3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f2, v4, v5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f3, v6, v7;\n\t"
      "mov.b32 %0, {f0, f1, f2, f3};\n\t"
      "}"
      : "=r"(output)
      : "l"(values_03), "l"(values_47), "f"(coefficient));
  return output;
}

__device__ __forceinline__ uint64_t pack_e2m1(
    const Bf16Block& block,
    float coefficient) {
  const auto* words = reinterpret_cast<const uint64_t*>(block.vectors);
  const uint32_t low = pack_e2m1_8(words[0], words[1], coefficient);
  const uint32_t high = pack_e2m1_8(words[2], words[3], coefficient);
  return static_cast<uint64_t>(low) |
      (static_cast<uint64_t>(high) << 32);
}

__device__ __forceinline__ Bf16Block dequantize_e2m1(
    uint64_t packed,
    float decode_scale) {
  FP4Block input;
  input.packed = packed;
  Bf16Block output;
  auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output.vectors);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const float4 values = static_cast<float4>(input.values[index]);
    output_pairs[index * 2] = __floats2bfloat162_rn(
        values.x * decode_scale,
        values.y * decode_scale);
    output_pairs[index * 2 + 1] = __floats2bfloat162_rn(
        values.z * decode_scale,
        values.w * decode_scale);
  }
  return output;
}

__device__ __forceinline__ float reciprocal_approximate(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

template <int kScaleCols>
__device__ __forceinline__ int local_scale_offset(
    int row_in_tile,
    int scale_column_in_cta) {
  return (scale_column_in_cta >> 2) * (kScaleRowTile * kScaleColTile) +
      ((row_in_tile & 31) << 4) |
      (((row_in_tile >> 5) & 3) << 2) |
      (scale_column_in_cta & 3);
}

__device__ __forceinline__ uint8_t quantize_block(
    const Bf16Block& input,
    float global_decode_scale,
    uint64_t* packed) {
  const float local_amax = block_amax(input);
  float scale_value = 0.0f;
  if (global_decode_scale > 0.0f && local_amax > 0.0f) {
    scale_value =
        local_amax * reciprocal_approximate(6.0f * global_decode_scale);
  }
  const __nv_fp8_e4m3 fp8_scale(scale_value);
  const float decode_scale =
      static_cast<float>(fp8_scale) * global_decode_scale;
  const float coefficient =
      decode_scale > 0.0f ? reciprocal_approximate(decode_scale) : 0.0f;
  *packed = pack_e2m1(input, coefficient);
  return fp8_scale.__x;
}

__device__ __forceinline__ bool activation_tile_metadata(
    int tile,
    const int32_t* offsets,
    int group_count,
    int* global_row_start,
    int* valid_rows) {
  int group_start = 0;
  int tile_start = 0;
  for (int group = 0; group < group_count; ++group) {
    const int group_end = offsets[group];
    const int group_rows = group_end - group_start;
    const int group_tiles =
        (group_rows + kScaleRowTile - 1) / kScaleRowTile;
    if (tile < tile_start + group_tiles) {
      const int tile_in_group = tile - tile_start;
      *global_row_start =
          group_start + tile_in_group * kScaleRowTile;
      *valid_rows = min(
          kScaleRowTile,
          group_end - *global_row_start);
      return true;
    }
    group_start = group_end;
    tile_start += group_tiles;
  }
  return false;
}

__global__ void activation_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ global_scales,
    int rows,
    int contraction_size) {
  __shared__ float warp_maxima[kReductionThreads / 32];
  const int scale_columns = contraction_size / kSfVectorSize;
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    float local_amax = 0.0f;
    for (int scale_column = threadIdx.x;
         scale_column < scale_columns;
         scale_column += blockDim.x) {
      const int64_t input_offset =
          static_cast<int64_t>(row) * contraction_size +
          scale_column * kSfVectorSize;
      local_amax = fmaxf(
          local_amax,
          block_amax(load_bf16_block(input + input_offset)));
    }
    const float maximum = block_max(local_amax, warp_maxima);
    if (threadIdx.x == 0) {
      global_scales[row] =
          maximum > 0.0f ? maximum / kGlobalScaleDenominator : 0.0f;
    }
    __syncthreads();
  }
}

__global__ void weight_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ expert_amax,
    int64_t elements_per_expert,
    int blocks_per_expert) {
  __shared__ float warp_maxima[kReductionThreads / 32];
  const int expert = blockIdx.x / blocks_per_expert;
  const int expert_block = blockIdx.x % blocks_per_expert;
  const int64_t scale_blocks = elements_per_expert / kSfVectorSize;
  float local_amax = 0.0f;
  for (int64_t scale_block =
           static_cast<int64_t>(expert_block) * blockDim.x + threadIdx.x;
       scale_block < scale_blocks;
       scale_block += static_cast<int64_t>(blocks_per_expert) * blockDim.x) {
    const int64_t input_offset =
        static_cast<int64_t>(expert) * elements_per_expert +
        scale_block * kSfVectorSize;
    local_amax = fmaxf(
        local_amax,
        block_amax(load_bf16_block(input + input_offset)));
  }
  const float maximum = block_max(local_amax, warp_maxima);
  if (threadIdx.x == 0) {
    atomicMax(
        reinterpret_cast<unsigned int*>(expert_amax + expert),
        __float_as_uint(maximum));
  }
}

__global__ void finalize_weight_scales_kernel(
    const float* __restrict__ expert_amax,
    float* __restrict__ global_scales,
    int groups) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert < groups) {
    const float maximum = expert_amax[expert];
    global_scales[expert] =
        maximum > 0.0f ? maximum / kGlobalScaleDenominator : 0.0f;
  }
}

__global__ void quantize_activations_tiled_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int32_t* __restrict__ offsets,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    int contraction_size,
    int group_count,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kQuantScaleTileElements];
  __shared__ int global_row_start;
  __shared__ int valid_rows;
  __shared__ int active;

  if (threadIdx.x == 0) {
    active = activation_tile_metadata(
        blockIdx.x,
        offsets,
        group_count,
        &global_row_start,
        &valid_rows);
  }
  __syncthreads();
  if (!active) {
    return;
  }

#pragma unroll
  for (int item = 0; item < kQuantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kQuantScaleColsPerCta;
    const int column_in_tile = tile_index % kQuantScaleColsPerCta;
    const int scale_column =
        blockIdx.y * kQuantScaleColsPerCta + column_in_tile;
    uint8_t scale = 0;
    if (row_in_tile < valid_rows &&
        scale_column < contraction_size / kSfVectorSize) {
      const int row = global_row_start + row_in_tile;
      const int64_t input_offset =
          static_cast<int64_t>(row) * contraction_size +
          scale_column * kSfVectorSize;
      uint64_t packed_value;
      scale = quantize_block(
          load_bf16_block(input + input_offset),
          global_scales[row],
          &packed_value);
      const int64_t packed_offset =
          static_cast<int64_t>(row) * (contraction_size / 2) +
          scale_column * 8;
      *reinterpret_cast<uint64_t*>(packed + packed_offset) =
          packed_value;
    }
    scale_tile[local_scale_offset<kQuantScaleColsPerCta>(
        row_in_tile,
        column_in_tile)] =
        scale;
  }
  __syncthreads();

  const int first_scale_tile =
      blockIdx.y * kQuantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kQuantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  if (threadIdx.x <
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
          static_cast<int>(sizeof(uint4))) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(block_scales + tile_offset)[threadIdx.x] =
        reinterpret_cast<const uint4*>(scale_tile)[threadIdx.x];
  }
}

__global__ void quantize_weights_tiled_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ block_scales,
    int output_size,
    int contraction_size,
    int output_row_tiles,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kQuantScaleTileElements];
  const int expert = blockIdx.x / output_row_tiles;
  const int output_tile = blockIdx.x % output_row_tiles;
  const int output_row_start = output_tile * kScaleRowTile;

#pragma unroll
  for (int item = 0; item < kQuantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kQuantScaleColsPerCta;
    const int column_in_tile = tile_index % kQuantScaleColsPerCta;
    const int output_row = output_row_start + row_in_tile;
    const int scale_column =
        blockIdx.y * kQuantScaleColsPerCta + column_in_tile;
    uint8_t scale = 0;
    if (output_row < output_size &&
        scale_column < contraction_size / kSfVectorSize) {
      const int64_t row =
          static_cast<int64_t>(expert) * output_size + output_row;
      const int64_t input_offset =
          row * contraction_size + scale_column * kSfVectorSize;
      uint64_t packed_value;
      scale = quantize_block(
          load_bf16_block(input + input_offset),
          global_scales[expert],
          &packed_value);
      const int64_t packed_offset =
          row * (contraction_size / 2) + scale_column * 8;
      *reinterpret_cast<uint64_t*>(packed + packed_offset) =
          packed_value;
    }
    scale_tile[local_scale_offset<kQuantScaleColsPerCta>(
        row_in_tile,
        column_in_tile)] =
        scale;
  }
  __syncthreads();

  const int first_scale_tile =
      blockIdx.y * kQuantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kQuantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  if (threadIdx.x <
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
          static_cast<int>(sizeof(uint4))) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(block_scales + tile_offset)[threadIdx.x] =
        reinterpret_cast<const uint4*>(scale_tile)[threadIdx.x];
  }
}

__global__ void dequantize_activations_tiled_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ offsets,
    __nv_bfloat16* __restrict__ output,
    int contraction_size,
    int group_count,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kDequantScaleTileElements];
  __shared__ int global_row_start;
  __shared__ int valid_rows;
  __shared__ int active;

  if (threadIdx.x == 0) {
    active = activation_tile_metadata(
        blockIdx.x,
        offsets,
        group_count,
        &global_row_start,
        &valid_rows);
  }
  const int first_scale_tile =
      blockIdx.y * kDequantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kDequantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  if (threadIdx.x <
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
          static_cast<int>(sizeof(uint4))) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(scale_tile)[threadIdx.x] =
        reinterpret_cast<const uint4*>(
            block_scales + tile_offset)[threadIdx.x];
  }
  __syncthreads();
  if (!active) {
    return;
  }

#pragma unroll
  for (int item = 0; item < kDequantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kDequantScaleColsPerCta;
    const int column_in_tile = tile_index % kDequantScaleColsPerCta;
    const int scale_column =
        blockIdx.y * kDequantScaleColsPerCta + column_in_tile;
    if (row_in_tile < valid_rows &&
        scale_column < contraction_size / kSfVectorSize) {
      const int row = global_row_start + row_in_tile;
      const int64_t packed_offset =
          static_cast<int64_t>(row) * (contraction_size / 2) +
          scale_column * 8;
      const uint64_t packed_value =
          *reinterpret_cast<const uint64_t*>(packed + packed_offset);
      const uint8_t scale_bits =
          scale_tile[local_scale_offset<kDequantScaleColsPerCta>(
              row_in_tile,
              column_in_tile)];
      __nv_fp8_e4m3 scale;
      scale.__x = scale_bits;
      const float decode_scale =
          static_cast<float>(scale) * global_scales[row];
      const Bf16Block values =
          dequantize_e2m1(packed_value, decode_scale);
      const int64_t output_offset =
          static_cast<int64_t>(row) * contraction_size +
          scale_column * kSfVectorSize;
      auto* output_vectors =
          reinterpret_cast<uint4*>(output + output_offset);
      output_vectors[0] = values.vectors[0];
      output_vectors[1] = values.vectors[1];
    }
  }
}

__global__ void dequantize_weights_tiled_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    __nv_bfloat16* __restrict__ output,
    int output_size,
    int contraction_size,
    int output_row_tiles,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kDequantScaleTileElements];
  const int expert = blockIdx.x / output_row_tiles;
  const int output_tile = blockIdx.x % output_row_tiles;
  const int output_row_start = output_tile * kScaleRowTile;

  const int first_scale_tile =
      blockIdx.y * kDequantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kDequantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  if (threadIdx.x <
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
          static_cast<int>(sizeof(uint4))) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(scale_tile)[threadIdx.x] =
        reinterpret_cast<const uint4*>(
            block_scales + tile_offset)[threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kDequantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kDequantScaleColsPerCta;
    const int column_in_tile = tile_index % kDequantScaleColsPerCta;
    const int output_row = output_row_start + row_in_tile;
    const int scale_column =
        blockIdx.y * kDequantScaleColsPerCta + column_in_tile;
    if (output_row < output_size &&
        scale_column < contraction_size / kSfVectorSize) {
      const int64_t row =
          static_cast<int64_t>(expert) * output_size + output_row;
      const int64_t packed_offset =
          row * (contraction_size / 2) + scale_column * 8;
      const uint64_t packed_value =
          *reinterpret_cast<const uint64_t*>(packed + packed_offset);
      const uint8_t scale_bits =
          scale_tile[local_scale_offset<kDequantScaleColsPerCta>(
              row_in_tile,
              column_in_tile)];
      __nv_fp8_e4m3 scale;
      scale.__x = scale_bits;
      const float decode_scale =
          static_cast<float>(scale) * global_scales[expert];
      const Bf16Block values =
          dequantize_e2m1(packed_value, decode_scale);
      const int64_t output_offset =
          row * contraction_size + scale_column * kSfVectorSize;
      auto* output_vectors =
          reinterpret_cast<uint4*>(output + output_offset);
      output_vectors[0] = values.vectors[0];
      output_vectors[1] = values.vectors[1];
    }
  }
}

int weight_amax_blocks_per_expert(
    int device,
    int groups,
    int64_t scale_blocks_per_expert) {
  const auto* properties = at::cuda::getDeviceProperties(device);
  const int target = std::max(
      1,
      (properties->multiProcessorCount * 4 + groups - 1) / groups);
  const int available = std::max<int64_t>(
      1,
      (scale_blocks_per_expert + kReductionThreads - 1) /
          kReductionThreads);
  return std::min(256, std::min(target, available));
}

int activation_amax_threads(int contraction_size) {
  const int scale_columns = contraction_size / kSfVectorSize;
  return std::min(
      kReductionThreads,
      std::max(32, static_cast<int>(round_up(scale_columns, 32))));
}

int activation_amax_blocks(
    int device,
    int rows,
    int threads,
    int contraction_size) {
  if (contraction_size >= 2688) {
    return rows;
  }
  const auto* properties = at::cuda::getDeviceProperties(device);
  const int blocks_per_sm = std::max(1, 1024 / threads);
  return std::min(rows, properties->multiProcessorCount * blocks_per_sm);
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_activations_cuda(
    const at::Tensor& matrix,
    const at::Tensor& offsets) {
  check_bf16_matrix(matrix, "matrix");
  TORCH_CHECK(matrix.dim() == 2, "matrix must be 2D.");
  TORCH_CHECK(
      matrix.size(1) > 0 && matrix.size(1) % 32 == 0,
      "matrix's contraction dimension must be a positive multiple of 32.");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor.");
  TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets must be int32.");
  TORCH_CHECK(
      offsets.dim() == 1 && offsets.numel() > 0,
      "offsets must be a non-empty 1D tensor.");
  TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous.");
  TORCH_CHECK(
      offsets.device() == matrix.device(),
      "matrix and offsets must be on the same device.");

  c10::cuda::CUDAGuard device_guard(matrix.device());
  check_sm100();

  const int64_t rows = matrix.size(0);
  const int64_t contraction_size = matrix.size(1);
  const int64_t group_count = offsets.numel();
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kQuantScaleTilesPerCta) /
      kQuantScaleTilesPerCta;
  const int64_t padded_scale_rows =
      round_up(rows + group_count * (kScaleRowTile - 1), kScaleRowTile);
  const int64_t scale_row_tiles =
      padded_scale_rows / kScaleRowTile;

  auto packed = at::empty(
      {rows, contraction_size / 2},
      matrix.options().dtype(at::kByte));
  auto block_scales = at::empty(
      {padded_scale_rows, scale_columns},
      matrix.options().dtype(at::kFloat8_e4m3fn));
  auto global_scales =
      at::empty({rows}, matrix.options().dtype(at::kFloat));

  if (rows > 0) {
    const int device = matrix.get_device();
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device);
    const int amax_threads =
        activation_amax_threads(contraction_size);
    const int amax_blocks =
        activation_amax_blocks(
            device,
            rows,
            amax_threads,
            contraction_size);
    activation_amax_kernel<<<amax_blocks, amax_threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(matrix.data_ptr()),
        global_scales.mutable_data_ptr<float>(),
        rows,
        contraction_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 grid(scale_row_tiles, scale_column_ctas);
    quantize_activations_tiled_kernel
        <<<grid, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(matrix.data_ptr()),
            offsets.const_data_ptr<int32_t>(),
            packed.mutable_data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(block_scales.mutable_data_ptr()),
            global_scales.const_data_ptr<float>(),
            contraction_size,
            group_count,
            scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed, block_scales, global_scales};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_weights_cuda(const at::Tensor& weight_rows) {
  check_bf16_matrix(weight_rows, "weight_rows");
  TORCH_CHECK(weight_rows.dim() == 3, "weight_rows must be 3D.");
  TORCH_CHECK(weight_rows.size(0) > 0, "weight_rows must contain an expert.");
  TORCH_CHECK(
      weight_rows.size(2) > 0 && weight_rows.size(2) % 32 == 0,
      "weight_rows' contraction dimension must be a positive multiple of 32.");

  c10::cuda::CUDAGuard device_guard(weight_rows.device());
  check_sm100();

  const int64_t groups = weight_rows.size(0);
  const int64_t output_size = weight_rows.size(1);
  const int64_t contraction_size = weight_rows.size(2);
  const int64_t padded_output_size =
      round_up(output_size, kScaleRowTile);
  const int64_t output_row_tiles =
      padded_output_size / kScaleRowTile;
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kQuantScaleTilesPerCta) /
      kQuantScaleTilesPerCta;

  auto packed = at::empty(
      {groups, output_size, contraction_size / 2},
      weight_rows.options().dtype(at::kByte));
  auto block_scales = at::empty(
      {groups, padded_output_size * scale_columns},
      weight_rows.options().dtype(at::kFloat8_e4m3fn));
  auto global_scales =
      at::empty({groups}, weight_rows.options().dtype(at::kFloat));

  if (output_size > 0) {
    const int device = weight_rows.get_device();
    const int64_t scale_blocks_per_expert =
        output_size * contraction_size / kSfVectorSize;
    const int blocks_per_expert = weight_amax_blocks_per_expert(
        device,
        groups,
        scale_blocks_per_expert);
    auto expert_amax =
        at::zeros({groups}, weight_rows.options().dtype(at::kFloat));
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device);
    weight_amax_kernel
        <<<groups * blocks_per_expert, kReductionThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                weight_rows.data_ptr()),
            expert_amax.mutable_data_ptr<float>(),
            output_size * contraction_size,
            blocks_per_expert);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finalize_weight_scales_kernel<<<1, 256, 0, stream>>>(
        expert_amax.const_data_ptr<float>(),
        global_scales.mutable_data_ptr<float>(),
        groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 grid(
        groups * output_row_tiles,
        scale_column_ctas);
    quantize_weights_tiled_kernel
        <<<grid, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                weight_rows.data_ptr()),
            global_scales.const_data_ptr<float>(),
            packed.mutable_data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(block_scales.mutable_data_ptr()),
            output_size,
            contraction_size,
            output_row_tiles,
            scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed, block_scales, global_scales};
}

at::Tensor dequantize_activations_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales,
    const at::Tensor& offsets) {
  check_quantized_tensors(packed, block_scales, global_scales);
  TORCH_CHECK(packed.dim() == 2, "packed activations must be 2D.");
  TORCH_CHECK(
      global_scales.dim() == 1 &&
          global_scales.size(0) == packed.size(0),
      "activation global_scales must contain one value per row.");
  TORCH_CHECK(
      offsets.is_cuda() && offsets.scalar_type() == at::kInt &&
          offsets.dim() == 1 && offsets.numel() > 0 &&
          offsets.is_contiguous() && offsets.device() == packed.device(),
      "offsets must be a non-empty contiguous CUDA int32 tensor on the "
      "same device.");

  c10::cuda::CUDAGuard device_guard(packed.device());
  check_sm100();

  const int64_t rows = packed.size(0);
  const int64_t contraction_size = packed.size(1) * 2;
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kDequantScaleTilesPerCta) /
      kDequantScaleTilesPerCta;
  TORCH_CHECK(
      block_scales.numel() % (scale_columns * kScaleRowTile) == 0,
      "activation block_scales have an invalid swizzled shape.");
  const int64_t scale_row_tiles =
      block_scales.numel() / (scale_columns * kScaleRowTile);
  auto output = at::empty(
      {rows, contraction_size},
      packed.options().dtype(at::kBFloat16));
  if (rows > 0) {
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(packed.get_device());
    const dim3 grid(scale_row_tiles, scale_column_ctas);
    dequantize_activations_tiled_kernel
        <<<grid, kQuantThreads, 0, stream>>>(
            packed.const_data_ptr<uint8_t>(),
            reinterpret_cast<const uint8_t*>(block_scales.data_ptr()),
            global_scales.const_data_ptr<float>(),
            offsets.const_data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.mutable_data_ptr()),
            contraction_size,
            offsets.numel(),
            scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor dequantize_weights_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales) {
  check_quantized_tensors(packed, block_scales, global_scales);
  TORCH_CHECK(packed.dim() == 3, "packed weights must be 3D.");
  TORCH_CHECK(
      global_scales.dim() == 1 &&
          global_scales.size(0) == packed.size(0),
      "weight global_scales must contain one value per expert.");

  c10::cuda::CUDAGuard device_guard(packed.device());
  check_sm100();

  const int64_t groups = packed.size(0);
  const int64_t output_size = packed.size(1);
  const int64_t contraction_size = packed.size(2) * 2;
  const int64_t padded_output_size =
      round_up(output_size, kScaleRowTile);
  const int64_t output_row_tiles =
      padded_output_size / kScaleRowTile;
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kDequantScaleTilesPerCta) /
      kDequantScaleTilesPerCta;
  TORCH_CHECK(
      block_scales.numel() ==
          groups * padded_output_size * scale_columns,
      "weight block_scales have an invalid swizzled shape.");
  auto output = at::empty(
      {groups, output_size, contraction_size},
      packed.options().dtype(at::kBFloat16));
  if (output_size > 0) {
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(packed.get_device());
    const dim3 grid(
        groups * output_row_tiles,
        scale_column_ctas);
    dequantize_weights_tiled_kernel
        <<<grid, kQuantThreads, 0, stream>>>(
            packed.const_data_ptr<uint8_t>(),
            reinterpret_cast<const uint8_t*>(block_scales.data_ptr()),
            global_scales.const_data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.mutable_data_ptr()),
            output_size,
            contraction_size,
            output_row_tiles,
            scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

} // namespace prime_rl_kernels::nvfp4
