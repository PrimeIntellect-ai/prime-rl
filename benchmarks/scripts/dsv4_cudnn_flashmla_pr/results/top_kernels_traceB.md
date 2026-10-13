# Top GPU kernels: `/home/garrett/tmp/dsv4-cfm-pr/traces/traceB/trace_0.json.gz`

- window: 2945.9 ms (`forward` [-4] to last GPU event), stream filter: none, grouped by name
- GPU busy (union over selected streams): 2555.8 ms (86.8% of window); non-NCCL busy 1763.6 ms (59.9%)
- GPU idle: 390.0 ms (13.2%); exposed NCCL (no non-NCCL kernel running): 792.3 ms (26.9%)
- summed durations: non-NCCL 1867.6 ms, NCCL 941.5 ms; 20538 GPU events
- kernel ms per stream (all streams, in window): 7: 2265.6, 32: 229.1, 23: 210.0, 28: 95.7, 24: 8.8

## Non-NCCL, by total time (top 200)

|  # | total ms | % busy | % win | calls | mean ms |  med ms | name                                                                                       |
|----|----------|--------|-------|-------|---------|---------|--------------------------------------------------------------------------------------------|
|  1 |    171.2 |    6.7 |   5.8 |    60 |   2.853 |   2.311 | chunk_cat_cuda_kernel<float, BFloat16>                                                     |
|  2 |    117.2 |    4.6 |   4.0 |    96 |   1.221 |   1.048 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  3 |     88.8 |    3.5 |   3.0 |  1103 |   0.080 |   0.002 | vectorized_elementwise_kernel<8, bfloat16_copy_kernel_cuda::lambda(float)>                 |
|  4 |     82.2 |    3.2 |   2.8 |   427 |   0.193 |   0.008 | Memcpy DtoD (Device -> Device)                                                             |
|  5 |     77.9 |    3.0 |   2.6 |   591 |   0.132 |   0.004 | elementwise_kernel<128, 4, BinaryFunctor<float, MulFunctor<float>>>                        |
|  6 |     71.3 |    2.8 |   2.4 |    24 |   2.969 |   2.062 | dsv4_sparse_attn_bwd_kernel_kernel                                                         |
|  7 |     61.0 |    2.4 |   2.1 |    48 |   1.271 |   1.067 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  8 |     58.2 |    2.3 |   2.0 |    48 |   1.212 |   1.023 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  9 |     56.3 |    2.2 |   1.9 |    48 |   1.174 |   1.080 | main_kernel                                                                                |
| 10 |     48.7 |    1.9 |   1.7 |   616 |   0.079 |   0.004 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(float), 4>                     |
| 11 |     46.4 |    1.8 |   1.6 |   184 |   0.252 |   0.147 | nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN                                             |
| 12 |     46.2 |    1.8 |   1.6 |   276 |   0.167 |   0.003 | elementwise_kernel<128, 2, BinaryFunctor<float, MulFunctor<float>>>                        |
| 13 |     44.4 |    1.7 |   1.5 |   490 |   0.091 |   0.004 | vectorized_elementwise_kernel<4, CUDAFunctor_add<float>>                                   |
| 14 |     36.3 |    1.4 |   1.2 |    85 |   0.428 |   0.755 | vectorized_elementwise_kernel<4, BUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 15 |     35.2 |    1.4 |   1.2 |    35 |   1.007 |   1.019 | multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<flo... |
| 16 |     34.3 |    1.3 |   1.2 |   296 |   0.116 |   0.062 | _triton_rope_kernel                                                                        |
| 17 |     31.7 |    1.2 |   1.1 |   164 |   0.194 |   0.169 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT                                             |
| 18 |     30.3 |    1.2 |   1.0 |    48 |   0.631 |   0.662 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>:... |
| 19 |     27.2 |    1.1 |   0.9 |    30 |   0.905 |   0.255 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT                                             |
| 20 |     25.7 |    1.0 |   0.9 |   148 |   0.174 |   0.141 | elementwise_kernel<128, 4, direct_copy_kernel_cuda::lambda(BFloat16)>                      |
| 21 |     25.3 |    1.0 |   0.9 |    96 |   0.263 |   0.128 | indexing_backward_kernel<BFloat16, 4>                                                      |
| 22 |     23.7 |    0.9 |   0.8 |   268 |   0.089 |   0.055 | vectorized_gather_kernel<16, long>                                                         |
| 23 |     23.4 |    0.9 |   0.8 |   102 |   0.229 |   0.054 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT                                             |
| 24 |     20.0 |    0.8 |   0.7 |   292 |   0.069 |   0.044 | vectorized_elementwise_kernel<8, CUDAFunctor_add<BFloat16>>                                |
| 25 |     18.3 |    0.7 |   0.6 |    96 |   0.190 |   0.200 | elementwise_kernel<128, 4, launch_clamp_scalar::lambda(BFloat16)>                          |
| 26 |     17.9 |    0.7 |   0.6 |    90 |   0.199 |   0.003 | elementwise_kernel<128, 2, CUDAFunctor_add<float>>                                         |
| 27 |     17.8 |    0.7 |   0.6 |   100 |   0.178 |   0.171 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64di... |
| 28 |     17.7 |    0.7 |   0.6 |     3 |   5.888 |   7.236 | nvjet_sm90_tst_128x256_64x4_2x1_v_badd_coopA_NTN                                           |
| 29 |     16.9 |    0.7 |   0.6 |   388 |   0.043 |   0.008 | reduce_kernel<128, 4, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 30 |     16.1 |    0.6 |   0.5 |    17 |   0.948 |   0.588 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_TNT                                             |
| 31 |     15.7 |    0.6 |   0.5 |    68 |   0.232 |   0.166 | split_with_sizes_copy_out_contiguous_no_cast_kernel                                        |
| 32 |     15.6 |    0.6 |   0.5 |    28 |   0.558 |   0.582 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT                                             |
| 33 |     14.0 |    0.5 |   0.5 |    96 |   0.145 |   0.069 | elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned sho... |
| 34 |     13.9 |    0.5 |   0.5 |    48 |   0.289 |   0.332 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div4... |
| 35 |     13.8 |    0.5 |   0.5 |    36 |   0.383 |   0.502 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_TNT                                             |
| 36 |     13.0 |    0.5 |   0.4 |   427 |   0.030 |   0.005 | reduce_kernel<512, 1, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 37 |     12.7 |    0.5 |   0.4 |   240 |   0.053 |   0.027 | vectorized_elementwise_kernel<8, BinaryFunctor<BFloat16, MulFunctor<float>>>               |
| 38 |     12.2 |    0.5 |   0.4 |    48 |   0.253 |   0.148 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_TNT                                             |
| 39 |     12.1 |    0.5 |   0.4 |    72 |   0.168 |   0.176 | elementwise_kernel<128, 4, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>              |
| 40 |     11.9 |    0.5 |   0.4 |    45 |   0.264 |   0.082 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_NNT                                             |
| 41 |     11.3 |    0.4 |   0.4 |    14 |   0.810 |   0.844 | vectorized_elementwise_kernel<4, exp_kernel_cuda::lambda(float)>                           |
| 42 |     11.2 |    0.4 |   0.4 |    92 |   0.121 |   0.002 | vectorized_elementwise_kernel<4, BinaryFunctor<float, MulFunctor<float>>>                  |
| 43 |     11.1 |    0.4 |   0.4 |    52 |   0.214 |   0.249 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16... |
| 44 |     10.6 |    0.4 |   0.4 |   148 |   0.072 |   0.005 | Memcpy HtoD (Pinned -> Device)                                                             |
| 45 |     10.4 |    0.4 |   0.4 |    35 |   0.296 |   0.300 | multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, ... |
| 46 |     10.3 |    0.4 |   0.3 |    96 |   0.107 |   0.120 | _triton_hpb_fwd_kernel                                                                     |
| 47 |     10.2 |    0.4 |   0.3 |   408 |   0.025 |   0.001 | vectorized_elementwise_kernel<4, AUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 48 |     10.0 |    0.4 |   0.3 |     1 |  10.008 |  10.008 | nvjet_sm90_tst_320x128_64x3_1x2_h_badd_coopB_NTT                                           |
| 49 |      9.7 |    0.4 |   0.3 |    18 |   0.541 |   0.705 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_NNT                                             |
| 50 |      9.7 |    0.4 |   0.3 |    39 |   0.248 |   0.072 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_NNT                                             |
| 51 |      9.6 |    0.4 |   0.3 |   356 |   0.027 |   0.010 | vectorized_elementwise_kernel<8, FillFunctor<BFloat16>>                                    |
| 52 |      9.2 |    0.4 |   0.3 |     1 |   9.158 |   9.158 | nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT                                             |
| 53 |      8.4 |    0.3 |   0.3 |    48 |   0.176 |   0.197 | tma_scatter_add_kernel<BFloat16, long>                                                     |
| 54 |      8.2 |    0.3 |   0.3 |    40 |   0.205 |   0.087 | elementwise_kernel<128, 4, BinaryFunctor<BFloat16, MulFunctor<float>>>                     |
| 55 |      7.8 |    0.3 |   0.3 |    64 |   0.122 |   0.070 | CatArrayBatchedCopy<OpaqueType<2u>, unsigned int, 4, 64, 64>                               |
| 56 |      7.3 |    0.3 |   0.2 |    48 |   0.152 |   0.157 | cunn_SpatialSoftMaxForward<float, float, float, int, SoftMaxForwardEpilogue>               |
| 57 |      7.3 |    0.3 |   0.2 |    50 |   0.145 |   0.132 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 58 |      7.1 |    0.3 |   0.2 |    24 |   0.298 |   0.333 | preprocess_kernel_kernel                                                                   |
| 59 |      7.0 |    0.3 |   0.2 |    24 |   0.292 |   0.329 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 60 |      6.7 |    0.3 |   0.2 |    48 |   0.139 |   0.144 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 1, 128, 1, 16, 8>             |
| 61 |      6.5 |    0.3 |   0.2 |    52 |   0.124 |   0.143 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>                      |
| 62 |      5.8 |    0.2 |   0.2 |    50 |   0.116 |   0.104 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 63 |      5.6 |    0.2 |   0.2 |    48 |   0.116 |   0.131 | _triton_hpb_bwd_g_hp_hr_kernel                                                             |
| 64 |      5.4 |    0.2 |   0.2 |    48 |   0.113 |   0.128 | _triton_hpb_bwd_g_x_orig_kernel                                                            |
| 65 |      5.0 |    0.2 |   0.2 |    40 |   0.125 |   0.117 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_32x3_nt_align4>                      |
| 66 |      4.8 |    0.2 |   0.2 |    35 |   0.139 |   0.140 | multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, floa... |
| 67 |      4.8 |    0.2 |   0.2 |    68 |   0.071 |   0.087 | multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Cop... |
| 68 |      4.8 |    0.2 |   0.2 |    32 |   0.149 |   0.172 | mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>                      |
| 69 |      4.8 |    0.2 |   0.2 |    64 |   0.075 |   0.043 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lam... |
| 70 |      4.3 |    0.2 |   0.1 |   633 |   0.007 |   0.006 | at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::rad... |
| 71 |      4.2 |    0.2 |   0.1 |    96 |   0.043 |   0.021 | vectorized_elementwise_kernel<8, silu_kernel::lambda(BFloat16)>                            |
| 72 |      4.0 |    0.2 |   0.1 |    68 |   0.058 |   0.041 | multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>,... |
| 73 |      3.7 |    0.1 |   0.1 |     6 |   0.616 |   0.624 | nvjet_sm90_tst_128x304_64x3_2x1_v_bz_coopA_NNT                                             |
| 74 |      3.6 |    0.1 |   0.1 |    80 |   0.045 |   0.019 | elementwise_kernel<128, 4, masked_fill_kernel::lambda(BFloat16, bool)>                     |
| 75 |      3.4 |    0.1 |   0.1 |     6 |   0.575 |   0.584 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_NNT                                             |
| 76 |      3.4 |    0.1 |   0.1 |     6 |   0.572 |   0.585 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_NNT                                             |
| 77 |      3.4 |    0.1 |   0.1 |    48 |   0.070 |   0.054 | elementwise_kernel<128, 4, CUDAFunctor_add<BFloat16>>                                      |
| 78 |      3.3 |    0.1 |   0.1 |    24 |   0.138 |   0.144 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 2, 128, 1, 16, 8>             |
| 79 |      3.3 |    0.1 |   0.1 |     6 |   0.544 |   0.551 | nvjet_sm90_tst_256x160_64x4_2x1_v_bz_coopA_NNT                                             |
| 80 |      3.2 |    0.1 |   0.1 |    24 |   0.133 |   0.133 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_TNT                                             |
| 81 |      3.1 |    0.1 |   0.1 |     6 |   0.523 |   0.537 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_NNN                                             |
| 82 |      3.1 |    0.1 |   0.1 |     6 |   0.512 |   0.521 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_NNT                                             |
| 83 |      3.0 |    0.1 |   0.1 |    40 |   0.074 |   0.073 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_TNT                                             |
| 84 |      2.9 |    0.1 |   0.1 |    48 |   0.061 |   0.030 | vectorized_elementwise_kernel<8, silu_backward_kernel::lambda(BFloat16, BFloat16)>         |
| 85 |      2.9 |    0.1 |   0.1 |    39 |   0.073 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsi... |
| 86 |      2.5 |    0.1 |   0.1 |    40 |   0.063 |   0.067 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN                                             |
| 87 |      2.5 |    0.1 |   0.1 |   100 |   0.025 |   0.028 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64di... |
| 88 |      2.4 |    0.1 |   0.1 |    96 |   0.025 |   0.014 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 3, 128, 1, 16, 8>             |
| 89 |      2.3 |    0.1 |   0.1 |    16 |   0.146 |   0.140 | cunn_SpatialSoftMaxBackward<float, float, float, SoftMaxBackwardEpilogue>                  |
| 90 |      2.2 |    0.1 |   0.1 |    68 |   0.033 |   0.024 | reduce_kernel<128, 4, ReduceOp<BFloat16, sum_functor, unsigned int, BFloat16, 4, 8>>       |
| 91 |      2.2 |    0.1 |   0.1 |    52 |   0.042 |   0.047 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4... |
| 92 |      2.2 |    0.1 |   0.1 |  1674 |   0.001 |   0.001 | Memset (Device)                                                                            |
| 93 |      2.1 |    0.1 |   0.1 |     8 |   0.265 |   0.297 | mbtopk::gatherTopK<float, unsigned int, 2>                                                 |
| 94 |      2.1 |    0.1 |   0.1 |     8 |   0.262 |   0.288 | _triton_fp8_indexer_kernel                                                                 |
| 95 |      1.9 |    0.1 |   0.1 |    16 |   0.117 |   0.045 | _per_token_group_quant_fp8                                                                 |
| 96 |      1.6 |    0.1 |   0.1 |    48 |   0.034 |   0.016 | vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cud... |
| 97 |      1.6 |    0.1 |   0.1 |    96 |   0.016 |   0.009 | _triton_sinkhorn_fwd_kernel                                                                |
| 98 |      1.3 |    0.1 |   0.0 |    40 |   0.032 |   0.028 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                                                   |
| 99 |      1.2 |    0.0 |   0.0 |    24 |   0.052 |   0.054 | postprocess_kernel_kernel                                                                  |
| 100 |      1.2 |    0.0 |   0.0 |    96 |   0.012 |   0.014 | vectorized_elementwise_kernel<8, launch_clamp_scalar::lambda(BFloat16)>                    |
| 101 |      1.2 |    0.0 |   0.0 |   288 |   0.004 |   0.003 | elementwise_kernel<128, 4, CUDAFunctor_add<float>>                                         |
| 102 |      1.1 |    0.0 |   0.0 |     8 |   0.140 |   0.158 | radixSortKVInPlace<2, -1, 32, 32, float, long, unsigned int>                               |
| 103 |      1.1 |    0.0 |   0.0 |    12 |   0.089 |   0.100 | sbtopk::gatherTopK<float, unsigned int, 2, false>                                          |
| 104 |      1.0 |    0.0 |   0.0 |     6 |   0.165 |   0.166 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_splitK_NNT                                      |
| 105 |      0.9 |    0.0 |   0.0 |   328 |   0.003 |   0.002 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(long), 4>                      |
| 106 |      0.9 |    0.0 |   0.0 |     8 |   0.111 |   0.126 | elementwise_kernel<128, 4, float8_copy_kernel_cuda::lambda(Float8_e4m3fn)>                 |
| 107 |      0.9 |    0.0 |   0.0 |   330 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, FillFunctor<float>>                                       |
| 108 |      0.8 |    0.0 |   0.0 |    48 |   0.016 |   0.018 | _triton_sinkhorn_bwd_kernel                                                                |
| 109 |      0.8 |    0.0 |   0.0 |     6 |   0.130 |   0.132 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_NNT                                             |
| 110 |      0.7 |    0.0 |   0.0 |    20 |   0.036 |   0.035 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_NTT                                                   |
| 111 |      0.7 |    0.0 |   0.0 |    72 |   0.010 |   0.011 | vectorized_elementwise_kernel<8, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>        |
| 112 |      0.6 |    0.0 |   0.0 |    20 |   0.029 |   0.029 | nvjet_sm90_tst_128x128_64x6_1x2_h_bz_NTT                                                   |
| 113 |      0.6 |    0.0 |   0.0 |    76 |   0.008 |   0.005 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>    |
| 114 |      0.6 |    0.0 |   0.0 |    24 |   0.023 |   0.026 | nvjet_sm90_tst_64x128_64x8_2x1_v_bz_NTN                                                    |
| 115 |      0.5 |    0.0 |   0.0 |    52 |   0.010 |   0.009 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned long, unsigned long)>  |
| 116 |      0.5 |    0.0 |   0.0 |    20 |   0.026 |   0.026 | nvjet_sm90_tst_128x120_64x6_2x4_h_bz_TNT                                                   |
| 117 |      0.5 |    0.0 |   0.0 |    20 |   0.025 |   0.028 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                                                    |
| 118 |      0.5 |    0.0 |   0.0 |     3 |   0.168 |   0.166 | compute_grad_weight<BFloat16, long>                                                        |
| 119 |      0.5 |    0.0 |   0.0 |    20 |   0.025 |   0.025 | nvjet_sm90_tst_64x200_64x6_2x1_v_bz_TNT                                                    |
| 120 |      0.5 |    0.0 |   0.0 |   180 |   0.003 |   0.001 | unrolled_elementwise_kernel<direct_copy_kernel_cuda, 4>                                    |
| 121 |      0.5 |    0.0 |   0.0 |   200 |   0.002 |   0.001 | vectorized_elementwise_kernel<2, CUDAFunctorOnSelf_add<long>>                              |
| 122 |      0.5 |    0.0 |   0.0 |    12 |   0.040 |   0.039 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_nt_align4>                      |
| 123 |      0.5 |    0.0 |   0.0 |   171 |   0.003 |   0.002 | at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::ra... |
| 124 |      0.4 |    0.0 |   0.0 |    96 |   0.005 |   0.004 | kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor... |
| 125 |      0.4 |    0.0 |   0.0 |   196 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, sigmoid_kernel_cuda::lambda(float)>                       |
| 126 |      0.4 |    0.0 |   0.0 |    24 |   0.017 |   0.016 | radixSortKVInPlace<-2, -1, 32, 32, float, long, unsigned int>                              |
| 127 |      0.4 |    0.0 |   0.0 |    55 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, ... |
| 128 |      0.4 |    0.0 |   0.0 |     3 |   0.126 |   0.123 | sum_and_scatter<BFloat16, long>                                                            |
| 129 |      0.4 |    0.0 |   0.0 |    32 |   0.012 |   0.014 | mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>                                 |
| 130 |      0.4 |    0.0 |   0.0 |    48 |   0.008 |   0.008 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64di... |
| 131 |      0.4 |    0.0 |   0.0 |    88 |   0.004 |   0.004 | vectorized_elementwise_kernel<8, compare_scalar_kernel<int>>                               |
| 132 |      0.3 |    0.0 |   0.0 |    80 |   0.004 |   0.004 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div... |
| 133 |      0.3 |    0.0 |   0.0 |    88 |   0.004 |   0.004 | reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned i... |
| 134 |      0.3 |    0.0 |   0.0 |   168 |   0.002 |   0.002 | elementwise_kernel<128, 2, BinaryFunctor<float, DivFunctor<float>>>                        |
| 135 |      0.3 |    0.0 |   0.0 |    48 |   0.006 |   0.006 | vectorized_elementwise_kernel<8, compare_scalar_kernel<long>::lambda(long)>                |
| 136 |      0.3 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda:... |
| 137 |      0.3 |    0.0 |   0.0 |   199 |   0.001 |   0.001 | vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<float>>                             |
| 138 |      0.3 |    0.0 |   0.0 |   116 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pinned)                                                             |
| 139 |      0.3 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 140 |      0.3 |    0.0 |   0.0 |   136 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda:... |
| 141 |      0.3 |    0.0 |   0.0 |   224 |   0.001 |   0.001 | vectorized_elementwise_kernel<2, AUnaryFunctor<long, long, long, MulFunctor<long>>>        |
| 142 |      0.3 |    0.0 |   0.0 |    24 |   0.011 |   0.012 | indexing_backward_kernel_stride_1<float>                                                   |
| 143 |      0.2 |    0.0 |   0.0 |    24 |   0.010 |   0.011 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1... |
| 144 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | vectorized_elementwise_kernel<4, softplus_kernel::lambda(float)>                           |
| 145 |      0.2 |    0.0 |   0.0 |    32 |   0.007 |   0.008 | mbtopk::computeDigitCumSum                                                                 |
| 146 |      0.2 |    0.0 |   0.0 |    40 |   0.006 |   0.006 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 147 |      0.2 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda:... |
| 148 |      0.2 |    0.0 |   0.0 |   154 |   0.001 |   0.001 | vectorized_elementwise_kernel<4, FillFunctor<int>>                                         |
| 149 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | _fill_indices_kernel                                                                       |
| 150 |      0.2 |    0.0 |   0.0 |    31 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, f... |
| 151 |      0.2 |    0.0 |   0.0 |   227 |   0.001 |   0.001 | elementwise_kernel_with_index<int, arange_cuda_out::lambda(long)> #2                       |
| 152 |      0.2 |    0.0 |   0.0 |    16 |   0.014 |   0.015 | elementwise_kernel<128, 4, CompareFunctor<long>>                                           |
| 153 |      0.2 |    0.0 |   0.0 |    20 |   0.011 |   0.011 | nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                                                    |
| 154 |      0.2 |    0.0 |   0.0 |    52 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lam... |
| 155 |      0.2 |    0.0 |   0.0 |   171 |   0.001 |   0.001 | at_cuda_detail::cub::radix_sort::DeviceRadixSortExclusiveSumKernel<at_cuda_detail::cub:... |
| 156 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | reduce_kernel<128, 4, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 157 |      0.2 |    0.0 |   0.0 |   112 |   0.002 |   0.001 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long,... |
| 158 |      0.2 |    0.0 |   0.0 |    24 |   0.008 |   0.009 | CatArrayBatchedCopy_vectorized<OpaqueType<4u>, unsigned int, 4, 128, 1, 16, 4>             |
| 159 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lam... |
| 160 |      0.2 |    0.0 |   0.0 |    58 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, BinaryFunctor<float, DivFunctor<float>>>                  |
| 161 |      0.2 |    0.0 |   0.0 |    24 |   0.007 |   0.007 | vectorized_elementwise_kernel<4, softplus_backward_kernel::lambda(float, float)>           |
| 162 |      0.2 |    0.0 |   0.0 |   100 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, sigmoid_backward_kernel_cuda::lambda(float, float)>       |
| 163 |      0.2 |    0.0 |   0.0 |    96 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |
| 164 |      0.1 |    0.0 |   0.0 |    28 |   0.005 |   0.004 | elementwise_kernel<128, 2, CUDAFunctor_add<long>>                                          |
| 165 |      0.1 |    0.0 |   0.0 |    40 |   0.003 |   0.003 | cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, fl... |
| 166 |      0.1 |    0.0 |   0.0 |    50 |   0.003 |   0.003 | vectorized_elementwise_kernel<4, sqrt_kernel_cuda::lambda(float)>                          |
| 167 |      0.1 |    0.0 |   0.0 |    18 |   0.007 |   0.009 | elementwise_kernel<128, 2, direct_copy_kernel_cuda>                                        |
| 168 |      0.1 |    0.0 |   0.0 |    20 |   0.006 |   0.006 | elementwise_kernel<128, 4, CompareFunctor<long>> #2                                        |
| 169 |      0.1 |    0.0 |   0.0 |    72 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, BUnaryFunctor<int, int, int, div_floor_kernel_cuda::la... |
| 170 |      0.1 |    0.0 |   0.0 |    68 |   0.002 |   0.001 | unrolled_elementwise_kernel<CUDAFunctor_add<long>, 4>                                      |
| 171 |      0.1 |    0.0 |   0.0 |    96 |   0.001 |   0.001 | CatArrayBatchedCopy_alignedK_contig<OpaqueType<2u>, unsigned int, 1, 128, 1, 8>            |
| 172 |      0.1 |    0.0 |   0.0 |    16 |   0.007 |   0.008 | vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned... |
| 173 |      0.1 |    0.0 |   0.0 |     1 |   0.109 |   0.109 | embedding_backward_feature_kernel<BFloat16, float, long>                                   |
| 174 |      0.1 |    0.0 |   0.0 |    56 |   0.002 |   0.002 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, ... |
| 175 |      0.1 |    0.0 |   0.0 |     4 |   0.025 |   0.024 | nvjet_sm90_tst_64x80_64x11_1x4_h_bz_TNT                                                    |
| 176 |      0.1 |    0.0 |   0.0 |    88 |   0.001 |   0.001 | vectorized_elementwise_kernel<2, CUDAFunctor_add<long>>                                    |
| 177 |      0.1 |    0.0 |   0.0 |    84 |   0.001 |   0.001 | vectorized_elementwise_kernel<2, launch_clamp_scalar::lambda(long)>                        |
| 178 |      0.1 |    0.0 |   0.0 |   115 |   0.001 |   0.001 | at_cuda_detail::cub::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long... |
| 179 |      0.1 |    0.0 |   0.0 |    12 |   0.008 |   0.007 | vectorized_elementwise_kernel<2, masked_fill_kernel::lambda(long, bool)>                   |
| 180 |      0.1 |    0.0 |   0.0 |    80 |   0.001 |   0.001 | vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<int>>                               |
| 181 |      0.1 |    0.0 |   0.0 |     4 |   0.022 |   0.022 | nvjet_sm90_tst_64x72_64x12_1x4_h_bz_TNT                                                    |
| 182 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |
| 183 |      0.1 |    0.0 |   0.0 |    12 |   0.007 |   0.007 | bitonicSortKVInPlace<2, -1, 16, 16, float, long, GTOp<float, true>, unsigned int>          |
| 184 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | CatArrayBatchedCopy_vectorized<OpaqueType<4u>, unsigned int, 3, 128, 1, 16, 4>             |
| 185 |      0.1 |    0.0 |   0.0 |     8 |   0.010 |   0.010 | CatArrayBatchedCopy_alignedK_contig<OpaqueType<4u>, unsigned int, 4, 128, 1, 16>           |
| 186 |      0.1 |    0.0 |   0.0 |     4 |   0.020 |   0.020 | nvjet_sm90_tst_64x64_64x13_1x4_h_bz_TNT                                                    |
| 187 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |
| 188 |      0.1 |    0.0 |   0.0 |    39 |   0.002 |   0.002 | unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory:... |
| 189 |      0.1 |    0.0 |   0.0 |    32 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pageable)                                                           |
| 190 |      0.1 |    0.0 |   0.0 |    33 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int... |
| 191 |      0.1 |    0.0 |   0.0 |    44 |   0.002 |   0.002 | vectorized_elementwise_kernel<8, AUnaryFunctor<long, long, bool, CompareEqFunctor<long>>>  |
| 192 |      0.1 |    0.0 |   0.0 |    10 |   0.006 |   0.009 | cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, ... |
| 193 |      0.1 |    0.0 |   0.0 |    16 |   0.004 |   0.004 | at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_ke... |
| 194 |      0.1 |    0.0 |   0.0 |    48 |   0.001 |   0.001 | fill_reverse_indices_kernel                                                                |
| 195 |      0.1 |    0.0 |   0.0 |    72 |   0.001 |   0.001 | elementwise_kernel_with_index<int, arange_cuda_out::lambda(long)>                          |
| 196 |      0.1 |    0.0 |   0.0 |    24 |   0.002 |   0.002 | unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4>                                     |
| 197 |      0.0 |    0.0 |   0.0 |    40 |   0.001 |   0.001 | CatArrayBatchedCopy_alignedK_contig<OpaqueType<4u>, unsigned int, 1, 128, 1, 16>           |
| 198 |      0.0 |    0.0 |   0.0 |    48 |   0.001 |   0.001 | CatArrayBatchedCopy_vectorized<OpaqueType<8u>, unsigned int, 1, 128, 1, 16, 2>             |
| 199 |      0.0 |    0.0 |   0.0 |    56 |   0.001 |   0.001 | at_cuda_detail::cub::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<int,... |
| 200 |      0.0 |    0.0 |   0.0 |    52 |   0.001 |   0.001 | vectorized_elementwise_kernel<2, FillFunctor<long>>                                        |

- 2: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 7: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 8: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 15: `multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<float, 4, (ADAM_MODE)1, false>, float const*, double, double, double, dou...`
- 18: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 27: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64div41_None_None_None_tensorptrf32gmemalign16o16384i64div41_None_tensorpt...`
- 33: `elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned short)>`
- 34: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div41_None_None_None_tensorptrf32gmemalign16o512i64div41_None_tensorptrf32...`
- 43: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16384i64div41_None_tensorptrf32gmemalign16o16384i64div41_None_tensorptrf...`
- 45: `multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, 1, 0>, multiplies<float>, float*, float>`
- 57: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas`
- 59: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_None_tensorptrbf16gmemalign16o512i64div81_None_tensorptrf32...`
- 62: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas`
- 66: `multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, float, 1, 1, 0>, float*, int>`
- 67: `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Copy<BFloat16, BFloat16>>`
- 69: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 70: `at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy10...`
- 72: `multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>, Copy<BFloat16, float>>`
- 85: `reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsigned int, float, 4, 4>>`
- 87: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_2409601_None_None_tensorptrbf16gmemal...`
- 91: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_1409601_tensorptrbf16gmemalig...`
- 96: `vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cuda::lambda(bool, bool)>>`
- 123: `at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy1...`
- 124: `kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor_histogram<float, float, false>(Tensor, Tensor, Tensor, long, Accumula...`
- 127: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, OpaqueType<4>, long>::operator()<TensorAssign>>`
- 130: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_4102401_None_None_tensorptrbf16gmemal...`
- 132: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_451201_None_None_tensorptrbf16gmemalig...`
- 133: `reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned int, int, 4, 4>>`
- 136: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda::lambda(long, long)>>`
- 140: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda::lambda(long, long)>>`
- 143: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_1102401_tensorptrbf16gmemalig...`
- 146: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_251201_tensorptrbf16gmemalign1...`
- 147: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda::lambda(long, long)>>`
- 150: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, float, long>::operator()<ReduceAdd>>`
- 154: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lambda(char*, char const*, long)>>`
- 155: `at_cuda_detail::cub::radix_sort::DeviceRadixSortExclusiveSumKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Poli...`
- 157: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long, long, long, unsigned int, plus<long>>::Policy1000, long const*, long*...`
- 159: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lambda(char*, char const*, long)>>`
- 163: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<long, cute::C<1>, cut...`
- 165: `cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>`
- 169: `vectorized_elementwise_kernel<4, BUnaryFunctor<int, int, int, div_floor_kernel_cuda::lambda(int, int)>>`
- 172: `vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>`
- 174: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, int, int, unsigned int, plus<int>>::Policy1000, int const*, int*, at_c...`
- 178: `at_cuda_detail::cub::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true>>`
- 182: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<cute::C<1>, long, cut...`
- 187: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<long, cute::C<1>, cut...`
- 188: `unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory::StoreWithoutCast>`
- 190: `reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int, float, 4, 4>>`
- 192: `cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>`
- 193: `at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_key::policy_hub<thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::t...`
- 199: `at_cuda_detail::cub::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<int, true>>`

## Non-NCCL, by time per call (top 200, calls >= 1)

|  # | total ms | % busy | % win | calls | mean ms |  med ms | name                                                                                       |
|----|----------|--------|-------|-------|---------|---------|--------------------------------------------------------------------------------------------|
|  1 |     10.0 |    0.4 |   0.3 |     1 |  10.008 |  10.008 | nvjet_sm90_tst_320x128_64x3_1x2_h_badd_coopB_NTT                                           |
|  2 |      9.2 |    0.4 |   0.3 |     1 |   9.158 |   9.158 | nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT                                             |
|  3 |     17.7 |    0.7 |   0.6 |     3 |   5.888 |   7.236 | nvjet_sm90_tst_128x256_64x4_2x1_v_badd_coopA_NTN                                           |
|  4 |     71.3 |    2.8 |   2.4 |    24 |   2.969 |   2.062 | dsv4_sparse_attn_bwd_kernel_kernel                                                         |
|  5 |    171.2 |    6.7 |   5.8 |    60 |   2.853 |   2.311 | chunk_cat_cuda_kernel<float, BFloat16>                                                     |
|  6 |     61.0 |    2.4 |   2.1 |    48 |   1.271 |   1.067 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  7 |    117.2 |    4.6 |   4.0 |    96 |   1.221 |   1.048 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  8 |     58.2 |    2.3 |   2.0 |    48 |   1.212 |   1.023 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  9 |     56.3 |    2.2 |   1.9 |    48 |   1.174 |   1.080 | main_kernel                                                                                |
| 10 |     35.2 |    1.4 |   1.2 |    35 |   1.007 |   1.019 | multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<flo... |
| 11 |     16.1 |    0.6 |   0.5 |    17 |   0.948 |   0.588 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_TNT                                             |
| 12 |     27.2 |    1.1 |   0.9 |    30 |   0.905 |   0.255 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT                                             |
| 13 |     11.3 |    0.4 |   0.4 |    14 |   0.810 |   0.844 | vectorized_elementwise_kernel<4, exp_kernel_cuda::lambda(float)>                           |
| 14 |     30.3 |    1.2 |   1.0 |    48 |   0.631 |   0.662 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>:... |
| 15 |      3.7 |    0.1 |   0.1 |     6 |   0.616 |   0.624 | nvjet_sm90_tst_128x304_64x3_2x1_v_bz_coopA_NNT                                             |
| 16 |      3.4 |    0.1 |   0.1 |     6 |   0.575 |   0.584 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_NNT                                             |
| 17 |      3.4 |    0.1 |   0.1 |     6 |   0.572 |   0.585 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_NNT                                             |
| 18 |     15.6 |    0.6 |   0.5 |    28 |   0.558 |   0.582 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT                                             |
| 19 |      3.3 |    0.1 |   0.1 |     6 |   0.544 |   0.551 | nvjet_sm90_tst_256x160_64x4_2x1_v_bz_coopA_NNT                                             |
| 20 |      9.7 |    0.4 |   0.3 |    18 |   0.541 |   0.705 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_NNT                                             |
| 21 |      3.1 |    0.1 |   0.1 |     6 |   0.523 |   0.537 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_NNN                                             |
| 22 |      3.1 |    0.1 |   0.1 |     6 |   0.512 |   0.521 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_NNT                                             |
| 23 |     36.3 |    1.4 |   1.2 |    85 |   0.428 |   0.755 | vectorized_elementwise_kernel<4, BUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 24 |     13.8 |    0.5 |   0.5 |    36 |   0.383 |   0.502 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_TNT                                             |
| 25 |      7.1 |    0.3 |   0.2 |    24 |   0.298 |   0.333 | preprocess_kernel_kernel                                                                   |
| 26 |     10.4 |    0.4 |   0.4 |    35 |   0.296 |   0.300 | multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, ... |
| 27 |      7.0 |    0.3 |   0.2 |    24 |   0.292 |   0.329 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 28 |     13.9 |    0.5 |   0.5 |    48 |   0.289 |   0.332 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div4... |
| 29 |      2.1 |    0.1 |   0.1 |     8 |   0.265 |   0.297 | mbtopk::gatherTopK<float, unsigned int, 2>                                                 |
| 30 |     11.9 |    0.5 |   0.4 |    45 |   0.264 |   0.082 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_NNT                                             |
| 31 |     25.3 |    1.0 |   0.9 |    96 |   0.263 |   0.128 | indexing_backward_kernel<BFloat16, 4>                                                      |
| 32 |      2.1 |    0.1 |   0.1 |     8 |   0.262 |   0.288 | _triton_fp8_indexer_kernel                                                                 |
| 33 |     12.2 |    0.5 |   0.4 |    48 |   0.253 |   0.148 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_TNT                                             |
| 34 |     46.4 |    1.8 |   1.6 |   184 |   0.252 |   0.147 | nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN                                             |
| 35 |      9.7 |    0.4 |   0.3 |    39 |   0.248 |   0.072 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_NNT                                             |
| 36 |     15.7 |    0.6 |   0.5 |    68 |   0.232 |   0.166 | split_with_sizes_copy_out_contiguous_no_cast_kernel                                        |
| 37 |     23.4 |    0.9 |   0.8 |   102 |   0.229 |   0.054 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT                                             |
| 38 |     11.1 |    0.4 |   0.4 |    52 |   0.214 |   0.249 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16... |
| 39 |      8.2 |    0.3 |   0.3 |    40 |   0.205 |   0.087 | elementwise_kernel<128, 4, BinaryFunctor<BFloat16, MulFunctor<float>>>                     |
| 40 |     17.9 |    0.7 |   0.6 |    90 |   0.199 |   0.003 | elementwise_kernel<128, 2, CUDAFunctor_add<float>>                                         |
| 41 |     31.7 |    1.2 |   1.1 |   164 |   0.194 |   0.169 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT                                             |
| 42 |     82.2 |    3.2 |   2.8 |   427 |   0.193 |   0.008 | Memcpy DtoD (Device -> Device)                                                             |
| 43 |     18.3 |    0.7 |   0.6 |    96 |   0.190 |   0.200 | elementwise_kernel<128, 4, launch_clamp_scalar::lambda(BFloat16)>                          |
| 44 |     17.8 |    0.7 |   0.6 |   100 |   0.178 |   0.171 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64di... |
| 45 |      8.4 |    0.3 |   0.3 |    48 |   0.176 |   0.197 | tma_scatter_add_kernel<BFloat16, long>                                                     |
| 46 |     25.7 |    1.0 |   0.9 |   148 |   0.174 |   0.141 | elementwise_kernel<128, 4, direct_copy_kernel_cuda::lambda(BFloat16)>                      |
| 47 |      0.5 |    0.0 |   0.0 |     3 |   0.168 |   0.166 | compute_grad_weight<BFloat16, long>                                                        |
| 48 |     12.1 |    0.5 |   0.4 |    72 |   0.168 |   0.176 | elementwise_kernel<128, 4, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>              |
| 49 |     46.2 |    1.8 |   1.6 |   276 |   0.167 |   0.003 | elementwise_kernel<128, 2, BinaryFunctor<float, MulFunctor<float>>>                        |
| 50 |      1.0 |    0.0 |   0.0 |     6 |   0.165 |   0.166 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_splitK_NNT                                      |
| 51 |      7.3 |    0.3 |   0.2 |    48 |   0.152 |   0.157 | cunn_SpatialSoftMaxForward<float, float, float, int, SoftMaxForwardEpilogue>               |
| 52 |      4.8 |    0.2 |   0.2 |    32 |   0.149 |   0.172 | mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>                      |
| 53 |      2.3 |    0.1 |   0.1 |    16 |   0.146 |   0.140 | cunn_SpatialSoftMaxBackward<float, float, float, SoftMaxBackwardEpilogue>                  |
| 54 |     14.0 |    0.5 |   0.5 |    96 |   0.145 |   0.069 | elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned sho... |
| 55 |      7.3 |    0.3 |   0.2 |    50 |   0.145 |   0.132 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 56 |      1.1 |    0.0 |   0.0 |     8 |   0.140 |   0.158 | radixSortKVInPlace<2, -1, 32, 32, float, long, unsigned int>                               |
| 57 |      6.7 |    0.3 |   0.2 |    48 |   0.139 |   0.144 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 1, 128, 1, 16, 8>             |
| 58 |      4.8 |    0.2 |   0.2 |    35 |   0.139 |   0.140 | multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, floa... |
| 59 |      3.3 |    0.1 |   0.1 |    24 |   0.138 |   0.144 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 2, 128, 1, 16, 8>             |
| 60 |      3.2 |    0.1 |   0.1 |    24 |   0.133 |   0.133 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_TNT                                             |
| 61 |     77.9 |    3.0 |   2.6 |   591 |   0.132 |   0.004 | elementwise_kernel<128, 4, BinaryFunctor<float, MulFunctor<float>>>                        |
| 62 |      0.8 |    0.0 |   0.0 |     6 |   0.130 |   0.132 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_NNT                                             |
| 63 |      0.4 |    0.0 |   0.0 |     3 |   0.126 |   0.123 | sum_and_scatter<BFloat16, long>                                                            |
| 64 |      5.0 |    0.2 |   0.2 |    40 |   0.125 |   0.117 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_32x3_nt_align4>                      |
| 65 |      6.5 |    0.3 |   0.2 |    52 |   0.124 |   0.143 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>                      |
| 66 |      7.8 |    0.3 |   0.3 |    64 |   0.122 |   0.070 | CatArrayBatchedCopy<OpaqueType<2u>, unsigned int, 4, 64, 64>                               |
| 67 |     11.2 |    0.4 |   0.4 |    92 |   0.121 |   0.002 | vectorized_elementwise_kernel<4, BinaryFunctor<float, MulFunctor<float>>>                  |
| 68 |      1.9 |    0.1 |   0.1 |    16 |   0.117 |   0.045 | _per_token_group_quant_fp8                                                                 |
| 69 |      5.6 |    0.2 |   0.2 |    48 |   0.116 |   0.131 | _triton_hpb_bwd_g_hp_hr_kernel                                                             |
| 70 |     34.3 |    1.3 |   1.2 |   296 |   0.116 |   0.062 | _triton_rope_kernel                                                                        |
| 71 |      5.8 |    0.2 |   0.2 |    50 |   0.116 |   0.104 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 72 |      5.4 |    0.2 |   0.2 |    48 |   0.113 |   0.128 | _triton_hpb_bwd_g_x_orig_kernel                                                            |
| 73 |      0.9 |    0.0 |   0.0 |     8 |   0.111 |   0.126 | elementwise_kernel<128, 4, float8_copy_kernel_cuda::lambda(Float8_e4m3fn)>                 |
| 74 |      0.1 |    0.0 |   0.0 |     1 |   0.109 |   0.109 | embedding_backward_feature_kernel<BFloat16, float, long>                                   |
| 75 |     10.3 |    0.4 |   0.3 |    96 |   0.107 |   0.120 | _triton_hpb_fwd_kernel                                                                     |
| 76 |     44.4 |    1.7 |   1.5 |   490 |   0.091 |   0.004 | vectorized_elementwise_kernel<4, CUDAFunctor_add<float>>                                   |
| 77 |      1.1 |    0.0 |   0.0 |    12 |   0.089 |   0.100 | sbtopk::gatherTopK<float, unsigned int, 2, false>                                          |
| 78 |     23.7 |    0.9 |   0.8 |   268 |   0.089 |   0.055 | vectorized_gather_kernel<16, long>                                                         |
| 79 |     88.8 |    3.5 |   3.0 |  1103 |   0.080 |   0.002 | vectorized_elementwise_kernel<8, bfloat16_copy_kernel_cuda::lambda(float)>                 |
| 80 |     48.7 |    1.9 |   1.7 |   616 |   0.079 |   0.004 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(float), 4>                     |
| 81 |      4.8 |    0.2 |   0.2 |    64 |   0.075 |   0.043 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lam... |
| 82 |      3.0 |    0.1 |   0.1 |    40 |   0.074 |   0.073 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_TNT                                             |
| 83 |      2.9 |    0.1 |   0.1 |    39 |   0.073 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsi... |
| 84 |     10.6 |    0.4 |   0.4 |   148 |   0.072 |   0.005 | Memcpy HtoD (Pinned -> Device)                                                             |
| 85 |      4.8 |    0.2 |   0.2 |    68 |   0.071 |   0.087 | multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Cop... |
| 86 |      3.4 |    0.1 |   0.1 |    48 |   0.070 |   0.054 | elementwise_kernel<128, 4, CUDAFunctor_add<BFloat16>>                                      |
| 87 |     20.0 |    0.8 |   0.7 |   292 |   0.069 |   0.044 | vectorized_elementwise_kernel<8, CUDAFunctor_add<BFloat16>>                                |
| 88 |      2.5 |    0.1 |   0.1 |    40 |   0.063 |   0.067 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN                                             |
| 89 |      2.9 |    0.1 |   0.1 |    48 |   0.061 |   0.030 | vectorized_elementwise_kernel<8, silu_backward_kernel::lambda(BFloat16, BFloat16)>         |
| 90 |      4.0 |    0.2 |   0.1 |    68 |   0.058 |   0.041 | multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>,... |
| 91 |     12.7 |    0.5 |   0.4 |   240 |   0.053 |   0.027 | vectorized_elementwise_kernel<8, BinaryFunctor<BFloat16, MulFunctor<float>>>               |
| 92 |      1.2 |    0.0 |   0.0 |    24 |   0.052 |   0.054 | postprocess_kernel_kernel                                                                  |
| 93 |      3.6 |    0.1 |   0.1 |    80 |   0.045 |   0.019 | elementwise_kernel<128, 4, masked_fill_kernel::lambda(BFloat16, bool)>                     |
| 94 |     16.9 |    0.7 |   0.6 |   388 |   0.043 |   0.008 | reduce_kernel<128, 4, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 95 |      4.2 |    0.2 |   0.1 |    96 |   0.043 |   0.021 | vectorized_elementwise_kernel<8, silu_kernel::lambda(BFloat16)>                            |
| 96 |      2.2 |    0.1 |   0.1 |    52 |   0.042 |   0.047 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4... |
| 97 |      0.5 |    0.0 |   0.0 |    12 |   0.040 |   0.039 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_nt_align4>                      |
| 98 |      0.7 |    0.0 |   0.0 |    20 |   0.036 |   0.035 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_NTT                                                   |
| 99 |      1.6 |    0.1 |   0.1 |    48 |   0.034 |   0.016 | vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cud... |
| 100 |      2.2 |    0.1 |   0.1 |    68 |   0.033 |   0.024 | reduce_kernel<128, 4, ReduceOp<BFloat16, sum_functor, unsigned int, BFloat16, 4, 8>>       |
| 101 |      1.3 |    0.1 |   0.0 |    40 |   0.032 |   0.028 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                                                   |
| 102 |     13.0 |    0.5 |   0.4 |   427 |   0.030 |   0.005 | reduce_kernel<512, 1, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 103 |      0.6 |    0.0 |   0.0 |    20 |   0.029 |   0.029 | nvjet_sm90_tst_128x128_64x6_1x2_h_bz_NTT                                                   |
| 104 |      9.6 |    0.4 |   0.3 |   356 |   0.027 |   0.010 | vectorized_elementwise_kernel<8, FillFunctor<BFloat16>>                                    |
| 105 |      0.5 |    0.0 |   0.0 |    20 |   0.026 |   0.026 | nvjet_sm90_tst_128x120_64x6_2x4_h_bz_TNT                                                   |
| 106 |      0.5 |    0.0 |   0.0 |    20 |   0.025 |   0.028 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                                                    |
| 107 |      2.4 |    0.1 |   0.1 |    96 |   0.025 |   0.014 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 3, 128, 1, 16, 8>             |
| 108 |      2.5 |    0.1 |   0.1 |   100 |   0.025 |   0.028 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64di... |
| 109 |     10.2 |    0.4 |   0.3 |   408 |   0.025 |   0.001 | vectorized_elementwise_kernel<4, AUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 110 |      0.5 |    0.0 |   0.0 |    20 |   0.025 |   0.025 | nvjet_sm90_tst_64x200_64x6_2x1_v_bz_TNT                                                    |
| 111 |      0.1 |    0.0 |   0.0 |     4 |   0.025 |   0.024 | nvjet_sm90_tst_64x80_64x11_1x4_h_bz_TNT                                                    |
| 112 |      0.6 |    0.0 |   0.0 |    24 |   0.023 |   0.026 | nvjet_sm90_tst_64x128_64x8_2x1_v_bz_NTN                                                    |
| 113 |      0.1 |    0.0 |   0.0 |     4 |   0.022 |   0.022 | nvjet_sm90_tst_64x72_64x12_1x4_h_bz_TNT                                                    |
| 114 |      0.1 |    0.0 |   0.0 |     4 |   0.020 |   0.020 | nvjet_sm90_tst_64x64_64x13_1x4_h_bz_TNT                                                    |
| 115 |      0.4 |    0.0 |   0.0 |    24 |   0.017 |   0.016 | radixSortKVInPlace<-2, -1, 32, 32, float, long, unsigned int>                              |
| 116 |      0.8 |    0.0 |   0.0 |    48 |   0.016 |   0.018 | _triton_sinkhorn_bwd_kernel                                                                |
| 117 |      1.6 |    0.1 |   0.1 |    96 |   0.016 |   0.009 | _triton_sinkhorn_fwd_kernel                                                                |
| 118 |      0.2 |    0.0 |   0.0 |    16 |   0.014 |   0.015 | elementwise_kernel<128, 4, CompareFunctor<long>>                                           |
| 119 |      1.2 |    0.0 |   0.0 |    96 |   0.012 |   0.014 | vectorized_elementwise_kernel<8, launch_clamp_scalar::lambda(BFloat16)>                    |
| 120 |      0.4 |    0.0 |   0.0 |    32 |   0.012 |   0.014 | mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>                                 |
| 121 |      0.2 |    0.0 |   0.0 |    20 |   0.011 |   0.011 | nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                                                    |
| 122 |      0.3 |    0.0 |   0.0 |    24 |   0.011 |   0.012 | indexing_backward_kernel_stride_1<float>                                                   |
| 123 |      0.5 |    0.0 |   0.0 |    52 |   0.010 |   0.009 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned long, unsigned long)>  |
| 124 |      0.1 |    0.0 |   0.0 |     8 |   0.010 |   0.010 | CatArrayBatchedCopy_alignedK_contig<OpaqueType<4u>, unsigned int, 4, 128, 1, 16>           |
| 125 |      0.2 |    0.0 |   0.0 |    24 |   0.010 |   0.011 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1... |
| 126 |      0.7 |    0.0 |   0.0 |    72 |   0.010 |   0.011 | vectorized_elementwise_kernel<8, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>        |
| 127 |      0.0 |    0.0 |   0.0 |     4 |   0.009 |   0.009 | nvjet_sm90_tst_32x64_64x16_2x1_v_bz_splitK_TNN                                             |
| 128 |      0.4 |    0.0 |   0.0 |    48 |   0.008 |   0.008 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64di... |
| 129 |      0.1 |    0.0 |   0.0 |    12 |   0.008 |   0.007 | vectorized_elementwise_kernel<2, masked_fill_kernel::lambda(long, bool)>                   |
| 130 |      0.2 |    0.0 |   0.0 |    24 |   0.008 |   0.009 | CatArrayBatchedCopy_vectorized<OpaqueType<4u>, unsigned int, 4, 128, 1, 16, 4>             |
| 131 |      0.6 |    0.0 |   0.0 |    76 |   0.008 |   0.005 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>    |
| 132 |      0.2 |    0.0 |   0.0 |    32 |   0.007 |   0.008 | mbtopk::computeDigitCumSum                                                                 |
| 133 |      0.1 |    0.0 |   0.0 |    18 |   0.007 |   0.009 | elementwise_kernel<128, 2, direct_copy_kernel_cuda>                                        |
| 134 |      0.4 |    0.0 |   0.0 |    55 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, ... |
| 135 |      0.2 |    0.0 |   0.0 |    31 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, f... |
| 136 |      0.1 |    0.0 |   0.0 |    16 |   0.007 |   0.008 | vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned... |
| 137 |      0.1 |    0.0 |   0.0 |    12 |   0.007 |   0.007 | bitonicSortKVInPlace<2, -1, 16, 16, float, long, GTOp<float, true>, unsigned int>          |
| 138 |      4.3 |    0.2 |   0.1 |   633 |   0.007 |   0.006 | at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::rad... |
| 139 |      0.2 |    0.0 |   0.0 |    24 |   0.007 |   0.007 | vectorized_elementwise_kernel<4, softplus_backward_kernel::lambda(float, float)>           |
| 140 |      0.1 |    0.0 |   0.0 |    20 |   0.006 |   0.006 | elementwise_kernel<128, 4, CompareFunctor<long>> #2                                        |
| 141 |      0.3 |    0.0 |   0.0 |    48 |   0.006 |   0.006 | vectorized_elementwise_kernel<8, compare_scalar_kernel<long>::lambda(long)>                |
| 142 |      0.1 |    0.0 |   0.0 |    10 |   0.006 |   0.009 | cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, ... |
| 143 |      0.0 |    0.0 |   0.0 |     3 |   0.006 |   0.006 | at_cuda_detail::cub::unique_by_key::DeviceUniqueByKeySweepKernel<at_cuda_detail::cub::u... |
| 144 |      0.0 |    0.0 |   0.0 |     3 |   0.006 |   0.006 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long,... |
| 145 |      0.2 |    0.0 |   0.0 |    40 |   0.006 |   0.006 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 146 |      0.0 |    0.0 |   0.0 |     2 |   0.005 |   0.005 | lpnorm_cleanup<float, (NormType)2, float, true, float>                                     |
| 147 |      0.0 |    0.0 |   0.0 |     8 |   0.005 |   0.005 | unrolled_elementwise_kernel<BinaryFunctor<float, MulFunctor<float>>, 4>                    |
| 148 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | vectorized_elementwise_kernel<4, softplus_kernel::lambda(float)>                           |
| 149 |      0.1 |    0.0 |   0.0 |    28 |   0.005 |   0.004 | elementwise_kernel<128, 2, CUDAFunctor_add<long>>                                          |
| 150 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | _fill_indices_kernel                                                                       |
| 151 |      0.4 |    0.0 |   0.0 |    96 |   0.005 |   0.004 | kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor... |
| 152 |      0.0 |    0.0 |   0.0 |     3 |   0.004 |   0.004 | krn_partials_per_segment<long>                                                             |
| 153 |      0.3 |    0.0 |   0.0 |    80 |   0.004 |   0.004 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div... |
| 154 |      0.2 |    0.0 |   0.0 |    52 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lam... |
| 155 |      0.4 |    0.0 |   0.0 |    88 |   0.004 |   0.004 | vectorized_elementwise_kernel<8, compare_scalar_kernel<int>>                               |
| 156 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | reduce_kernel<128, 4, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 157 |      1.2 |    0.0 |   0.0 |   288 |   0.004 |   0.003 | elementwise_kernel<128, 4, CUDAFunctor_add<float>>                                         |
| 158 |      0.1 |    0.0 |   0.0 |    16 |   0.004 |   0.004 | at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_ke... |
| 159 |      0.3 |    0.0 |   0.0 |    88 |   0.004 |   0.004 | reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned i... |
| 160 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lam... |
| 161 |      0.0 |    0.0 |   0.0 |     3 |   0.004 |   0.004 | compute_num_of_partial_segments<long>                                                      |
| 162 |      0.0 |    0.0 |   0.0 |     3 |   0.003 |   0.004 | at_cuda_detail::cub::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<u... |
| 163 |      0.1 |    0.0 |   0.0 |    40 |   0.003 |   0.003 | cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, fl... |
| 164 |      0.0 |    0.0 |   0.0 |     1 |   0.003 |   0.003 | reduce_kernel<512, 1, ReduceOp<float, WelfordOps<float, float, int, cuda::__4::pair<flo... |
| 165 |      0.0 |    0.0 |   0.0 |     8 |   0.003 |   0.003 | vectorized_elementwise_kernel<8, BinaryFunctor<bool, BitwiseOrFunctor<bool>>>              |
| 166 |      0.0 |    0.0 |   0.0 |     8 |   0.003 |   0.003 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, ... |
| 167 |      0.0 |    0.0 |   0.0 |     3 |   0.003 |   0.002 | krn_partial_segment_offset<long>                                                           |
| 168 |      0.9 |    0.0 |   0.0 |   328 |   0.003 |   0.002 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(long), 4>                      |
| 169 |      0.2 |    0.0 |   0.0 |    58 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, BinaryFunctor<float, DivFunctor<float>>>                  |
| 170 |      0.5 |    0.0 |   0.0 |   180 |   0.003 |   0.001 | unrolled_elementwise_kernel<direct_copy_kernel_cuda, 4>                                    |
| 171 |      0.1 |    0.0 |   0.0 |    50 |   0.003 |   0.003 | vectorized_elementwise_kernel<4, sqrt_kernel_cuda::lambda(float)>                          |
| 172 |      0.5 |    0.0 |   0.0 |   171 |   0.003 |   0.002 | at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::ra... |
| 173 |      0.0 |    0.0 |   0.0 |     6 |   0.003 |   0.003 | reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, or_kernel_cuda::lambda(bool, ... |
| 174 |      0.9 |    0.0 |   0.0 |   330 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, FillFunctor<float>>                                       |
| 175 |      0.3 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda:... |
| 176 |      0.0 |    0.0 |   0.0 |    16 |   0.002 |   0.003 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o128i64div... |
| 177 |      0.3 |    0.0 |   0.0 |   116 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pinned)                                                             |
| 178 |      0.5 |    0.0 |   0.0 |   200 |   0.002 |   0.001 | vectorized_elementwise_kernel<2, CUDAFunctorOnSelf_add<long>>                              |
| 179 |      0.1 |    0.0 |   0.0 |    32 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pageable)                                                           |
| 180 |      0.3 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 181 |      0.0 |    0.0 |   0.0 |     8 |   0.002 |   0.002 | compute_cuda_kernel<long>                                                                  |
| 182 |      0.0 |    0.0 |   0.0 |     1 |   0.002 |   0.002 | multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarFunctor<float, 1, 1, 0>,... |
| 183 |      0.4 |    0.0 |   0.0 |   196 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, sigmoid_kernel_cuda::lambda(float)>                       |
| 184 |      0.1 |    0.0 |   0.0 |    24 |   0.002 |   0.002 | unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4>                                     |
| 185 |      0.0 |    0.0 |   0.0 |     2 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, and_kernel_cuda::lambda(bool,... |
| 186 |      0.0 |    0.0 |   0.0 |     1 |   0.002 |   0.002 | elementwise_kernel<128, 4, CompareFunctor<float>>                                          |
| 187 |      0.1 |    0.0 |   0.0 |    33 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int... |
| 188 |      0.1 |    0.0 |   0.0 |    39 |   0.002 |   0.002 | unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory:... |
| 189 |      0.3 |    0.0 |   0.0 |   136 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda:... |
| 190 |      0.2 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda:... |
| 191 |      0.0 |    0.0 |   0.0 |     8 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<long, func_wrapper_t<long, MaxNanFunctor<long>>, unsigne... |
| 192 |      0.1 |    0.0 |   0.0 |    56 |   0.002 |   0.002 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, ... |
| 193 |      0.3 |    0.0 |   0.0 |   168 |   0.002 |   0.002 | elementwise_kernel<128, 2, BinaryFunctor<float, DivFunctor<float>>>                        |
| 194 |      0.1 |    0.0 |   0.0 |    68 |   0.002 |   0.001 | unrolled_elementwise_kernel<CUDAFunctor_add<long>, 4>                                      |
| 195 |      0.0 |    0.0 |   0.0 |     3 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, NormTwoOps<float, float, float, true>, unsigned i... |
| 196 |      0.1 |    0.0 |   0.0 |    72 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, BUnaryFunctor<int, int, int, div_floor_kernel_cuda::la... |
| 197 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |
| 198 |      0.0 |    0.0 |   0.0 |     4 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, log_kernel_cuda::lambda(float)>                           |
| 199 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | CatArrayBatchedCopy_vectorized<OpaqueType<4u>, unsigned int, 3, 128, 1, 16, 4>             |
| 200 |      0.2 |    0.0 |   0.0 |   112 |   0.002 |   0.001 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long,... |

- 6: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 7: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 8: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 10: `multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<float, 4, (ADAM_MODE)1, false>, float const*, double, double, double, dou...`
- 14: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 26: `multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, 1, 0>, multiplies<float>, float*, float>`
- 27: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_None_tensorptrbf16gmemalign16o512i64div81_None_tensorptrf32...`
- 28: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div41_None_None_None_tensorptrf32gmemalign16o512i64div41_None_tensorptrf32...`
- 38: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16384i64div41_None_tensorptrf32gmemalign16o16384i64div41_None_tensorptrf...`
- 44: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64div41_None_None_None_tensorptrf32gmemalign16o16384i64div41_None_tensorpt...`
- 54: `elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned short)>`
- 55: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas`
- 58: `multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, float, 1, 1, 0>, float*, int>`
- 71: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas`
- 81: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 83: `reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsigned int, float, 4, 4>>`
- 85: `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Copy<BFloat16, BFloat16>>`
- 90: `multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>, Copy<BFloat16, float>>`
- 96: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_1409601_tensorptrbf16gmemalig...`
- 99: `vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cuda::lambda(bool, bool)>>`
- 108: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_2409601_None_None_tensorptrbf16gmemal...`
- 125: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_1102401_tensorptrbf16gmemalig...`
- 128: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_4102401_None_None_tensorptrbf16gmemal...`
- 134: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, OpaqueType<4>, long>::operator()<TensorAssign>>`
- 135: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, float, long>::operator()<ReduceAdd>>`
- 136: `vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>`
- 138: `at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy10...`
- 142: `cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>`
- 143: `at_cuda_detail::cub::unique_by_key::DeviceUniqueByKeySweepKernel<at_cuda_detail::cub::unique_by_key::policy_hub<long, long>::Policy1000, long const*, thrust:...`
- 144: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long, long, long, unsigned int, cub::SumOp<long>>::Policy1000, long const*,...`
- 145: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_251201_tensorptrbf16gmemalign1...`
- 151: `kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor_histogram<float, float, false>(Tensor, Tensor, Tensor, long, Accumula...`
- 153: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_451201_None_None_tensorptrbf16gmemalig...`
- 154: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lambda(char*, char const*, long)>>`
- 158: `at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_key::policy_hub<thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::t...`
- 159: `reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned int, int, 4, 4>>`
- 160: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lambda(char*, char const*, long)>>`
- 162: `at_cuda_detail::cub::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<unsigned long long, true>, long*>`
- 163: `cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>`
- 164: `reduce_kernel<512, 1, ReduceOp<float, WelfordOps<float, float, int, cuda::__4::pair<float, float>>, unsigned int, float, 2, 2>>`
- 166: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, OpaqueType<8>, long>::operator()<TensorAssign>>`
- 172: `at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy1...`
- 173: `reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, or_kernel_cuda::lambda(bool, bool)>, unsigned int, bool, 4, 4>>`
- 175: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda::lambda(long, long)>>`
- 176: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o128i64div81_tensorptrbf16_gmem_align16_o_812801_None_None_tensorptrbf16gmemalig...`
- 182: `multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarFunctor<float, 1, 1, 0>, plus<float>, float>`
- 185: `reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, and_kernel_cuda::lambda(bool, bool)>, unsigned int, bool, 4, 4>>`
- 187: `reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int, float, 4, 4>>`
- 188: `unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory::StoreWithoutCast>`
- 189: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda::lambda(long, long)>>`
- 190: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda::lambda(long, long)>>`
- 191: `reduce_kernel<512, 1, ReduceOp<long, func_wrapper_t<long, MaxNanFunctor<long>>, unsigned int, long, 4, 4>>`
- 192: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, int, int, unsigned int, plus<int>>::Policy1000, int const*, int*, at_c...`
- 195: `reduce_kernel<512, 1, ReduceOp<float, NormTwoOps<float, float, float, true>, unsigned int, float, 4, 4>>`
- 196: `vectorized_elementwise_kernel<4, BUnaryFunctor<int, int, int, div_floor_kernel_cuda::lambda(int, int)>>`
- 197: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<cute::C<1>, long, cut...`
- 200: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long, long, long, unsigned int, plus<long>>::Policy1000, long const*, long*...`

## NCCL (durations include waiting for peers)

|  # | total ms | % busy | % win | calls | mean ms |  med ms | name                                                                                       |
|----|----------|--------|-------|-------|---------|---------|--------------------------------------------------------------------------------------------|
|  1 |    418.5 |   16.4 |  14.2 |   192 |   2.180 |   1.078 | ncclDevKernel_SendRecv                                                                     |
|  2 |    261.4 |   10.2 |   8.9 |   164 |   1.594 |   0.564 | ncclDevKernel_AllGather_RING_LL                                                            |
|  3 |    146.5 |    5.7 |   5.0 |    40 |   3.662 |   0.242 | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL                                               |
|  4 |    112.2 |    4.4 |   3.8 |    36 |   3.117 |   2.560 | ncclDevKernel_ReduceScatter_Sum_f32_RING_LL                                                |
|  5 |      2.5 |    0.1 |   0.1 |    16 |   0.154 |   0.040 | ncclDevKernel_AllReduce_Sum_f32_RING_LL                                                    |
|  6 |      0.4 |    0.0 |   0.0 |     2 |   0.217 |   0.217 | ncclDevKernel_AllReduce_Sum_u64_RING_LL                                                    |

## By category (first matching `--category` rule on the raw kernel name)

| category | total ms | union ms | % busy | % win | calls |
|---|---|---|---|---|---|
| sparse_attn | 136.0 | 136.0 | 5.3 | 4.6 | 120 |
| other | 2673.2 | 2419.8 | 104.6 | 90.7 | 20418 |

union ms = time covered by at least one kernel of the category (overlap across streams counted once).

Legend: total = summed durations in the window; % busy = total / GPU busy time and % win = total / window (shares
can sum past 100 when streams overlap); calls = launches; mean/med = per launch. Names drop namespaces, lambdas,
and argument lists (mangled names list their first identifiers); `#n` separates distinct kernels that shorten
alike; truncated labels are listed under each table by rank; raw names are in the CSV. Site = `[<nearest non-aten
enclosing op> >] <innermost cpu_op> <first 3 Input Dims>`; a site can launch several kernels, so its mean/median
mix them.
