# Top GPU kernels: `/home/garrett/tmp/dsv4-cfm-pr/traces/traceC/trace_0.json.gz`

- window: 2735.8 ms (`forward` [-4] to last GPU event), stream filter: none, grouped by name
- GPU busy (union over selected streams): 2407.4 ms (88.0% of window); non-NCCL busy 1709.5 ms (62.5%)
- GPU idle: 328.4 ms (12.0%); exposed NCCL (no non-NCCL kernel running): 697.9 ms (25.5%)
- summed durations: non-NCCL 1813.2 ms, NCCL 845.6 ms; 21192 GPU events
- kernel ms per stream (all streams, in window): 7: 2175.2, 32: 315.3, 28: 95.3, 23: 64.3, 24: 8.7

## Non-NCCL, by total time (top 200)

|  # | total ms | % busy | % win | calls | mean ms |  med ms | name                                                                                       |
|----|----------|--------|-------|-------|---------|---------|--------------------------------------------------------------------------------------------|
|  1 |    171.3 |    7.1 |   6.3 |    60 |   2.855 |   2.312 | chunk_cat_cuda_kernel<float, BFloat16>                                                     |
|  2 |    117.1 |    4.9 |   4.3 |    96 |   1.220 |   1.048 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  3 |     88.7 |    3.7 |   3.2 |  1103 |   0.080 |   0.002 | vectorized_elementwise_kernel<8, bfloat16_copy_kernel_cuda::lambda(float)>                 |
|  4 |     82.6 |    3.4 |   3.0 |   481 |   0.172 |   0.008 | Memcpy DtoD (Device -> Device)                                                             |
|  5 |     78.5 |    3.3 |   2.9 |   591 |   0.133 |   0.004 | elementwise_kernel<128, 4, BinaryFunctor<float, MulFunctor<float>>>                        |
|  6 |     61.0 |    2.5 |   2.2 |    48 |   1.272 |   1.068 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  7 |     58.2 |    2.4 |   2.1 |    48 |   1.212 |   1.026 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  8 |     48.8 |    2.0 |   1.8 |   616 |   0.079 |   0.004 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(float), 4>                     |
|  9 |     46.5 |    1.9 |   1.7 |   184 |   0.253 |   0.148 | nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN                                             |
| 10 |     46.3 |    1.9 |   1.7 |   276 |   0.168 |   0.003 | elementwise_kernel<128, 2, BinaryFunctor<float, MulFunctor<float>>>                        |
| 11 |     44.3 |    1.8 |   1.6 |   490 |   0.090 |   0.004 | vectorized_elementwise_kernel<4, CUDAFunctor_add<float>>                                   |
| 12 |     36.7 |    1.5 |   1.3 |    24 |   1.531 |   1.168 | kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm... |
| 13 |     36.3 |    1.5 |   1.3 |    85 |   0.427 |   0.754 | vectorized_elementwise_kernel<4, BUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 14 |     35.3 |    1.5 |   1.3 |    35 |   1.008 |   1.020 | multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<flo... |
| 15 |     34.1 |    1.4 |   1.2 |   296 |   0.115 |   0.062 | _triton_rope_kernel                                                                        |
| 16 |     31.6 |    1.3 |   1.2 |   164 |   0.193 |   0.169 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT                                             |
| 17 |     30.4 |    1.3 |   1.1 |    48 |   0.633 |   0.664 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>:... |
| 18 |     27.2 |    1.1 |   1.0 |    30 |   0.907 |   0.251 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT                                             |
| 19 |     26.5 |    1.1 |   1.0 |    48 |   0.551 |   0.505 | sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true, 0>, sm90::fwd::K... |
| 20 |     25.8 |    1.1 |   0.9 |   148 |   0.174 |   0.141 | elementwise_kernel<128, 4, direct_copy_kernel_cuda::lambda(BFloat16)>                      |
| 21 |     25.3 |    1.1 |   0.9 |    96 |   0.264 |   0.128 | indexing_backward_kernel<BFloat16, 4>                                                      |
| 22 |     23.9 |    1.0 |   0.9 |   102 |   0.234 |   0.054 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT                                             |
| 23 |     23.7 |    1.0 |   0.9 |   268 |   0.089 |   0.055 | vectorized_gather_kernel<16, long>                                                         |
| 24 |     20.0 |    0.8 |   0.7 |   292 |   0.069 |   0.044 | vectorized_elementwise_kernel<8, CUDAFunctor_add<BFloat16>>                                |
| 25 |     18.3 |    0.8 |   0.7 |    96 |   0.191 |   0.201 | elementwise_kernel<128, 4, launch_clamp_scalar::lambda(BFloat16)>                          |
| 26 |     18.1 |    0.8 |   0.7 |    90 |   0.201 |   0.003 | elementwise_kernel<128, 2, CUDAFunctor_add<float>>                                         |
| 27 |     17.6 |    0.7 |   0.6 |     3 |   5.883 |   7.229 | nvjet_sm90_tst_128x256_64x4_2x1_v_badd_coopA_NTN                                           |
| 28 |     17.6 |    0.7 |   0.6 |   100 |   0.176 |   0.171 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64di... |
| 29 |     16.8 |    0.7 |   0.6 |   388 |   0.043 |   0.008 | reduce_kernel<128, 4, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 30 |     16.1 |    0.7 |   0.6 |    17 |   0.948 |   0.587 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_TNT                                             |
| 31 |     15.7 |    0.7 |   0.6 |    28 |   0.561 |   0.586 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT                                             |
| 32 |     15.6 |    0.6 |   0.6 |    68 |   0.229 |   0.169 | split_with_sizes_copy_out_contiguous_no_cast_kernel                                        |
| 33 |     14.1 |    0.6 |   0.5 |    48 |   0.294 |   0.333 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div4... |
| 34 |     14.0 |    0.6 |   0.5 |    96 |   0.146 |   0.069 | elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned sho... |
| 35 |     13.8 |    0.6 |   0.5 |    36 |   0.383 |   0.502 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_TNT                                             |
| 36 |     13.0 |    0.5 |   0.5 |   427 |   0.030 |   0.005 | reduce_kernel<512, 1, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 37 |     12.8 |    0.5 |   0.5 |   240 |   0.053 |   0.027 | vectorized_elementwise_kernel<8, BinaryFunctor<BFloat16, MulFunctor<float>>>               |
| 38 |     12.2 |    0.5 |   0.4 |    48 |   0.253 |   0.149 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_TNT                                             |
| 39 |     12.1 |    0.5 |   0.4 |    72 |   0.168 |   0.176 | elementwise_kernel<128, 4, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>              |
| 40 |     11.9 |    0.5 |   0.4 |    45 |   0.265 |   0.082 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_NNT                                             |
| 41 |     11.4 |    0.5 |   0.4 |    14 |   0.811 |   0.846 | vectorized_elementwise_kernel<4, exp_kernel_cuda::lambda(float)>                           |
| 42 |     11.2 |    0.5 |   0.4 |    92 |   0.121 |   0.002 | vectorized_elementwise_kernel<4, BinaryFunctor<float, MulFunctor<float>>>                  |
| 43 |     11.1 |    0.5 |   0.4 |    52 |   0.214 |   0.247 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16... |
| 44 |     10.7 |    0.4 |   0.4 |   148 |   0.073 |   0.005 | Memcpy HtoD (Pinned -> Device)                                                             |
| 45 |     10.4 |    0.4 |   0.4 |     1 |  10.391 |  10.391 | nvjet_sm90_tst_320x128_64x3_1x2_h_badd_coopB_NTT                                           |
| 46 |     10.4 |    0.4 |   0.4 |    35 |   0.297 |   0.300 | multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, ... |
| 47 |     10.3 |    0.4 |   0.4 |    96 |   0.107 |   0.121 | _triton_hpb_fwd_kernel                                                                     |
| 48 |     10.3 |    0.4 |   0.4 |   480 |   0.021 |   0.001 | vectorized_elementwise_kernel<4, AUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 49 |      9.8 |    0.4 |   0.4 |    18 |   0.542 |   0.705 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_NNT                                             |
| 50 |      9.7 |    0.4 |   0.4 |    39 |   0.250 |   0.072 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_NNT                                             |
| 51 |      9.6 |    0.4 |   0.4 |   356 |   0.027 |   0.011 | vectorized_elementwise_kernel<8, FillFunctor<BFloat16>>                                    |
| 52 |      9.2 |    0.4 |   0.3 |     1 |   9.177 |   9.177 | nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT                                             |
| 53 |      8.4 |    0.4 |   0.3 |    48 |   0.176 |   0.197 | tma_scatter_add_kernel<BFloat16, long>                                                     |
| 54 |      8.3 |    0.3 |   0.3 |    40 |   0.207 |   0.088 | elementwise_kernel<128, 4, BinaryFunctor<BFloat16, MulFunctor<float>>>                     |
| 55 |      7.8 |    0.3 |   0.3 |    64 |   0.122 |   0.069 | CatArrayBatchedCopy<OpaqueType<2u>, unsigned int, 4, 64, 64>                               |
| 56 |      7.3 |    0.3 |   0.3 |    48 |   0.152 |   0.157 | cunn_SpatialSoftMaxForward<float, float, float, int, SoftMaxForwardEpilogue>               |
| 57 |      7.2 |    0.3 |   0.3 |    50 |   0.143 |   0.133 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 58 |      7.1 |    0.3 |   0.3 |    24 |   0.294 |   0.330 | preprocess_kernel_kernel                                                                   |
| 59 |      7.0 |    0.3 |   0.3 |    24 |   0.292 |   0.328 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 60 |      6.7 |    0.3 |   0.2 |    48 |   0.139 |   0.144 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 1, 128, 1, 16, 8>             |
| 61 |      6.5 |    0.3 |   0.2 |    52 |   0.124 |   0.142 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>                      |
| 62 |      5.9 |    0.2 |   0.2 |    50 |   0.118 |   0.105 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 63 |      5.6 |    0.2 |   0.2 |    48 |   0.116 |   0.131 | _triton_hpb_bwd_g_hp_hr_kernel                                                             |
| 64 |      5.4 |    0.2 |   0.2 |    48 |   0.112 |   0.127 | _triton_hpb_bwd_g_x_orig_kernel                                                            |
| 65 |      5.0 |    0.2 |   0.2 |    40 |   0.125 |   0.117 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_32x3_nt_align4>                      |
| 66 |      4.9 |    0.2 |   0.2 |    35 |   0.139 |   0.141 | multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, floa... |
| 67 |      4.8 |    0.2 |   0.2 |    68 |   0.071 |   0.089 | multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Cop... |
| 68 |      4.8 |    0.2 |   0.2 |    32 |   0.149 |   0.171 | mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>                      |
| 69 |      4.8 |    0.2 |   0.2 |    64 |   0.075 |   0.043 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lam... |
| 70 |      4.5 |    0.2 |   0.2 |    24 |   0.186 |   0.207 | kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm... |
| 71 |      4.3 |    0.2 |   0.2 |   633 |   0.007 |   0.006 | at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::rad... |
| 72 |      4.2 |    0.2 |   0.2 |    96 |   0.043 |   0.021 | vectorized_elementwise_kernel<8, silu_kernel::lambda(BFloat16)>                            |
| 73 |      3.9 |    0.2 |   0.1 |    68 |   0.057 |   0.040 | multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>,... |
| 74 |      3.7 |    0.2 |   0.1 |     6 |   0.617 |   0.625 | nvjet_sm90_tst_128x304_64x3_2x1_v_bz_coopA_NNT                                             |
| 75 |      3.6 |    0.1 |   0.1 |    80 |   0.045 |   0.019 | elementwise_kernel<128, 4, masked_fill_kernel::lambda(BFloat16, bool)>                     |
| 76 |      3.5 |    0.1 |   0.1 |     6 |   0.576 |   0.585 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_NNT                                             |
| 77 |      3.4 |    0.1 |   0.1 |     6 |   0.572 |   0.586 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_NNT                                             |
| 78 |      3.4 |    0.1 |   0.1 |    48 |   0.071 |   0.055 | elementwise_kernel<128, 4, CUDAFunctor_add<BFloat16>>                                      |
| 79 |      3.3 |    0.1 |   0.1 |    24 |   0.138 |   0.146 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 2, 128, 1, 16, 8>             |
| 80 |      3.3 |    0.1 |   0.1 |     6 |   0.544 |   0.549 | nvjet_sm90_tst_256x160_64x4_2x1_v_bz_coopA_NNT                                             |
| 81 |      3.2 |    0.1 |   0.1 |    24 |   0.133 |   0.133 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_TNT                                             |
| 82 |      3.2 |    0.1 |   0.1 |     6 |   0.525 |   0.538 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_NNN                                             |
| 83 |      3.1 |    0.1 |   0.1 |    40 |   0.079 |   0.074 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_TNT                                             |
| 84 |      3.1 |    0.1 |   0.1 |     6 |   0.514 |   0.521 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_NNT                                             |
| 85 |      2.9 |    0.1 |   0.1 |    48 |   0.061 |   0.030 | vectorized_elementwise_kernel<8, silu_backward_kernel::lambda(BFloat16, BFloat16)>         |
| 86 |      2.9 |    0.1 |   0.1 |    39 |   0.073 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsi... |
| 87 |      2.6 |    0.1 |   0.1 |    40 |   0.064 |   0.066 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN                                             |
| 88 |      2.5 |    0.1 |   0.1 |   100 |   0.025 |   0.028 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64di... |
| 89 |      2.4 |    0.1 |   0.1 |    96 |   0.025 |   0.014 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 3, 128, 1, 16, 8>             |
| 90 |      2.3 |    0.1 |   0.1 |    16 |   0.146 |   0.140 | cunn_SpatialSoftMaxBackward<float, float, float, SoftMaxBackwardEpilogue>                  |
| 91 |      2.2 |    0.1 |   0.1 |    68 |   0.033 |   0.024 | reduce_kernel<128, 4, ReduceOp<BFloat16, sum_functor, unsigned int, BFloat16, 4, 8>>       |
| 92 |      2.2 |    0.1 |   0.1 |    52 |   0.042 |   0.048 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4... |
| 93 |      2.2 |    0.1 |   0.1 |  1674 |   0.001 |   0.001 | Memset (Device)                                                                            |
| 94 |      2.1 |    0.1 |   0.1 |     8 |   0.265 |   0.297 | mbtopk::gatherTopK<float, unsigned int, 2>                                                 |
| 95 |      2.1 |    0.1 |   0.1 |     8 |   0.263 |   0.289 | _triton_fp8_indexer_kernel                                                                 |
| 96 |      2.0 |    0.1 |   0.1 |    24 |   0.082 |   0.091 | kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm... |
| 97 |      1.9 |    0.1 |   0.1 |    16 |   0.117 |   0.045 | _per_token_group_quant_fp8                                                                 |
| 98 |      1.7 |    0.1 |   0.1 |    96 |   0.017 |   0.009 | _triton_sinkhorn_fwd_kernel                                                                |
| 99 |      1.6 |    0.1 |   0.1 |    48 |   0.034 |   0.016 | vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cud... |
| 100 |      1.4 |    0.1 |   0.1 |   196 |   0.007 |   0.004 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>    |
| 101 |      1.2 |    0.1 |   0.0 |   288 |   0.004 |   0.003 | elementwise_kernel<128, 4, CUDAFunctor_add<float>>                                         |
| 102 |      1.2 |    0.0 |   0.0 |    40 |   0.030 |   0.027 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                                                   |
| 103 |      1.2 |    0.0 |   0.0 |    96 |   0.012 |   0.014 | vectorized_elementwise_kernel<8, launch_clamp_scalar::lambda(BFloat16)>                    |
| 104 |      1.1 |    0.0 |   0.0 |     8 |   0.140 |   0.158 | radixSortKVInPlace<2, -1, 32, 32, float, long, unsigned int>                               |
| 105 |      1.1 |    0.0 |   0.0 |    12 |   0.090 |   0.101 | sbtopk::gatherTopK<float, unsigned int, 2, false>                                          |
| 106 |      1.0 |    0.0 |   0.0 |     6 |   0.165 |   0.166 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_splitK_NNT                                      |
| 107 |      0.9 |    0.0 |   0.0 |   328 |   0.003 |   0.002 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(long), 4>                      |
| 108 |      0.9 |    0.0 |   0.0 |     8 |   0.111 |   0.126 | elementwise_kernel<128, 4, float8_copy_kernel_cuda::lambda(Float8_e4m3fn)>                 |
| 109 |      0.9 |    0.0 |   0.0 |   330 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, FillFunctor<float>>                                       |
| 110 |      0.8 |    0.0 |   0.0 |    48 |   0.016 |   0.018 | _triton_sinkhorn_bwd_kernel                                                                |
| 111 |      0.8 |    0.0 |   0.0 |     6 |   0.131 |   0.132 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_NNT                                             |
| 112 |      0.7 |    0.0 |   0.0 |    20 |   0.036 |   0.036 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_NTT                                                   |
| 113 |      0.7 |    0.0 |   0.0 |    72 |   0.010 |   0.011 | vectorized_elementwise_kernel<8, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>        |
| 114 |      0.6 |    0.0 |   0.0 |    20 |   0.029 |   0.029 | nvjet_sm90_tst_128x128_64x6_1x2_h_bz_NTT                                                   |
| 115 |      0.6 |    0.0 |   0.0 |    72 |   0.008 |   0.007 | elementwise_kernel<128, 2, CUDAFunctor_add<int>>                                           |
| 116 |      0.6 |    0.0 |   0.0 |    24 |   0.023 |   0.026 | nvjet_sm90_tst_64x128_64x8_2x1_v_bz_NTN                                                    |
| 117 |      0.5 |    0.0 |   0.0 |    52 |   0.011 |   0.009 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned long, unsigned long)>  |
| 118 |      0.5 |    0.0 |   0.0 |    20 |   0.026 |   0.026 | nvjet_sm90_tst_128x120_64x6_2x4_h_bz_TNT                                                   |
| 119 |      0.5 |    0.0 |   0.0 |     3 |   0.173 |   0.169 | compute_grad_weight<BFloat16, long>                                                        |
| 120 |      0.5 |    0.0 |   0.0 |    20 |   0.025 |   0.025 | nvjet_sm90_tst_64x200_64x6_2x1_v_bz_TNT                                                    |
| 121 |      0.5 |    0.0 |   0.0 |   180 |   0.003 |   0.001 | unrolled_elementwise_kernel<direct_copy_kernel_cuda, 4>                                    |
| 122 |      0.5 |    0.0 |   0.0 |   200 |   0.002 |   0.001 | vectorized_elementwise_kernel<2, CUDAFunctorOnSelf_add<long>>                              |
| 123 |      0.5 |    0.0 |   0.0 |    20 |   0.024 |   0.028 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                                                    |
| 124 |      0.5 |    0.0 |   0.0 |    12 |   0.039 |   0.039 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_nt_align4>                      |
| 125 |      0.5 |    0.0 |   0.0 |   196 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, sigmoid_kernel_cuda::lambda(float)>                       |
| 126 |      0.5 |    0.0 |   0.0 |   171 |   0.003 |   0.002 | at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::ra... |
| 127 |      0.4 |    0.0 |   0.0 |    96 |   0.005 |   0.004 | kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor... |
| 128 |      0.4 |    0.0 |   0.0 |    24 |   0.017 |   0.016 | radixSortKVInPlace<-2, -1, 32, 32, float, long, unsigned int>                              |
| 129 |      0.4 |    0.0 |   0.0 |    55 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, ... |
| 130 |      0.4 |    0.0 |   0.0 |     3 |   0.130 |   0.128 | sum_and_scatter<BFloat16, long>                                                            |
| 131 |      0.4 |    0.0 |   0.0 |    32 |   0.012 |   0.013 | mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>                                 |
| 132 |      0.4 |    0.0 |   0.0 |    48 |   0.008 |   0.008 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64di... |
| 133 |      0.3 |    0.0 |   0.0 |    80 |   0.004 |   0.004 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div... |
| 134 |      0.3 |    0.0 |   0.0 |    88 |   0.004 |   0.004 | reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned i... |
| 135 |      0.3 |    0.0 |   0.0 |   199 |   0.002 |   0.001 | vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<float>>                             |
| 136 |      0.3 |    0.0 |   0.0 |   168 |   0.002 |   0.002 | elementwise_kernel<128, 2, BinaryFunctor<float, DivFunctor<float>>>                        |
| 137 |      0.3 |    0.0 |   0.0 |   120 |   0.003 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda:... |
| 138 |      0.3 |    0.0 |   0.0 |    48 |   0.006 |   0.006 | vectorized_elementwise_kernel<8, compare_scalar_kernel<long>::lambda(long)>                |
| 139 |      0.3 |    0.0 |   0.0 |    88 |   0.003 |   0.004 | vectorized_elementwise_kernel<8, compare_scalar_kernel<int>>                               |
| 140 |      0.3 |    0.0 |   0.0 |   116 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pinned)                                                             |
| 141 |      0.3 |    0.0 |   0.0 |   226 |   0.001 |   0.001 | vectorized_elementwise_kernel<4, FillFunctor<int>>                                         |
| 142 |      0.3 |    0.0 |   0.0 |   224 |   0.001 |   0.001 | vectorized_elementwise_kernel<2, AUnaryFunctor<long, long, long, MulFunctor<long>>>        |
| 143 |      0.3 |    0.0 |   0.0 |   136 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda:... |
| 144 |      0.3 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 145 |      0.3 |    0.0 |   0.0 |    24 |   0.011 |   0.012 | indexing_backward_kernel_stride_1<float>                                                   |
| 146 |      0.2 |    0.0 |   0.0 |    24 |   0.010 |   0.011 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1... |
| 147 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | vectorized_elementwise_kernel<4, softplus_kernel::lambda(float)>                           |
| 148 |      0.2 |    0.0 |   0.0 |    32 |   0.007 |   0.008 | mbtopk::computeDigitCumSum                                                                 |
| 149 |      0.2 |    0.0 |   0.0 |    40 |   0.006 |   0.006 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 150 |      0.2 |    0.0 |   0.0 |   227 |   0.001 |   0.001 | elementwise_kernel_with_index<int, arange_cuda_out::lambda(long)> #2                       |
| 151 |      0.2 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda:... |
| 152 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | _fill_indices_kernel                                                                       |
| 153 |      0.2 |    0.0 |   0.0 |    31 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, f... |
| 154 |      0.2 |    0.0 |   0.0 |    16 |   0.014 |   0.015 | elementwise_kernel<128, 4, CompareFunctor<long>>                                           |
| 155 |      0.2 |    0.0 |   0.0 |    20 |   0.011 |   0.011 | nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                                                    |
| 156 |      0.2 |    0.0 |   0.0 |    52 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lam... |
| 157 |      0.2 |    0.0 |   0.0 |   171 |   0.001 |   0.001 | at_cuda_detail::cub::radix_sort::DeviceRadixSortExclusiveSumKernel<at_cuda_detail::cub:... |
| 158 |      0.2 |    0.0 |   0.0 |   112 |   0.002 |   0.002 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long,... |
| 159 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | reduce_kernel<128, 4, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 160 |      0.2 |    0.0 |   0.0 |    24 |   0.008 |   0.009 | CatArrayBatchedCopy_vectorized<OpaqueType<4u>, unsigned int, 4, 128, 1, 16, 4>             |
| 161 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lam... |
| 162 |      0.2 |    0.0 |   0.0 |    58 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, BinaryFunctor<float, DivFunctor<float>>>                  |
| 163 |      0.2 |    0.0 |   0.0 |    24 |   0.007 |   0.007 | vectorized_elementwise_kernel<4, softplus_backward_kernel::lambda(float, float)>           |
| 164 |      0.2 |    0.0 |   0.0 |   100 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, sigmoid_backward_kernel_cuda::lambda(float, float)>       |
| 165 |      0.2 |    0.0 |   0.0 |    96 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |
| 166 |      0.1 |    0.0 |   0.0 |    48 |   0.003 |   0.003 | elementwise_kernel<128, 2, logaddexp_kernel_cuda::lambda(float, float)>                    |
| 167 |      0.1 |    0.0 |   0.0 |    28 |   0.005 |   0.004 | elementwise_kernel<128, 2, CUDAFunctor_add<long>>                                          |
| 168 |      0.1 |    0.0 |   0.0 |    50 |   0.003 |   0.003 | vectorized_elementwise_kernel<4, sqrt_kernel_cuda::lambda(float)>                          |
| 169 |      0.1 |    0.0 |   0.0 |    18 |   0.007 |   0.009 | elementwise_kernel<128, 2, direct_copy_kernel_cuda>                                        |
| 170 |      0.1 |    0.0 |   0.0 |    40 |   0.003 |   0.003 | cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, fl... |
| 171 |      0.1 |    0.0 |   0.0 |    20 |   0.006 |   0.007 | elementwise_kernel<128, 4, CompareFunctor<long>> #2                                        |
| 172 |      0.1 |    0.0 |   0.0 |    68 |   0.002 |   0.001 | unrolled_elementwise_kernel<CUDAFunctor_add<long>, 4>                                      |
| 173 |      0.1 |    0.0 |   0.0 |    96 |   0.001 |   0.001 | CatArrayBatchedCopy_alignedK_contig<OpaqueType<2u>, unsigned int, 1, 128, 1, 8>            |
| 174 |      0.1 |    0.0 |   0.0 |    52 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, AbsFunctor<float>>                                        |
| 175 |      0.1 |    0.0 |   0.0 |    16 |   0.007 |   0.008 | vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned... |
| 176 |      0.1 |    0.0 |   0.0 |    88 |   0.001 |   0.001 | vectorized_elementwise_kernel<2, CUDAFunctor_add<long>>                                    |
| 177 |      0.1 |    0.0 |   0.0 |     1 |   0.109 |   0.109 | embedding_backward_feature_kernel<BFloat16, float, long>                                   |
| 178 |      0.1 |    0.0 |   0.0 |   144 |   0.001 |   0.001 | elementwise_kernel_with_index<int, arange_cuda_out::lambda(long)>                          |
| 179 |      0.1 |    0.0 |   0.0 |    56 |   0.002 |   0.002 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, ... |
| 180 |      0.1 |    0.0 |   0.0 |     4 |   0.024 |   0.024 | nvjet_sm90_tst_64x80_64x11_1x4_h_bz_TNT                                                    |
| 181 |      0.1 |    0.0 |   0.0 |    84 |   0.001 |   0.001 | vectorized_elementwise_kernel<2, launch_clamp_scalar::lambda(long)>                        |
| 182 |      0.1 |    0.0 |   0.0 |   115 |   0.001 |   0.001 | at_cuda_detail::cub::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long... |
| 183 |      0.1 |    0.0 |   0.0 |    12 |   0.008 |   0.007 | vectorized_elementwise_kernel<2, masked_fill_kernel::lambda(long, bool)>                   |
| 184 |      0.1 |    0.0 |   0.0 |    80 |   0.001 |   0.001 | vectorized_elementwise_kernel<4, AUnaryFunctor<int, int, int, MulFunctor<int>>>            |
| 185 |      0.1 |    0.0 |   0.0 |     4 |   0.022 |   0.022 | nvjet_sm90_tst_64x72_64x12_1x4_h_bz_TNT                                                    |
| 186 |      0.1 |    0.0 |   0.0 |    39 |   0.002 |   0.002 | unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory:... |
| 187 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |
| 188 |      0.1 |    0.0 |   0.0 |    12 |   0.007 |   0.007 | bitonicSortKVInPlace<2, -1, 16, 16, float, long, GTOp<float, true>, unsigned int>          |
| 189 |      0.1 |    0.0 |   0.0 |     8 |   0.010 |   0.010 | CatArrayBatchedCopy_alignedK_contig<OpaqueType<4u>, unsigned int, 4, 128, 1, 16>           |
| 190 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | CatArrayBatchedCopy_vectorized<OpaqueType<4u>, unsigned int, 3, 128, 1, 16, 4>             |
| 191 |      0.1 |    0.0 |   0.0 |     4 |   0.020 |   0.020 | nvjet_sm90_tst_64x64_64x13_1x4_h_bz_TNT                                                    |
| 192 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |
| 193 |      0.1 |    0.0 |   0.0 |    52 |   0.001 |   0.002 | vectorized_elementwise_kernel<8, BinaryFunctor<float, CompareEqFunctor<float>>>            |
| 194 |      0.1 |    0.0 |   0.0 |    32 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pageable)                                                           |
| 195 |      0.1 |    0.0 |   0.0 |    52 |   0.001 |   0.001 | vectorized_elementwise_kernel<8, AUnaryFunctor<float, float, bool, CompareEqFunctor<flo... |
| 196 |      0.1 |    0.0 |   0.0 |    48 |   0.001 |   0.002 | vectorized_elementwise_kernel<8, BinaryFunctor<bool, MulFunctor<bool>>>                    |
| 197 |      0.1 |    0.0 |   0.0 |    33 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int... |
| 198 |      0.1 |    0.0 |   0.0 |    44 |   0.002 |   0.002 | vectorized_elementwise_kernel<8, AUnaryFunctor<long, long, bool, CompareEqFunctor<long>>>  |
| 199 |      0.1 |    0.0 |   0.0 |    10 |   0.006 |   0.009 | cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, ... |
| 200 |      0.1 |    0.0 |   0.0 |    16 |   0.004 |   0.004 | at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_ke... |

- 2: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 6: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 7: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 12: `kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm90FlashAttentionDSABackwardSm90_object_at__tensor00000o641110121314_te...`
- 14: `multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<float, 4, (ADAM_MODE)1, false>, float const*, double, double, double, dou...`
- 17: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 19: `sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true, 0>, sm90::fwd::KernelTemplate<512, true, 0>::TmaParams<cute::tuple<int, int, int>, cut...`
- 28: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64div41_None_None_None_tensorptrf32gmemalign16o16384i64div41_None_tensorpt...`
- 33: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div41_None_None_None_tensorptrf32gmemalign16o512i64div41_None_tensorptrf32...`
- 34: `elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned short)>`
- 43: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16384i64div41_None_tensorptrf32gmemalign16o16384i64div41_None_tensorptrf...`
- 46: `multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, 1, 0>, multiplies<float>, float*, float>`
- 57: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas`
- 59: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_None_tensorptrbf16gmemalign16o512i64div81_None_tensorptrf32...`
- 62: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas`
- 66: `multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, float, 1, 1, 0>, float*, int>`
- 67: `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Copy<BFloat16, BFloat16>>`
- 69: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 70: `kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm90_FlashAttentionDSABackwardPreprocessSm90_object_at__tensorptrbf16gme...`
- 71: `at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy10...`
- 73: `multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>, Copy<BFloat16, float>>`
- 86: `reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsigned int, float, 4, 4>>`
- 88: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_2409601_None_None_tensorptrbf16gmemal...`
- 92: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_1409601_tensorptrbf16gmemalig...`
- 96: `kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm90_FlashAttentionDSABackwardPostprocessSm90_object_at__tensorptrf32gme...`
- 99: `vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cuda::lambda(bool, bool)>>`
- 126: `at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy1...`
- 127: `kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor_histogram<float, float, false>(Tensor, Tensor, Tensor, long, Accumula...`
- 129: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, OpaqueType<4>, long>::operator()<TensorAssign>>`
- 132: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_4102401_None_None_tensorptrbf16gmemal...`
- 133: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_451201_None_None_tensorptrbf16gmemalig...`
- 134: `reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned int, int, 4, 4>>`
- 137: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda::lambda(long, long)>>`
- 143: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda::lambda(long, long)>>`
- 146: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_1102401_tensorptrbf16gmemalig...`
- 149: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_251201_tensorptrbf16gmemalign1...`
- 151: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda::lambda(long, long)>>`
- 153: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, float, long>::operator()<ReduceAdd>>`
- 156: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lambda(char*, char const*, long)>>`
- 157: `at_cuda_detail::cub::radix_sort::DeviceRadixSortExclusiveSumKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Poli...`
- 158: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long, long, long, unsigned int, plus<long>>::Policy1000, long const*, long*...`
- 161: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lambda(char*, char const*, long)>>`
- 165: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<long, cute::C<1>, cut...`
- 170: `cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>`
- 175: `vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>`
- 179: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, int, int, unsigned int, plus<int>>::Policy1000, int const*, int*, at_c...`
- 182: `at_cuda_detail::cub::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true>>`
- 186: `unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory::StoreWithoutCast>`
- 187: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<cute::C<1>, long, cut...`
- 192: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<long, cute::C<1>, cut...`
- 195: `vectorized_elementwise_kernel<8, AUnaryFunctor<float, float, bool, CompareEqFunctor<float>>>`
- 197: `reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int, float, 4, 4>>`
- 199: `cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>`
- 200: `at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_key::policy_hub<thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::t...`

## Non-NCCL, by time per call (top 200, calls >= 1)

|  # | total ms | % busy | % win | calls | mean ms |  med ms | name                                                                                       |
|----|----------|--------|-------|-------|---------|---------|--------------------------------------------------------------------------------------------|
|  1 |     10.4 |    0.4 |   0.4 |     1 |  10.391 |  10.391 | nvjet_sm90_tst_320x128_64x3_1x2_h_badd_coopB_NTT                                           |
|  2 |      9.2 |    0.4 |   0.3 |     1 |   9.177 |   9.177 | nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT                                             |
|  3 |     17.6 |    0.7 |   0.6 |     3 |   5.883 |   7.229 | nvjet_sm90_tst_128x256_64x4_2x1_v_badd_coopA_NTN                                           |
|  4 |    171.3 |    7.1 |   6.3 |    60 |   2.855 |   2.312 | chunk_cat_cuda_kernel<float, BFloat16>                                                     |
|  5 |     36.7 |    1.5 |   1.3 |    24 |   1.531 |   1.168 | kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm... |
|  6 |     61.0 |    2.5 |   2.2 |    48 |   1.272 |   1.068 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  7 |    117.1 |    4.9 |   4.3 |    96 |   1.220 |   1.048 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  8 |     58.2 |    2.4 |   2.1 |    48 |   1.212 |   1.026 | cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaG... |
|  9 |     35.3 |    1.5 |   1.3 |    35 |   1.008 |   1.020 | multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<flo... |
| 10 |     16.1 |    0.7 |   0.6 |    17 |   0.948 |   0.587 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_TNT                                             |
| 11 |     27.2 |    1.1 |   1.0 |    30 |   0.907 |   0.251 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT                                             |
| 12 |     11.4 |    0.5 |   0.4 |    14 |   0.811 |   0.846 | vectorized_elementwise_kernel<4, exp_kernel_cuda::lambda(float)>                           |
| 13 |     30.4 |    1.3 |   1.1 |    48 |   0.633 |   0.664 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>:... |
| 14 |      3.7 |    0.2 |   0.1 |     6 |   0.617 |   0.625 | nvjet_sm90_tst_128x304_64x3_2x1_v_bz_coopA_NNT                                             |
| 15 |      3.5 |    0.1 |   0.1 |     6 |   0.576 |   0.585 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_NNT                                             |
| 16 |      3.4 |    0.1 |   0.1 |     6 |   0.572 |   0.586 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_NNT                                             |
| 17 |     15.7 |    0.7 |   0.6 |    28 |   0.561 |   0.586 | nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT                                             |
| 18 |     26.5 |    1.1 |   1.0 |    48 |   0.551 |   0.505 | sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true, 0>, sm90::fwd::K... |
| 19 |      3.3 |    0.1 |   0.1 |     6 |   0.544 |   0.549 | nvjet_sm90_tst_256x160_64x4_2x1_v_bz_coopA_NNT                                             |
| 20 |      9.8 |    0.4 |   0.4 |    18 |   0.542 |   0.705 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_NNT                                             |
| 21 |      3.2 |    0.1 |   0.1 |     6 |   0.525 |   0.538 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_NNN                                             |
| 22 |      3.1 |    0.1 |   0.1 |     6 |   0.514 |   0.521 | nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_NNT                                             |
| 23 |     36.3 |    1.5 |   1.3 |    85 |   0.427 |   0.754 | vectorized_elementwise_kernel<4, BUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 24 |     13.8 |    0.6 |   0.5 |    36 |   0.383 |   0.502 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_TNT                                             |
| 25 |     10.4 |    0.4 |   0.4 |    35 |   0.297 |   0.300 | multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, ... |
| 26 |     14.1 |    0.6 |   0.5 |    48 |   0.294 |   0.333 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div4... |
| 27 |      7.1 |    0.3 |   0.3 |    24 |   0.294 |   0.330 | preprocess_kernel_kernel                                                                   |
| 28 |      7.0 |    0.3 |   0.3 |    24 |   0.292 |   0.328 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 29 |      2.1 |    0.1 |   0.1 |     8 |   0.265 |   0.297 | mbtopk::gatherTopK<float, unsigned int, 2>                                                 |
| 30 |     11.9 |    0.5 |   0.4 |    45 |   0.265 |   0.082 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_NNT                                             |
| 31 |     25.3 |    1.1 |   0.9 |    96 |   0.264 |   0.128 | indexing_backward_kernel<BFloat16, 4>                                                      |
| 32 |      2.1 |    0.1 |   0.1 |     8 |   0.263 |   0.289 | _triton_fp8_indexer_kernel                                                                 |
| 33 |     12.2 |    0.5 |   0.4 |    48 |   0.253 |   0.149 | nvjet_sm90_tst_256x144_64x4_2x1_v_bz_coopA_TNT                                             |
| 34 |     46.5 |    1.9 |   1.7 |   184 |   0.253 |   0.148 | nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN                                             |
| 35 |      9.7 |    0.4 |   0.4 |    39 |   0.250 |   0.072 | nvjet_sm90_tst_256x152_64x4_2x1_v_bz_coopA_NNT                                             |
| 36 |     23.9 |    1.0 |   0.9 |   102 |   0.234 |   0.054 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT                                             |
| 37 |     15.6 |    0.6 |   0.6 |    68 |   0.229 |   0.169 | split_with_sizes_copy_out_contiguous_no_cast_kernel                                        |
| 38 |     11.1 |    0.5 |   0.4 |    52 |   0.214 |   0.247 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16... |
| 39 |      8.3 |    0.3 |   0.3 |    40 |   0.207 |   0.088 | elementwise_kernel<128, 4, BinaryFunctor<BFloat16, MulFunctor<float>>>                     |
| 40 |     18.1 |    0.8 |   0.7 |    90 |   0.201 |   0.003 | elementwise_kernel<128, 2, CUDAFunctor_add<float>>                                         |
| 41 |     31.6 |    1.3 |   1.2 |   164 |   0.193 |   0.169 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT                                             |
| 42 |     18.3 |    0.8 |   0.7 |    96 |   0.191 |   0.201 | elementwise_kernel<128, 4, launch_clamp_scalar::lambda(BFloat16)>                          |
| 43 |      4.5 |    0.2 |   0.2 |    24 |   0.186 |   0.207 | kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm... |
| 44 |     17.6 |    0.7 |   0.6 |   100 |   0.176 |   0.171 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64di... |
| 45 |      8.4 |    0.4 |   0.3 |    48 |   0.176 |   0.197 | tma_scatter_add_kernel<BFloat16, long>                                                     |
| 46 |     25.8 |    1.1 |   0.9 |   148 |   0.174 |   0.141 | elementwise_kernel<128, 4, direct_copy_kernel_cuda::lambda(BFloat16)>                      |
| 47 |      0.5 |    0.0 |   0.0 |     3 |   0.173 |   0.169 | compute_grad_weight<BFloat16, long>                                                        |
| 48 |     82.6 |    3.4 |   3.0 |   481 |   0.172 |   0.008 | Memcpy DtoD (Device -> Device)                                                             |
| 49 |     12.1 |    0.5 |   0.4 |    72 |   0.168 |   0.176 | elementwise_kernel<128, 4, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>              |
| 50 |     46.3 |    1.9 |   1.7 |   276 |   0.168 |   0.003 | elementwise_kernel<128, 2, BinaryFunctor<float, MulFunctor<float>>>                        |
| 51 |      1.0 |    0.0 |   0.0 |     6 |   0.165 |   0.166 | nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_splitK_NNT                                      |
| 52 |      7.3 |    0.3 |   0.3 |    48 |   0.152 |   0.157 | cunn_SpatialSoftMaxForward<float, float, float, int, SoftMaxForwardEpilogue>               |
| 53 |      4.8 |    0.2 |   0.2 |    32 |   0.149 |   0.171 | mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>                      |
| 54 |      2.3 |    0.1 |   0.1 |    16 |   0.146 |   0.140 | cunn_SpatialSoftMaxBackward<float, float, float, SoftMaxBackwardEpilogue>                  |
| 55 |     14.0 |    0.6 |   0.5 |    96 |   0.146 |   0.069 | elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned sho... |
| 56 |      7.2 |    0.3 |   0.3 |    50 |   0.143 |   0.133 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 57 |      1.1 |    0.0 |   0.0 |     8 |   0.140 |   0.158 | radixSortKVInPlace<2, -1, 32, 32, float, long, unsigned int>                               |
| 58 |      6.7 |    0.3 |   0.2 |    48 |   0.139 |   0.144 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 1, 128, 1, 16, 8>             |
| 59 |      4.9 |    0.2 |   0.2 |    35 |   0.139 |   0.141 | multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, floa... |
| 60 |      3.3 |    0.1 |   0.1 |    24 |   0.138 |   0.146 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 2, 128, 1, 16, 8>             |
| 61 |      3.2 |    0.1 |   0.1 |    24 |   0.133 |   0.133 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_TNT                                             |
| 62 |     78.5 |    3.3 |   2.9 |   591 |   0.133 |   0.004 | elementwise_kernel<128, 4, BinaryFunctor<float, MulFunctor<float>>>                        |
| 63 |      0.8 |    0.0 |   0.0 |     6 |   0.131 |   0.132 | nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_NNT                                             |
| 64 |      0.4 |    0.0 |   0.0 |     3 |   0.130 |   0.128 | sum_and_scatter<BFloat16, long>                                                            |
| 65 |      5.0 |    0.2 |   0.2 |    40 |   0.125 |   0.117 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_32x3_nt_align4>                      |
| 66 |      6.5 |    0.3 |   0.2 |    52 |   0.124 |   0.142 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>                      |
| 67 |      7.8 |    0.3 |   0.3 |    64 |   0.122 |   0.069 | CatArrayBatchedCopy<OpaqueType<2u>, unsigned int, 4, 64, 64>                               |
| 68 |     11.2 |    0.5 |   0.4 |    92 |   0.121 |   0.002 | vectorized_elementwise_kernel<4, BinaryFunctor<float, MulFunctor<float>>>                  |
| 69 |      5.9 |    0.2 |   0.2 |    50 |   0.118 |   0.105 | sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segm... |
| 70 |      1.9 |    0.1 |   0.1 |    16 |   0.117 |   0.045 | _per_token_group_quant_fp8                                                                 |
| 71 |      5.6 |    0.2 |   0.2 |    48 |   0.116 |   0.131 | _triton_hpb_bwd_g_hp_hr_kernel                                                             |
| 72 |     34.1 |    1.4 |   1.2 |   296 |   0.115 |   0.062 | _triton_rope_kernel                                                                        |
| 73 |      5.4 |    0.2 |   0.2 |    48 |   0.112 |   0.127 | _triton_hpb_bwd_g_x_orig_kernel                                                            |
| 74 |      0.9 |    0.0 |   0.0 |     8 |   0.111 |   0.126 | elementwise_kernel<128, 4, float8_copy_kernel_cuda::lambda(Float8_e4m3fn)>                 |
| 75 |      0.1 |    0.0 |   0.0 |     1 |   0.109 |   0.109 | embedding_backward_feature_kernel<BFloat16, float, long>                                   |
| 76 |     10.3 |    0.4 |   0.4 |    96 |   0.107 |   0.121 | _triton_hpb_fwd_kernel                                                                     |
| 77 |     44.3 |    1.8 |   1.6 |   490 |   0.090 |   0.004 | vectorized_elementwise_kernel<4, CUDAFunctor_add<float>>                                   |
| 78 |      1.1 |    0.0 |   0.0 |    12 |   0.090 |   0.101 | sbtopk::gatherTopK<float, unsigned int, 2, false>                                          |
| 79 |     23.7 |    1.0 |   0.9 |   268 |   0.089 |   0.055 | vectorized_gather_kernel<16, long>                                                         |
| 80 |      2.0 |    0.1 |   0.1 |    24 |   0.082 |   0.091 | kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm... |
| 81 |     88.7 |    3.7 |   3.2 |  1103 |   0.080 |   0.002 | vectorized_elementwise_kernel<8, bfloat16_copy_kernel_cuda::lambda(float)>                 |
| 82 |     48.8 |    2.0 |   1.8 |   616 |   0.079 |   0.004 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(float), 4>                     |
| 83 |      3.1 |    0.1 |   0.1 |    40 |   0.079 |   0.074 | nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_TNT                                             |
| 84 |      4.8 |    0.2 |   0.2 |    64 |   0.075 |   0.043 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lam... |
| 85 |      2.9 |    0.1 |   0.1 |    39 |   0.073 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsi... |
| 86 |     10.7 |    0.4 |   0.4 |   148 |   0.073 |   0.005 | Memcpy HtoD (Pinned -> Device)                                                             |
| 87 |      4.8 |    0.2 |   0.2 |    68 |   0.071 |   0.089 | multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Cop... |
| 88 |      3.4 |    0.1 |   0.1 |    48 |   0.071 |   0.055 | elementwise_kernel<128, 4, CUDAFunctor_add<BFloat16>>                                      |
| 89 |     20.0 |    0.8 |   0.7 |   292 |   0.069 |   0.044 | vectorized_elementwise_kernel<8, CUDAFunctor_add<BFloat16>>                                |
| 90 |      2.6 |    0.1 |   0.1 |    40 |   0.064 |   0.066 | nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN                                             |
| 91 |      2.9 |    0.1 |   0.1 |    48 |   0.061 |   0.030 | vectorized_elementwise_kernel<8, silu_backward_kernel::lambda(BFloat16, BFloat16)>         |
| 92 |      3.9 |    0.2 |   0.1 |    68 |   0.057 |   0.040 | multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>,... |
| 93 |     12.8 |    0.5 |   0.5 |   240 |   0.053 |   0.027 | vectorized_elementwise_kernel<8, BinaryFunctor<BFloat16, MulFunctor<float>>>               |
| 94 |      3.6 |    0.1 |   0.1 |    80 |   0.045 |   0.019 | elementwise_kernel<128, 4, masked_fill_kernel::lambda(BFloat16, bool)>                     |
| 95 |     16.8 |    0.7 |   0.6 |   388 |   0.043 |   0.008 | reduce_kernel<128, 4, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 96 |      4.2 |    0.2 |   0.2 |    96 |   0.043 |   0.021 | vectorized_elementwise_kernel<8, silu_kernel::lambda(BFloat16)>                            |
| 97 |      2.2 |    0.1 |   0.1 |    52 |   0.042 |   0.048 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4... |
| 98 |      0.5 |    0.0 |   0.0 |    12 |   0.039 |   0.039 | cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_nt_align4>                      |
| 99 |      0.7 |    0.0 |   0.0 |    20 |   0.036 |   0.036 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_NTT                                                   |
| 100 |      1.6 |    0.1 |   0.1 |    48 |   0.034 |   0.016 | vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cud... |
| 101 |      2.2 |    0.1 |   0.1 |    68 |   0.033 |   0.024 | reduce_kernel<128, 4, ReduceOp<BFloat16, sum_functor, unsigned int, BFloat16, 4, 8>>       |
| 102 |     13.0 |    0.5 |   0.5 |   427 |   0.030 |   0.005 | reduce_kernel<512, 1, ReduceOp<float, sum_functor, unsigned int, float, 4, 4>>             |
| 103 |      1.2 |    0.0 |   0.0 |    40 |   0.030 |   0.027 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                                                   |
| 104 |      0.6 |    0.0 |   0.0 |    20 |   0.029 |   0.029 | nvjet_sm90_tst_128x128_64x6_1x2_h_bz_NTT                                                   |
| 105 |      9.6 |    0.4 |   0.4 |   356 |   0.027 |   0.011 | vectorized_elementwise_kernel<8, FillFunctor<BFloat16>>                                    |
| 106 |      0.5 |    0.0 |   0.0 |    20 |   0.026 |   0.026 | nvjet_sm90_tst_128x120_64x6_2x4_h_bz_TNT                                                   |
| 107 |      2.4 |    0.1 |   0.1 |    96 |   0.025 |   0.014 | CatArrayBatchedCopy_vectorized<OpaqueType<2u>, unsigned int, 3, 128, 1, 16, 8>             |
| 108 |      2.5 |    0.1 |   0.1 |   100 |   0.025 |   0.028 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64di... |
| 109 |      0.5 |    0.0 |   0.0 |    20 |   0.025 |   0.025 | nvjet_sm90_tst_64x200_64x6_2x1_v_bz_TNT                                                    |
| 110 |      0.1 |    0.0 |   0.0 |     4 |   0.024 |   0.024 | nvjet_sm90_tst_64x80_64x11_1x4_h_bz_TNT                                                    |
| 111 |      0.5 |    0.0 |   0.0 |    20 |   0.024 |   0.028 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                                                    |
| 112 |      0.6 |    0.0 |   0.0 |    24 |   0.023 |   0.026 | nvjet_sm90_tst_64x128_64x8_2x1_v_bz_NTN                                                    |
| 113 |      0.1 |    0.0 |   0.0 |     4 |   0.022 |   0.022 | nvjet_sm90_tst_64x72_64x12_1x4_h_bz_TNT                                                    |
| 114 |     10.3 |    0.4 |   0.4 |   480 |   0.021 |   0.001 | vectorized_elementwise_kernel<4, AUnaryFunctor<float, float, float, MulFunctor<float>>>    |
| 115 |      0.1 |    0.0 |   0.0 |     4 |   0.020 |   0.020 | nvjet_sm90_tst_64x64_64x13_1x4_h_bz_TNT                                                    |
| 116 |      0.4 |    0.0 |   0.0 |    24 |   0.017 |   0.016 | radixSortKVInPlace<-2, -1, 32, 32, float, long, unsigned int>                              |
| 117 |      1.7 |    0.1 |   0.1 |    96 |   0.017 |   0.009 | _triton_sinkhorn_fwd_kernel                                                                |
| 118 |      0.8 |    0.0 |   0.0 |    48 |   0.016 |   0.018 | _triton_sinkhorn_bwd_kernel                                                                |
| 119 |      0.2 |    0.0 |   0.0 |    16 |   0.014 |   0.015 | elementwise_kernel<128, 4, CompareFunctor<long>>                                           |
| 120 |      1.2 |    0.0 |   0.0 |    96 |   0.012 |   0.014 | vectorized_elementwise_kernel<8, launch_clamp_scalar::lambda(BFloat16)>                    |
| 121 |      0.4 |    0.0 |   0.0 |    32 |   0.012 |   0.013 | mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>                                 |
| 122 |      0.2 |    0.0 |   0.0 |    20 |   0.011 |   0.011 | nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                                                    |
| 123 |      0.3 |    0.0 |   0.0 |    24 |   0.011 |   0.012 | indexing_backward_kernel_stride_1<float>                                                   |
| 124 |      0.5 |    0.0 |   0.0 |    52 |   0.011 |   0.009 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned long, unsigned long)>  |
| 125 |      0.1 |    0.0 |   0.0 |     8 |   0.010 |   0.010 | CatArrayBatchedCopy_alignedK_contig<OpaqueType<4u>, unsigned int, 4, 128, 1, 16>           |
| 126 |      0.2 |    0.0 |   0.0 |    24 |   0.010 |   0.011 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1... |
| 127 |      0.7 |    0.0 |   0.0 |    72 |   0.010 |   0.011 | vectorized_elementwise_kernel<8, compare_scalar_kernel<BFloat16>::lambda(BFloat16)>        |
| 128 |      0.0 |    0.0 |   0.0 |     4 |   0.009 |   0.009 | nvjet_sm90_tst_32x64_64x16_2x1_v_bz_splitK_TNN                                             |
| 129 |      0.6 |    0.0 |   0.0 |    72 |   0.008 |   0.007 | elementwise_kernel<128, 2, CUDAFunctor_add<int>>                                           |
| 130 |      0.1 |    0.0 |   0.0 |    12 |   0.008 |   0.007 | vectorized_elementwise_kernel<2, masked_fill_kernel::lambda(long, bool)>                   |
| 131 |      0.4 |    0.0 |   0.0 |    48 |   0.008 |   0.008 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64di... |
| 132 |      0.2 |    0.0 |   0.0 |    24 |   0.008 |   0.009 | CatArrayBatchedCopy_vectorized<OpaqueType<4u>, unsigned int, 4, 128, 1, 16, 4>             |
| 133 |      0.2 |    0.0 |   0.0 |    32 |   0.007 |   0.008 | mbtopk::computeDigitCumSum                                                                 |
| 134 |      0.1 |    0.0 |   0.0 |    18 |   0.007 |   0.009 | elementwise_kernel<128, 2, direct_copy_kernel_cuda>                                        |
| 135 |      0.1 |    0.0 |   0.0 |    16 |   0.007 |   0.008 | vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned... |
| 136 |      0.4 |    0.0 |   0.0 |    55 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, ... |
| 137 |      0.2 |    0.0 |   0.0 |    31 |   0.007 |   0.007 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, f... |
| 138 |      1.4 |    0.1 |   0.1 |   196 |   0.007 |   0.004 | elementwise_kernel<128, 2, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>    |
| 139 |      0.1 |    0.0 |   0.0 |    12 |   0.007 |   0.007 | bitonicSortKVInPlace<2, -1, 16, 16, float, long, GTOp<float, true>, unsigned int>          |
| 140 |      4.3 |    0.2 |   0.2 |   633 |   0.007 |   0.006 | at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::rad... |
| 141 |      0.2 |    0.0 |   0.0 |    24 |   0.007 |   0.007 | vectorized_elementwise_kernel<4, softplus_backward_kernel::lambda(float, float)>           |
| 142 |      0.0 |    0.0 |   0.0 |     3 |   0.007 |   0.007 | at_cuda_detail::cub::unique_by_key::DeviceUniqueByKeySweepKernel<at_cuda_detail::cub::u... |
| 143 |      0.1 |    0.0 |   0.0 |    20 |   0.006 |   0.007 | elementwise_kernel<128, 4, CompareFunctor<long>> #2                                        |
| 144 |      0.1 |    0.0 |   0.0 |    10 |   0.006 |   0.009 | cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, ... |
| 145 |      0.3 |    0.0 |   0.0 |    48 |   0.006 |   0.006 | vectorized_elementwise_kernel<8, compare_scalar_kernel<long>::lambda(long)>                |
| 146 |      0.2 |    0.0 |   0.0 |    40 |   0.006 |   0.006 | kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o5... |
| 147 |      0.0 |    0.0 |   0.0 |     2 |   0.005 |   0.005 | lpnorm_cleanup<float, (NormType)2, float, true, float>                                     |
| 148 |      0.0 |    0.0 |   0.0 |     8 |   0.005 |   0.005 | unrolled_elementwise_kernel<BinaryFunctor<float, MulFunctor<float>>, 4>                    |
| 149 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | vectorized_elementwise_kernel<4, softplus_kernel::lambda(float)>                           |
| 150 |      0.1 |    0.0 |   0.0 |    28 |   0.005 |   0.004 | elementwise_kernel<128, 2, CUDAFunctor_add<long>>                                          |
| 151 |      0.2 |    0.0 |   0.0 |    48 |   0.005 |   0.005 | _fill_indices_kernel                                                                       |
| 152 |      0.4 |    0.0 |   0.0 |    96 |   0.005 |   0.004 | kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor... |
| 153 |      0.0 |    0.0 |   0.0 |     3 |   0.005 |   0.004 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long,... |
| 154 |      1.2 |    0.1 |   0.0 |   288 |   0.004 |   0.003 | elementwise_kernel<128, 4, CUDAFunctor_add<float>>                                         |
| 155 |      0.3 |    0.0 |   0.0 |    80 |   0.004 |   0.004 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div... |
| 156 |      0.2 |    0.0 |   0.0 |    52 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lam... |
| 157 |      0.1 |    0.0 |   0.0 |    16 |   0.004 |   0.004 | at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_ke... |
| 158 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | reduce_kernel<128, 4, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 159 |      0.3 |    0.0 |   0.0 |    88 |   0.004 |   0.004 | reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned i... |
| 160 |      0.2 |    0.0 |   0.0 |    48 |   0.004 |   0.004 | index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lam... |
| 161 |      0.0 |    0.0 |   0.0 |     3 |   0.004 |   0.004 | krn_partials_per_segment<long>                                                             |
| 162 |      0.3 |    0.0 |   0.0 |    88 |   0.003 |   0.004 | vectorized_elementwise_kernel<8, compare_scalar_kernel<int>>                               |
| 163 |      0.1 |    0.0 |   0.0 |    40 |   0.003 |   0.003 | cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, fl... |
| 164 |      0.0 |    0.0 |   0.0 |     8 |   0.003 |   0.003 | _scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, ... |
| 165 |      0.0 |    0.0 |   0.0 |     8 |   0.003 |   0.003 | vectorized_elementwise_kernel<8, BinaryFunctor<bool, BitwiseOrFunctor<bool>>>              |
| 166 |      0.0 |    0.0 |   0.0 |     1 |   0.003 |   0.003 | reduce_kernel<512, 1, ReduceOp<float, WelfordOps<float, float, int, cuda::__4::pair<flo... |
| 167 |      0.0 |    0.0 |   0.0 |     6 |   0.003 |   0.003 | reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, or_kernel_cuda::lambda(bool, ... |
| 168 |      0.1 |    0.0 |   0.0 |    48 |   0.003 |   0.003 | elementwise_kernel<128, 2, logaddexp_kernel_cuda::lambda(float, float)>                    |
| 169 |      0.0 |    0.0 |   0.0 |     3 |   0.003 |   0.002 | at_cuda_detail::cub::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<u... |
| 170 |      0.9 |    0.0 |   0.0 |   328 |   0.003 |   0.002 | unrolled_elementwise_kernel<direct_copy_kernel_cuda::lambda(long), 4>                      |
| 171 |      0.0 |    0.0 |   0.0 |     3 |   0.003 |   0.003 | krn_partial_segment_offset<long>                                                           |
| 172 |      0.2 |    0.0 |   0.0 |    58 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, BinaryFunctor<float, DivFunctor<float>>>                  |
| 173 |      0.5 |    0.0 |   0.0 |   180 |   0.003 |   0.001 | unrolled_elementwise_kernel<direct_copy_kernel_cuda, 4>                                    |
| 174 |      0.1 |    0.0 |   0.0 |    50 |   0.003 |   0.003 | vectorized_elementwise_kernel<4, sqrt_kernel_cuda::lambda(float)>                          |
| 175 |      0.9 |    0.0 |   0.0 |   330 |   0.003 |   0.001 | vectorized_elementwise_kernel<4, FillFunctor<float>>                                       |
| 176 |      0.5 |    0.0 |   0.0 |   171 |   0.003 |   0.002 | at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::ra... |
| 177 |      0.3 |    0.0 |   0.0 |   120 |   0.003 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda:... |
| 178 |      0.0 |    0.0 |   0.0 |    16 |   0.002 |   0.003 | kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o128i64div... |
| 179 |      0.1 |    0.0 |   0.0 |    32 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pageable)                                                           |
| 180 |      0.3 |    0.0 |   0.0 |   116 |   0.002 |   0.002 | Memcpy DtoH (Device -> Pinned)                                                             |
| 181 |      0.5 |    0.0 |   0.0 |   196 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, sigmoid_kernel_cuda::lambda(float)>                       |
| 182 |      0.5 |    0.0 |   0.0 |   200 |   0.002 |   0.001 | vectorized_elementwise_kernel<2, CUDAFunctorOnSelf_add<long>>                              |
| 183 |      0.1 |    0.0 |   0.0 |    52 |   0.002 |   0.002 | vectorized_elementwise_kernel<4, AbsFunctor<float>>                                        |
| 184 |      0.0 |    0.0 |   0.0 |     8 |   0.002 |   0.002 | compute_cuda_kernel<long>                                                                  |
| 185 |      0.0 |    0.0 |   0.0 |     2 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, and_kernel_cuda::lambda(bool,... |
| 186 |      0.1 |    0.0 |   0.0 |    39 |   0.002 |   0.002 | unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory:... |
| 187 |      0.3 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<long, sum_functor, unsigned int, long, 4, 4>>               |
| 188 |      0.0 |    0.0 |   0.0 |     3 |   0.002 |   0.001 | compute_num_of_partial_segments<long>                                                      |
| 189 |      0.0 |    0.0 |   0.0 |     1 |   0.002 |   0.002 | elementwise_kernel<128, 4, CompareFunctor<float>>                                          |
| 190 |      0.0 |    0.0 |   0.0 |     1 |   0.002 |   0.002 | multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarFunctor<float, 1, 1, 0>,... |
| 191 |      0.0 |    0.0 |   0.0 |    24 |   0.002 |   0.002 | unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4>                                     |
| 192 |      0.1 |    0.0 |   0.0 |    33 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int... |
| 193 |      0.3 |    0.0 |   0.0 |   136 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda:... |
| 194 |      0.0 |    0.0 |   0.0 |     8 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<long, func_wrapper_t<long, MaxNanFunctor<long>>, unsigne... |
| 195 |      0.1 |    0.0 |   0.0 |    56 |   0.002 |   0.002 | at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, ... |
| 196 |      0.2 |    0.0 |   0.0 |   120 |   0.002 |   0.002 | vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda:... |
| 197 |      0.1 |    0.0 |   0.0 |    68 |   0.002 |   0.001 | unrolled_elementwise_kernel<CUDAFunctor_add<long>, 4>                                      |
| 198 |      0.3 |    0.0 |   0.0 |   168 |   0.002 |   0.002 | elementwise_kernel<128, 2, BinaryFunctor<float, DivFunctor<float>>>                        |
| 199 |      0.0 |    0.0 |   0.0 |     3 |   0.002 |   0.002 | reduce_kernel<512, 1, ReduceOp<float, NormTwoOps<float, float, float, true>, unsigned i... |
| 200 |      0.1 |    0.0 |   0.0 |    48 |   0.002 |   0.002 | prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t... |

- 5: `kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm90FlashAttentionDSABackwardSm90_object_at__tensor00000o641110121314_te...`
- 6: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 7: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 8: `cutlass enable_3x_kernel_for_sm9x GemmUniversal GroupProblemShape MainloopSm90ArrayTmaGmmaWarpSpecialized KernelPtrArrayTmaWarpSpecializedCooperative bfloat1...`
- 9: `multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<float, 4, (ADAM_MODE)1, false>, float const*, double, double, double, dou...`
- 13: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_put_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 18: `sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true, 0>, sm90::fwd::KernelTemplate<512, true, 0>::TmaParams<cute::tuple<int, int, int>, cut...`
- 25: `multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarTensorFunctor<float, 1, 1, 0>, multiplies<float>, float*, float>`
- 26: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o512i64div41_None_None_None_tensorptrf32gmemalign16o512i64div41_None_tensorptrf32...`
- 28: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_None_tensorptrbf16gmemalign16o512i64div81_None_tensorptrf32...`
- 38: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign8o16384i64div41_None_tensorptrf32gmemalign16o16384i64div41_None_tensorptrf...`
- 43: `kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm90_FlashAttentionDSABackwardPreprocessSm90_object_at__tensorptrbf16gme...`
- 44: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign8o16384i64div41_None_None_None_tensorptrf32gmemalign16o16384i64div41_None_tensorpt...`
- 55: `elementwise_kernel<128, 4, where_kernel_impl::lambda(bool, unsigned short, unsigned short)>`
- 56: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas`
- 59: `multi_tensor_apply_kernel<TensorListMetadata<1>, LpNormFunctor<float, (NormType)2, float, 1, 1, 0>, float*, int>`
- 69: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x32_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas`
- 80: `kernel_cutlass_kernel_cudnndeepseek_sparse_attentionsparse_attention_backwarddsa_bwd_sm90_FlashAttentionDSABackwardPostprocessSm90_object_at__tensorptrf32gme...`
- 84: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<2>>::lambda(char*, char const*, long)>>`
- 85: `reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float>>, unsigned int, float, 4, 4>>`
- 87: `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<BFloat16, 2, 1, 1>, Copy<BFloat16, BFloat16>>`
- 92: `multi_tensor_apply_kernel<TensorListMetadata<2>, CopyFunctor<BFloat16, float, 2, 1, 1>, Copy<BFloat16, float>>`
- 97: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_1409601_tensorptrbf16gmemalig...`
- 100: `vectorized_elementwise_kernel<8, BinaryFunctor<bool, bool, bool, logical_and_kernel_cuda::lambda(bool, bool)>>`
- 108: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o4096i64div81_tensorptrbf16_gmem_align16_o_2409601_None_None_tensorptrbf16gmemal...`
- 126: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_1102401_tensorptrbf16gmemalig...`
- 131: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16_gmem_align16_o_4102401_None_None_tensorptrbf16gmemal...`
- 135: `vectorized_elementwise_kernel<4, where_kernel_impl::lambda(bool, unsigned int, unsigned int)>`
- 136: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, OpaqueType<4>, long>::operator()<TensorAssign>>`
- 137: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<true, float, long>::operator()<ReduceAdd>>`
- 140: `at_cuda_detail::cub::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy10...`
- 142: `at_cuda_detail::cub::unique_by_key::DeviceUniqueByKeySweepKernel<at_cuda_detail::cub::unique_by_key::policy_hub<long, long>::Policy1000, long const*, thrust:...`
- 144: `cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>`
- 146: `kernel_cutlass_kernel_quackrmsnormRMSNormBackward_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_251201_tensorptrbf16gmemalign1...`
- 152: `kernelHistogram1D<float, float, long, 1, 2, -1, (CUDAHistogramMemoryType)0, CUDA_tensor_histogram<float, float, false>(Tensor, Tensor, Tensor, long, Accumula...`
- 153: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<long, long, long, unsigned int, cub::SumOp<long>>::Policy1000, long const*,...`
- 155: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o512i64div81_tensorptrbf16_gmem_align16_o_451201_None_None_tensorptrbf16gmemalig...`
- 156: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<4>>::lambda(char*, char const*, long)>>`
- 157: `at_cuda_detail::cub::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::scan_by_key::policy_hub<thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::t...`
- 159: `reduce_kernel<512, 1, ReduceOp<int, func_wrapper_t<int, MaxNanFunctor<int>>, unsigned int, int, 4, 4>>`
- 160: `index_elementwise_kernel<128, 4, gpu_index_kernel<index_kernel_impl<OpaqueType<8>>::lambda(char*, char const*, long)>>`
- 163: `cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>`
- 164: `_scatter_gather_elementwise_kernel<128, 8, _cuda_scatter_gather_internal_kernel<false, OpaqueType<8>, long>::operator()<TensorAssign>>`
- 166: `reduce_kernel<512, 1, ReduceOp<float, WelfordOps<float, float, int, cuda::__4::pair<float, float>>, unsigned int, float, 2, 2>>`
- 167: `reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, or_kernel_cuda::lambda(bool, bool)>, unsigned int, bool, 4, 4>>`
- 169: `at_cuda_detail::cub::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<unsigned long long, true>, long*>`
- 176: `at_cuda_detail::cub::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::radix::policy_hub<long, cub::OpaqueType<8>, unsigned long long>::Policy1...`
- 177: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, remainder_kernel_cuda::lambda(long, long)>>`
- 178: `kernel_cutlass_kernel_quackrmsnormRMSNorm_object_at__tensorptrbf16gmemalign16o128i64div81_tensorptrbf16_gmem_align16_o_812801_None_None_tensorptrbf16gmemalig...`
- 185: `reduce_kernel<512, 1, ReduceOp<bool, func_wrapper_t<bool, and_kernel_cuda::lambda(bool, bool)>, unsigned int, bool, 4, 4>>`
- 186: `unrolled_elementwise_kernel<CUDAFunctor_add<float>, 4, memory::LoadWithoutCast, memory::StoreWithoutCast>`
- 190: `multi_tensor_apply_kernel<TensorListMetadata<1>, BinaryOpScalarFunctor<float, 1, 1, 0>, plus<float>, float>`
- 192: `reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int, float, 4, 4>>`
- 193: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_floor_kernel_cuda::lambda(long, long)>>`
- 194: `reduce_kernel<512, 1, ReduceOp<long, func_wrapper_t<long, MaxNanFunctor<long>>, unsigned int, long, 4, 4>>`
- 195: `at_cuda_detail::cub::scan::DeviceScanKernel<at_cuda_detail::cub::scan::policy_hub<int, int, int, unsigned int, plus<int>>::Policy1000, int const*, int*, at_c...`
- 196: `vectorized_elementwise_kernel<2, BUnaryFunctor<long, long, long, div_trunc_kernel_cuda::lambda(long, long)>>`
- 199: `reduce_kernel<512, 1, ReduceOp<float, NormTwoOps<float, float, float, true>, unsigned int, float, 4, 4>>`
- 200: `prepare_grouped_gemm_data<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cute::tuple<int, int, int>, cute::tuple<cute::C<1>, long, cut...`

## NCCL (durations include waiting for peers)

|  # | total ms | % busy | % win | calls | mean ms |  med ms | name                                                                                       |
|----|----------|--------|-------|-------|---------|---------|--------------------------------------------------------------------------------------------|
|  1 |    414.0 |   17.2 |  15.1 |   192 |   2.156 |   1.107 | ncclDevKernel_SendRecv                                                                     |
|  2 |    198.5 |    8.2 |   7.3 |    36 |   5.513 |   2.587 | ncclDevKernel_ReduceScatter_Sum_f32_RING_LL                                                |
|  3 |    177.7 |    7.4 |   6.5 |   164 |   1.083 |   0.492 | ncclDevKernel_AllGather_RING_LL                                                            |
|  4 |     54.5 |    2.3 |   2.0 |    40 |   1.362 |   0.242 | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL                                               |
|  5 |      0.9 |    0.0 |   0.0 |    16 |   0.059 |   0.026 | ncclDevKernel_AllReduce_Sum_f32_RING_LL                                                    |
|  6 |      0.0 |    0.0 |   0.0 |     2 |   0.009 |   0.009 | ncclDevKernel_AllReduce_Sum_u64_RING_LL                                                    |

## By category (first matching `--category` rule on the raw kernel name)

| category | total ms | union ms | % busy | % win | calls |
|---|---|---|---|---|---|
| sparse_attn | 76.7 | 76.7 | 3.2 | 2.8 | 144 |
| other | 2582.1 | 2330.7 | 107.3 | 94.4 | 21048 |

union ms = time covered by at least one kernel of the category (overlap across streams counted once).

Legend: total = summed durations in the window; % busy = total / GPU busy time and % win = total / window (shares
can sum past 100 when streams overlap); calls = launches; mean/med = per launch. Names drop namespaces, lambdas,
and argument lists (mangled names list their first identifiers); `#n` separates distinct kernels that shorten
alike; truncated labels are listed under each table by rank; raw names are in the CSV. Site = `[<nearest non-aten
enclosing op> >] <innermost cpu_op> <first 3 Input Dims>`; a site can launch several kernels, so its mean/median
mix them.
