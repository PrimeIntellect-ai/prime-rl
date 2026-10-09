# Kernel time diff (compute): main -> branch

- summed kernel time: 1973.0 ms -> 1908.2 ms (-64.8 ms)
- before: `/home/garrett/tmp/profiling/fp8-cleanup-h200/derived/before_r0_kernels.csv`
- after: `/home/garrett/tmp/profiling/fp8-cleanup-h200/derived/after_r0_kernels.csv`

| # | main ms | branch ms | delta ms | calls before | calls after | kernel |
|---|---|---|---|---|---|---|
| 1 | 60.1 | 27.2 | -33.0 | 49 | 37 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` |
| 2 | 26.5 | 0.0 | -26.5 | 36 | 0 | `_unpack_grouped_rows_kernel` |
| 3 | 0.0 | 22.7 | +22.7 | 0 | 6 | `void deep_gemm::sm90_fp8_gemm_1d1d_impl<4096u, 2048u, 0u, 32u, 128u, 152u, 128u, 128u, ...` |
| 4 | 21.6 | 0.0 | -21.6 | 6 | 0 | `void deep_gemm::sm90_fp8_gemm_1d1d_impl<2048u, 4096u, 0u, 32u, 128u, 152u, 128u, 128u, ...` |
| 5 | 27.4 | 13.4 | -14.0 | 24 | 18 | `void (anonymous namespace)::indexing_backward_kernel<c10::BFloat16, 4>(long const*, lon...` |
| 6 | 31.6 | 36.9 | +5.4 | 70 | 77 | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` |
| 7 | 9.6 | 14.5 | +4.9 | 12 | 18 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(ano...` |
| 8 | 8.1 | 5.2 | -2.8 | 90 | 89 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` |
| 9 | 19.7 | 18.7 | -1.1 | 12 | 12 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 8192u, 1u, 256...` |
| 10 | 43.3 | 44.4 | +1.1 | 14 | 14 | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` |
| 11 | 108.3 | 107.4 | -0.9 | 154 | 154 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::Bin...` |
| 12 | 47.3 | 46.5 | -0.8 | 14 | 14 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT` |
| 13 | 22.5 | 21.7 | -0.8 | 12 | 12 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 32768u, 1024u, 1u, 25...` |
| 14 | 88.8 | 89.6 | +0.8 | 18 | 18 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 4096u, 32u, 12...` |
| 15 | 86.8 | 86.2 | -0.7 | 12 | 12 | `main_kernel` |
| 16 | 45.6 | 45.0 | -0.7 | 4 | 4 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` |
| 17 | 26.0 | 26.5 | +0.6 | 28 | 28 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` |
| 18 | 122.9 | 122.5 | -0.4 | 6 | 6 | `dsv4_sparse_attn_bwd_kernel_kernel` |
| 19 | 25.8 | 26.2 | +0.4 | 36 | 36 | `_grouped_per_token_fp8_kernel` |
| 20 | 36.3 | 36.0 | -0.4 | 28 | 28 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` |
