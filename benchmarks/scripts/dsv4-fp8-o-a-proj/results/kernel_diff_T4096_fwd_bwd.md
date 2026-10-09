

# Kernel time diff (compute): bf16 bmm -> fp8 einsum

- summed kernel time: 16.1 ms -> 13.3 ms (-2.8 ms)
- before: `/home/garrett/tmp/profiling/dsv4-fp8-o-a-proj/derived/T=4096_fwd+bwd__bf16_bmm.csv`
- after: `/home/garrett/tmp/profiling/dsv4-fp8-o-a-proj/derived/T=4096_fwd+bwd__fp8_einsum.csv`

| # | bf16 bmm ms | fp8 einsum ms | delta ms | calls before | calls after | kernel |
|---|---|---|---|---|---|---|
| 1 | 5.5 | 0.0 | -5.5 | 20 | 0 | `void at::native::elementwise_kernel<128, 4, at::native::g...` |
| 2 | 0.0 | 3.6 | +3.6 | 0 | 80 | `void deep_gemm::sm90_fp8_gemm_1d1d_impl<0u, 4096u, 4096u,...` |
| 3 | 3.6 | 0.0 | -3.6 | 10 | 0 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT` |
| 4 | 0.0 | 3.6 | +3.6 | 0 | 40 | `_per_token_fp8_kernel` |
| 5 | 3.5 | 0.0 | -3.5 | 10 | 0 | `nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN` |
| 6 | 3.5 | 0.0 | -3.5 | 10 | 0 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` |
| 7 | 0.0 | 2.3 | +2.3 | 0 | 10 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Majo...` |
| 8 | 0.0 | 2.0 | +2.0 | 0 | 10 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Majo...` |
| 9 | 0.0 | 0.5 | +0.5 | 0 | 20 | `_grouped_per_block_fp8_kernel` |
| 10 | 0.0 | 0.5 | +0.5 | 0 | 10 | `void at::native::vectorized_elementwise_kernel<8, at::nat...` |
| 11 | 0.0 | 0.3 | +0.3 | 0 | 160 | `void deep_gemm::transpose_fp32<512u, 64u, 32u, 33u>(float...` |
| 12 | 0.0 | 0.3 | +0.3 | 0 | 10 | `void at::native::vectorized_elementwise_kernel<4, at::nat...` |
| 13 | 0.0 | 0.1 | +0.1 | 0 | 20 | `void at::native::elementwise_kernel<128, 2, at::native::g...` |
| 14 | 0.0 | 0.0 | -0.0 | 30 | 0 | `Memset (Device)` |

| category | bf16 bmm ms | fp8 einsum ms | delta ms |
|---|---|---|---|
| GEMM | 10.6 | 8.0 | -2.6 |
| FP8 cast | 0.0 | 4.5 | +4.5 |
| copy and fill | 5.5 | 0.9 | -4.6 |
| other | 0.0 | 0.0 | +0.0 |
