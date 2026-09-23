# DeepSeek-V4 RoPE benchmark

Produced by `benchmarks/scripts/bench_dsv4_rope.py` on one H200 (torch 2.13.0+cu130, bf16). Times are
medians of 20 CUDA-event timings after 5 warmup iterations. Memory is peak allocated above the
pre-case baseline. Layer rows run one `DeepseekV4Attention` on a row of that many tokens, no CP.

## Before (eager RoPE, commit ae37b35a3)

### Op level

| case | variant | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |
|---|---|---|---|---|---|---|
| main_q | eager | 131072 | 37.841 | 24608.0 | 90.337 | 24608.0 |
| kv | eager | 131072 | 0.699 | 304.0 | 1.441 | 384.0 |
| indexer_q | eager | 131072 | 18.095 | 8224.0 | n/a | n/a |
| attn_output | eager | 131072 | 26.169 | 17440.0 | 68.923 | 24576.0 |
| main_q | eager | 32768 | 12.494 | 6152.0 | 25.622 | 6152.0 |
| kv | eager | 32768 | 0.241 | 76.0 | 0.558 | 96.0 |
| indexer_q | eager | 32768 | 4.532 | 2056.0 | n/a | n/a |
| attn_output | eager | 32768 | 9.863 | 4360.0 | 20.607 | 6144.0 |
| main_q | eager | 16384 | 6.289 | 3076.0 | 12.905 | 3076.0 |
| kv | eager | 16384 | 0.187 | 38.0 | 0.509 | 48.0 |
| indexer_q | eager | 16384 | 2.292 | 1028.0 | n/a | n/a |
| attn_output | eager | 16384 | 4.972 | 2180.0 | 10.391 | 3072.0 |

### Layer level

| layer | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |
|---|---|---|---|---|---|
| csa | 131072 | 268.904 | 48073.4 | 611.822 | 52489.4 |
| csa | 32768 | 58.562 | 12018.4 | 142.082 | 13122.4 |
| csa | 16384 | 28.514 | 6009.2 | 69.792 | 6561.2 |
| hca | 131072 | 145.329 | 47268.3 | 441.190 | 51844.3 |
| hca | 32768 | 38.332 | 11710.1 | 99.804 | 12925.8 |
| hca | 16384 | 18.951 | 5846.5 | 48.407 | 6460.4 |
| sliding | 131072 | 121.925 | 45985.0 | 346.447 | 51073.0 |
| sliding | 32768 | 35.742 | 11496.2 | 90.337 | 12768.2 |
| sliding | 16384 | 17.972 | 5748.1 | 45.086 | 6384.1 |
