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

## After (fused in-place RoPE, commit 51db4bbd5)

### Op level

| case | variant | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |
|---|---|---|---|---|---|---|
| main_q | eager | 131072 | 37.873 | 24608.0 | 90.398 | 24608.0 |
| main_q | fused | 131072 | 4.846 | 8192.0 | 15.586 | 16384.0 |
| kv | eager | 131072 | 0.705 | 304.0 | 1.451 | 384.0 |
| kv | fused | 131072 | 0.175 | 128.0 | 0.568 | 256.0 |
| indexer_q | eager | 131072 | 18.086 | 8224.0 | n/a | n/a |
| indexer_q | fused | 131072 | 1.815 | 2048.0 | n/a | n/a |
| attn_output | eager | 131072 | 26.174 | 17440.0 | 68.941 | 24576.0 |
| attn_output | fused | 131072 | 8.838 | 16384.0 | 19.577 | 16384.0 |
| main_q | eager | 32768 | 12.518 | 6152.0 | 25.637 | 6152.0 |
| main_q | fused | 32768 | 1.228 | 2048.0 | 3.913 | 4096.0 |
| kv | eager | 32768 | 0.226 | 76.0 | 0.505 | 96.0 |
| kv | fused | 32768 | 0.115 | 32.0 | 0.383 | 64.0 |
| indexer_q | eager | 32768 | 4.527 | 2056.0 | n/a | n/a |
| indexer_q | fused | 32768 | 0.469 | 512.0 | n/a | n/a |
| attn_output | eager | 32768 | 9.865 | 4360.0 | 20.618 | 6144.0 |
| attn_output | fused | 32768 | 2.228 | 4096.0 | 4.918 | 4096.0 |
| main_q | eager | 16384 | 6.292 | 3076.0 | 12.922 | 3076.0 |
| main_q | fused | 16384 | 0.626 | 1024.0 | 1.975 | 2048.0 |
| kv | eager | 16384 | 0.189 | 38.0 | 0.495 | 48.0 |
| kv | fused | 16384 | 0.122 | 16.0 | 0.393 | 32.0 |
| indexer_q | eager | 16384 | 2.296 | 1028.0 | n/a | n/a |
| indexer_q | fused | 16384 | 0.244 | 256.0 | n/a | n/a |
| attn_output | eager | 16384 | 4.977 | 2180.0 | 10.368 | 3072.0 |
| attn_output | fused | 16384 | 1.130 | 2048.0 | 2.483 | 2048.0 |

### Layer level

| layer | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |
|---|---|---|---|---|---|
| csa | 131072 | 198.923 | 39785.7 | 484.578 | 46345.7 |
| csa | 32768 | 35.252 | 9946.4 | 105.120 | 11634.4 |
| csa | 16384 | 16.627 | 4973.2 | 50.825 | 5849.2 |
| hca | 131072 | 95.540 | 38980.3 | 336.594 | 45699.3 |
| hca | 32768 | 19.536 | 9637.1 | 67.487 | 11436.8 |
| hca | 16384 | 9.614 | 4809.5 | 32.380 | 5747.4 |
| sliding | 131072 | 71.332 | 37697.0 | 236.465 | 44929.0 |
| sliding | 32768 | 17.587 | 9424.2 | 58.607 | 11280.2 |
| sliding | 16384 | 8.785 | 4712.1 | 29.475 | 5672.1 |

## After, q normed and rotated in fp32 (commit 7cfe93796)

### Layer level

| layer | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |
|---|---|---|---|---|---|
| csa | 131072 | 204.804 | 39785.7 | 518.345 | 59936.5 |
| csa | 32768 | 36.763 | 9946.4 | 113.388 | 14984.1 |
| csa | 16384 | 17.428 | 4973.2 | 55.134 | 7492.1 |
| hca | 131072 | 101.514 | 38980.3 | 370.424 | 59936.5 |
| hca | 32768 | 21.367 | 9637.1 | 76.462 | 14984.1 |
| hca | 16384 | 10.459 | 4809.5 | 36.674 | 7492.1 |
| sliding | 131072 | 77.019 | 37697.0 | 272.364 | 59936.5 |
| sliding | 32768 | 19.312 | 9424.2 | 66.080 | 14984.1 |
| sliding | 16384 | 9.516 | 4712.1 | 33.291 | 7492.1 |

## After, fused q norm + RoPE node (commit 46e538b50)

### Layer level

| layer | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |
|---|---|---|---|---|---|
| csa | 131072 | 205.186 | 39785.4 | 469.417 | 46345.4 |
| csa | 32768 | 36.890 | 9946.4 | 99.854 | 11634.4 |
| csa | 16384 | 17.447 | 4973.2 | 49.360 | 5849.2 |
| hca | 131072 | 101.486 | 38980.3 | 320.151 | 45700.3 |
| hca | 32768 | 21.509 | 9637.1 | 64.597 | 11436.8 |
| hca | 16384 | 10.535 | 4810.5 | 30.487 | 5748.4 |
| sliding | 131072 | 77.731 | 37697.0 | 220.389 | 44929.0 |
| sliding | 32768 | 19.557 | 9424.2 | 54.861 | 11280.2 |
| sliding | 16384 | 9.630 | 4712.1 | 27.753 | 5672.1 |
