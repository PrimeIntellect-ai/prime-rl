Dynamic stream of 32 items, each called once; times in seconds, lower is better.
`compiles` and `loads` come from the compile-entry-point wrapper; `new files` from the cache dirs.

`first item` carries the per-process costs; `rest` sums the other items' first calls.

| backend | phase | first item s | rest s | process wall s | compiles | loads | new cache files |
|---|---|---|---|---|---|---|---|
| tilelang | cold | 20.73 | 0.615 | 72.5 | 4 | 0 | 21 |
| tilelang | warm | 0.28 | 0.616 | 52.6 | 0 | 4 | 0 |
| cudnn_flashmla | cold | 6.98 | 7.968 | 67.6 | 7 | 0 | 6 |
| cudnn_flashmla | warm | 3.45 | 7.940 | 63.1 | 6 | 1 | 0 |
| cute | cold | 14.03 | 0.586 | 66.9 | 4 | 0 | 16 |
| cute | warm | 1.44 | 0.583 | 53.6 | 1 | 3 | 0 |
| cute_ws | cold | 15.72 | 0.512 | 68.4 | 4 | 0 | 16 |
| cute_ws | warm | 3.16 | 0.512 | 55.6 | 1 | 3 | 0 |

Per-process startup breakdown in seconds (lower is better).

| backend | phase | stage | s |
|---|---|---|---|
| tilelang | cold | interpreter_and_torch_import | 3.22 |
| tilelang | cold | import tilelang | 2.05 |
| tilelang | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.47 |
| tilelang | cold | cuda_context | 0.18 |
| tilelang | cold | first_item_jit_compile | 15.86 |
| tilelang | cold | first_item_jit_disk_load | 0.00 |
| tilelang | cold | first_item_jit_other | 2.51 |
| tilelang | cold | first_item_launch_and_run | 2.36 |
| tilelang | warm | interpreter_and_torch_import | 3.91 |
| tilelang | warm | import tilelang | 2.35 |
| tilelang | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.01 |
| tilelang | warm | cuda_context | 0.16 |
| tilelang | warm | first_item_jit_compile | 0.00 |
| tilelang | warm | first_item_jit_disk_load | 0.03 |
| tilelang | warm | first_item_jit_other | 0.14 |
| tilelang | warm | first_item_launch_and_run | 0.11 |
| cudnn_flashmla | cold | interpreter_and_torch_import | 3.71 |
| cudnn_flashmla | cold | import flash_mla | 0.01 |
| cudnn_flashmla | cold | import cutlass.cute | 0.52 |
| cudnn_flashmla | cold | import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 | 0.07 |
| cudnn_flashmla | cold | import tilelang | 2.46 |
| cudnn_flashmla | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.51 |
| cudnn_flashmla | cold | cuda_context | 0.14 |
| cudnn_flashmla | cold | first_item_jit_compile | 6.03 |
| cudnn_flashmla | cold | first_item_jit_disk_load | 0.00 |
| cudnn_flashmla | cold | first_item_jit_other | 0.46 |
| cudnn_flashmla | cold | first_item_launch_and_run | 0.49 |
| cudnn_flashmla | warm | interpreter_and_torch_import | 3.75 |
| cudnn_flashmla | warm | import flash_mla | 0.01 |
| cudnn_flashmla | warm | import cutlass.cute | 0.60 |
| cudnn_flashmla | warm | import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 | 0.09 |
| cudnn_flashmla | warm | import tilelang | 2.42 |
| cudnn_flashmla | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 40.51 |
| cudnn_flashmla | warm | cuda_context | 0.21 |
| cudnn_flashmla | warm | first_item_jit_compile | 3.30 |
| cudnn_flashmla | warm | first_item_jit_disk_load | 0.01 |
| cudnn_flashmla | warm | first_item_jit_other | 0.04 |
| cudnn_flashmla | warm | first_item_launch_and_run | 0.10 |
| cute | cold | interpreter_and_torch_import | 3.68 |
| cute | cold | import tilelang | 2.45 |
| cute | cold | import cutlass.cute | 0.68 |
| cute | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.07 |
| cute | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute | 0.00 |
| cute | cold | cuda_context | 0.20 |
| cute | cold | first_item_jit_compile | 9.46 |
| cute | cold | first_item_jit_disk_load | 0.00 |
| cute | cold | first_item_jit_other | 1.71 |
| cute | cold | first_item_launch_and_run | 2.87 |
| cute | warm | interpreter_and_torch_import | 3.66 |
| cute | warm | import tilelang | 2.43 |
| cute | warm | import cutlass.cute | 0.74 |
| cute | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 40.53 |
| cute | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute | 0.00 |
| cute | warm | cuda_context | 0.15 |
| cute | warm | first_item_jit_compile | 0.00 |
| cute | warm | first_item_jit_disk_load | 0.02 |
| cute | warm | first_item_jit_other | 0.08 |
| cute | warm | first_item_launch_and_run | 1.34 |
| cute_ws | cold | interpreter_and_torch_import | 3.72 |
| cute_ws | cold | import tilelang | 2.37 |
| cute_ws | cold | import cutlass.cute | 0.66 |
| cute_ws | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 40.94 |
| cute_ws | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute_ws | 0.00 |
| cute_ws | cold | cuda_context | 0.20 |
| cute_ws | cold | first_item_jit_compile | 9.45 |
| cute_ws | cold | first_item_jit_disk_load | 0.00 |
| cute_ws | cold | first_item_jit_other | 1.70 |
| cute_ws | cold | first_item_launch_and_run | 4.57 |
| cute_ws | warm | interpreter_and_torch_import | 3.86 |
| cute_ws | warm | import tilelang | 2.39 |
| cute_ws | warm | import cutlass.cute | 0.76 |
| cute_ws | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 40.39 |
| cute_ws | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute_ws | 0.00 |
| cute_ws | warm | cuda_context | 0.15 |
| cute_ws | warm | first_item_jit_compile | 0.00 |
| cute_ws | warm | first_item_jit_disk_load | 0.02 |
| cute_ws | warm | first_item_jit_other | 0.09 |
| cute_ws | warm | first_item_launch_and_run | 3.05 |

Per-item first-call time in ms (lower is better) and compiles triggered by that item.

| # | item | backend | cold ms | cold compiles | warm ms | warm compiles |
|---|---|---|---|---|---|---|
| 0 | stream00-short-14224-csa-cp8r1 | tilelang | 20734.2 | 4 | 278.8 | 0 |
| 0 | stream00-short-14224-csa-cp8r1 | cudnn_flashmla | 6975.5 | 4 | 3453.1 | 3 |
| 0 | stream00-short-14224-csa-cp8r1 | cute | 14033.7 | 4 | 1443.4 | 1 |
| 0 | stream00-short-14224-csa-cp8r1 | cute_ws | 15724.1 | 4 | 3157.2 | 1 |
| 1 | stream01-tiny-48096-hca-cp1 | tilelang | 17.4 | 0 | 18.2 | 0 |
| 1 | stream01-tiny-48096-hca-cp1 | cudnn_flashmla | 2513.1 | 1 | 2477.5 | 1 |
| 1 | stream01-tiny-48096-hca-cp1 | cute | 16.2 | 0 | 16.1 | 0 |
| 1 | stream01-tiny-48096-hca-cp1 | cute_ws | 15.7 | 0 | 13.3 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | tilelang | 9.6 | 0 | 9.6 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | cudnn_flashmla | 2587.1 | 1 | 2507.8 | 1 |
| 2 | stream02-heavy-38136-hca-cp8r6 | cute | 7.4 | 0 | 7.1 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | cute_ws | 10.6 | 0 | 11.8 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | tilelang | 3.4 | 0 | 3.2 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | cudnn_flashmla | 3.3 | 0 | 2.8 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | cute | 2.7 | 0 | 2.6 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | cute_ws | 2.7 | 0 | 2.8 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | tilelang | 5.2 | 0 | 5.1 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | cudnn_flashmla | 3.1 | 0 | 3.2 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | cute | 4.5 | 0 | 4.5 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | cute_ws | 4.1 | 0 | 4.4 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | tilelang | 5.6 | 0 | 5.8 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | cudnn_flashmla | 3.4 | 0 | 67.5 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | cute | 5.1 | 0 | 5.0 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | cute_ws | 5.0 | 0 | 4.9 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | tilelang | 42.4 | 0 | 42.6 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | cudnn_flashmla | 25.4 | 0 | 25.2 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | cute | 41.3 | 0 | 41.1 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | cute_ws | 35.7 | 0 | 35.8 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | tilelang | 3.3 | 0 | 3.6 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | cudnn_flashmla | 2.1 | 0 | 2.2 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | cute | 2.9 | 0 | 2.7 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | cute_ws | 2.7 | 0 | 2.8 | 0 |
| 8 | stream08-single-28000-hca-cp1 | tilelang | 20.6 | 0 | 20.6 | 0 |
| 8 | stream08-single-28000-hca-cp1 | cudnn_flashmla | 2527.1 | 1 | 2551.0 | 1 |
| 8 | stream08-single-28000-hca-cp1 | cute | 19.6 | 0 | 19.6 | 0 |
| 8 | stream08-single-28000-hca-cp1 | cute_ws | 16.7 | 0 | 16.8 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | tilelang | 11.9 | 0 | 11.8 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | cudnn_flashmla | 9.2 | 0 | 9.0 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | cute | 10.9 | 0 | 10.9 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | cute_ws | 8.9 | 0 | 9.0 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | tilelang | 52.7 | 0 | 52.6 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | cudnn_flashmla | 28.9 | 0 | 29.0 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | cute | 51.5 | 0 | 51.4 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | cute_ws | 45.2 | 0 | 45.3 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | tilelang | 5.1 | 0 | 5.1 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | cudnn_flashmla | 3.7 | 0 | 3.8 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | cute | 4.4 | 0 | 4.4 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | cute_ws | 3.6 | 0 | 3.6 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | tilelang | 13.8 | 0 | 13.8 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | cudnn_flashmla | 10.4 | 0 | 10.3 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | cute | 12.7 | 0 | 12.7 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | cute_ws | 10.4 | 0 | 10.5 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | tilelang | 12.7 | 0 | 12.7 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | cudnn_flashmla | 6.7 | 0 | 6.7 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | cute | 12.0 | 0 | 12.0 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | cute_ws | 10.6 | 0 | 10.6 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | tilelang | 14.2 | 0 | 14.2 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | cudnn_flashmla | 8.0 | 0 | 8.0 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | cute | 13.4 | 0 | 13.4 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | cute_ws | 11.6 | 0 | 11.7 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | tilelang | 22.0 | 0 | 22.1 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | cudnn_flashmla | 14.8 | 0 | 14.8 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | cute | 20.9 | 0 | 20.9 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | cute_ws | 17.1 | 0 | 17.0 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | tilelang | 22.8 | 0 | 22.8 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | cudnn_flashmla | 17.5 | 0 | 17.6 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | cute | 21.6 | 0 | 21.5 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | cute_ws | 17.6 | 0 | 18.0 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | tilelang | 2.3 | 0 | 2.4 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | cudnn_flashmla | 1.6 | 0 | 1.6 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | cute | 1.8 | 0 | 1.7 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | cute_ws | 1.6 | 0 | 1.5 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | tilelang | 11.9 | 0 | 11.9 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | cudnn_flashmla | 7.8 | 0 | 7.8 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | cute | 11.0 | 0 | 11.0 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | cute_ws | 9.0 | 0 | 9.0 | 0 |
| 19 | stream19-single-36016-hca-cp1 | tilelang | 28.2 | 0 | 28.3 | 0 |
| 19 | stream19-single-36016-hca-cp1 | cudnn_flashmla | 16.5 | 0 | 16.6 | 0 |
| 19 | stream19-single-36016-hca-cp1 | cute | 27.2 | 0 | 27.1 | 0 |
| 19 | stream19-single-36016-hca-cp1 | cute_ws | 23.5 | 0 | 23.4 | 0 |
| 20 | stream20-short-24768-hca-cp1 | tilelang | 13.3 | 0 | 13.3 | 0 |
| 20 | stream20-short-24768-hca-cp1 | cudnn_flashmla | 8.8 | 0 | 8.8 | 0 |
| 20 | stream20-short-24768-hca-cp1 | cute | 12.5 | 0 | 12.4 | 0 |
| 20 | stream20-short-24768-hca-cp1 | cute_ws | 10.5 | 0 | 10.5 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | tilelang | 17.9 | 0 | 18.0 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | cudnn_flashmla | 12.0 | 0 | 12.0 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | cute | 16.9 | 0 | 16.9 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | cute_ws | 13.7 | 0 | 13.8 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | tilelang | 28.0 | 0 | 28.1 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | cudnn_flashmla | 18.8 | 0 | 18.8 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | cute | 26.8 | 0 | 26.9 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | cute_ws | 22.0 | 0 | 22.2 | 0 |
| 23 | stream23-single-61784-csa-cp1 | tilelang | 90.7 | 0 | 90.4 | 0 |
| 23 | stream23-single-61784-csa-cp1 | cudnn_flashmla | 47.2 | 0 | 47.3 | 0 |
| 23 | stream23-single-61784-csa-cp1 | cute | 88.8 | 0 | 88.8 | 0 |
| 23 | stream23-single-61784-csa-cp1 | cute_ws | 79.1 | 0 | 79.1 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | tilelang | 2.9 | 0 | 2.5 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | cudnn_flashmla | 1.9 | 0 | 1.6 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | cute | 2.7 | 0 | 1.9 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | cute_ws | 2.0 | 0 | 1.9 | 0 |
| 25 | stream25-single-56616-hca-cp1 | tilelang | 52.5 | 0 | 52.7 | 0 |
| 25 | stream25-single-56616-hca-cp1 | cudnn_flashmla | 30.0 | 0 | 30.0 | 0 |
| 25 | stream25-single-56616-hca-cp1 | cute | 51.2 | 0 | 51.2 | 0 |
| 25 | stream25-single-56616-hca-cp1 | cute_ws | 44.6 | 0 | 44.6 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | tilelang | 3.6 | 0 | 3.6 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | cudnn_flashmla | 2.8 | 0 | 2.6 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | cute | 2.8 | 0 | 2.8 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | cute_ws | 2.7 | 0 | 2.7 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | tilelang | 3.6 | 0 | 3.6 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | cudnn_flashmla | 2.5 | 0 | 2.7 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | cute | 2.9 | 0 | 2.9 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | cute_ws | 2.4 | 0 | 2.4 | 0 |
| 28 | stream28-single-42136-csa-cp1 | tilelang | 60.8 | 0 | 60.9 | 0 |
| 28 | stream28-single-42136-csa-cp1 | cudnn_flashmla | 31.8 | 0 | 31.9 | 0 |
| 28 | stream28-single-42136-csa-cp1 | cute | 59.6 | 0 | 59.6 | 0 |
| 28 | stream28-single-42136-csa-cp1 | cute_ws | 52.8 | 0 | 52.8 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | tilelang | 2.5 | 0 | 2.5 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | cudnn_flashmla | 1.9 | 0 | 1.9 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | cute | 2.0 | 0 | 2.1 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | cute_ws | 2.0 | 0 | 1.9 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | tilelang | 31.1 | 0 | 31.3 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | cudnn_flashmla | 18.2 | 0 | 18.1 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | cute | 30.1 | 0 | 30.0 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | cute_ws | 26.2 | 0 | 26.1 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | tilelang | 2.7 | 0 | 2.7 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | cudnn_flashmla | 2.0 | 0 | 1.9 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | cute | 2.2 | 0 | 2.3 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | cute_ws | 2.2 | 0 | 1.9 | 0 |
